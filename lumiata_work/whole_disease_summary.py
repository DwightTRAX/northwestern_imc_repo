#!/usr/bin/env python
import time
from lum_model_pipeline.config import Config
from lumiata.utils import file_exists, date_to_string, load_file, save_npz, write_file
import os
import sys
import json
from pprint import pprint
import argparse
from dateutil.relativedelta import relativedelta
from tensorflow.python.lib.io import file_io
from scipy import sparse
from sklearn.metrics import *
from scipy.stats import pearsonr
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import rc

import shap
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from pyspark.sql import SQLContext, SparkSession

from multiprocessing import Pool

def _load_segment(source, file_type):
    if file_type == "bq":
        seg = Segment.load_from_bq(source)
    else:
        schema = LDM_schema()
        seg = Segment.load_from_file(segment_location=source,
                                     file_type=file_type, schema=schema)
    return seg

def _load_sparse_matrix(data_dict):
    return sparse.csr_matrix(
        (data_dict["data"], data_dict["indices"], data_dict["indptr"]),
        shape=data_dict["_shape"]
    )

def _load_dataset(path):
    data_dict = load_file(path)
    data_dict["ids"] = data_dict["ids"].reshape(-1)
    sparse_mat = _load_sparse_matrix(data_dict)
    data_dict["matrix"] = sparse_mat
    del data_dict["data"], data_dict["indices"], data_dict["indptr"],         data_dict["_shape"], data_dict["maxprint"]
    return data_dict

def _get_prior_cost_index(feature_names, right_censor):
    right = right_censor
    left = str(to_datetime(right_censor).date() - relativedelta(years=1))
    col = f"Claims|{left}|{right}|AllowedCost"
    where = np.where(feature_names == col)[0]
    return where[0] if len(where) else None

def get_group_id_map(segment_location, slice_date):
    # the input segment was already sliced on the slice date
    seg = _load_segment(source=segment_location, file_type="parquet")
    member_to_group_map = (
        seg.df
            .select("MBR", fn.explode("coverage").alias("coverage"))
            .select("MBR", "coverage.*")
            .select("MBR", "group_number", "start_date", "end_date")
            .where((fn.col("start_date") <= slice_date) &
                   (slice_date <= fn.col("end_date")))
            .select("MBR", fn.col("group_number"))
            .distinct()
            .groupBy("MBR")
            .agg(fn.first("group_number").alias("GROUP"))
            .rdd
            .map(lambda row: (row.MBR, row.GROUP))
    ).collectAsMap()
    return member_to_group_map

def _get_top_k_shap_df_ind(ind_shap_path, k):
    ind_dd = load_file(ind_shap_path)
    ind_mat = _load_sparse_matrix(ind_dd).toarray()
    ind_feats = ind_dd['feature_names']
    ind_ids = list(ind_dd['ids'])
#     group_mem_indices = [ind_ids.index(mem) for mem in
#                          membership_map[group]]
    top_shaps_dict = {}
    for mem_index in group_mem_indices:
        if mbr_to_match[ind_ids[mem_index]]:
            mem_shaps = ind_mat[mem_index]
            sorted_indices = np.argsort(abs(mem_shaps))[::-1]
            # Remove bias index
            bias_index = np.where(ind_feats == 'Bias')[0][0]
            sorted_indices = sorted_indices[sorted_indices != bias_index]
            feats = ind_feats[sorted_indices[:k]]
            shaps = np.around(mem_shaps[sorted_indices[:k]], decimals=2)
            one_line_output = [item for pair in zip(feats, shaps) for item
                               in pair]
            one_line_output.insert(0, mem_shaps[bias_index])
            one_line_output.insert(0, ind_ids[mem_index])
            top_shaps_dict[ind_ids[mem_index]] = one_line_output
        # If not a match, return all N/A
        else:
            one_line_output = ['N/A' for i in range(1 + 2 * k)]
            one_line_output.insert(0, ind_ids[mem_index])
            top_shaps_dict[ind_ids[mem_index]] = one_line_output
    # Get column headers
    feat_heads = [f"Feature_{i}" for i in range(1, k + 1)]
    shap_heads = [f"Shap_Value_{i}" for i in range(1, k + 1)]
    headers = [item for pair in zip(feat_heads, shap_heads) for item in
               pair]
    headers.insert(0, 'Intercept')
    headers.insert(0, 'Member_ID')
    ind_shap_df = pd.DataFrame.from_dict(top_shaps_dict, orient='index',
                                         columns=headers).reset_index(
        drop=True)
    return ind_shap_df

def _get_agg_features(col_names, summary=False):
    # TODO: need to come up with new method to use ai-studio to load
    #   these maps in every environment
    all_codes_map = load_file(
        "gs://data-science-experiments-dev-env-0884/"
        "codes_maps/all_displays.json")
    short_name_to_system_url = load_file(
        "gs://data-science-experiments-dev-env-0884/codes_maps/"
        "short_name_to_system_url.json")

    original_to_agg_feats = dict()
    for feat in col_names:
        feat_type = feat.split('|')[0]
        new_feat = feat
        if feat_type in {'Claims', 'Coverage', 'Claims_Coverage',
                         'Demographic'}:
            new_feat = feat.split('|')[-1]
        elif feat_type == 'MedicalCodeCounter':
            system = feat.split('|')[-1].split('_')[1]
            new_feat = system + '|' + feat.split('|')[-1].split('-')[-1]
        elif feat_type == 'DiseaseTagCounter':
            new_feat = feat.split('|')[-1].split("_")[1]
        elif feat_type == 'CrossCodeCounter':
            system = feat.split('|')[-1].split('_')[1]
            code = feat.split('|')[-1].split('_')[-1].split('-')[-1]
            long_url = short_name_to_system_url[system.lower()]
            if code in all_codes_map[long_url]:
                ip_name = all_codes_map[long_url][code]
                new_feat = system + '|' + ip_name
        elif feat_type == 'LabInterpreter':
            new_feat = feat_type + '|' + feat.split('|')[-1]
        elif feat_type == 'Summary':
            if summary:
                new_feat = feat_type + '|' + feat.split('|')[-1]
            else:
                continue
        original_to_agg_feats[feat] = new_feat
    return original_to_agg_feats

             
def _get_agg_shaps(shap_matrix, mbrs, preds, feature_names,
                   original_to_agg_feats, use_buckets=True):
    features = []
    path = 'gs://data-science-experiments-dev-env-0884/wellmark_deliveries/shaps/agg_features_to_buckets.json'
    agg_feats_to_bucket = load_file(path)
    print('\nstart loop')
    for idx, (pred, mbr) in enumerate(zip(preds, mbrs)):
#         print(f'mbr {idx}')
        shaps = shap_matrix[idx, :].toarray().reshape(-1)
        bias = round(pred - shaps.sum(), 2)

        # create aggregated shap values
        agg_shaps = defaultdict(float)
#         print('looping through feat,val adding up aggs')

        for feat, val in zip(feature_names, shaps):
            if feat in original_to_agg_feats:
                agg_feat = original_to_agg_feats[feat]
                if agg_feat in agg_feats_to_bucket and use_buckets:
                    agg_shaps[agg_feats_to_bucket[agg_feat]] += val
                else: # if agg feat not in a bucket
                    agg_shaps[agg_feat] += val
            else: # if feat not in an agg_feat
                agg_shaps[feat] += val
            
        agg_shaps["Bias"] = bias
        features.append(dict(agg_shaps))
    print('vectorizer')
    v = DictVectorizer(sparse=True)
    print('transform')
    matrix = v.fit_transform(features)
    print('the rest')
    shaps_dict = {"ids": mbrs, "feature_names": v.get_feature_names()}
    shaps_dict.update(matrix.__dict__)
    return shaps_dict

def _agg_shaps_to_buckets(shaps_dict):
#     dict_keys(['ids', 'feature_names', '_shape', 'maxprint', 'indices', 'indptr', 'data', '_has_sorted_indices'])
    path = 'gs://data-science-experiments-dev-env-0884/wellmark_deliveries/shaps/agg_features_to_buckets.json'
    agg_feats_to_bucket = load_file(path)
#     mbrs = 
    
    

def get_shaps(model_path, dataset_paths_dict, pred_paths_dict,
              output_paths_dict, groups=None):
    model_dict = load_file(model_path)
    model = model_dict["model"].model
    feature_selection_indices = model_dict["feature_selection_indices"]
    for split_name, path in dataset_paths_dict.items():
        data = _load_dataset(path)
        matrix = data["matrix"][:, feature_selection_indices]
        feature_names = data["feature_names"][feature_selection_indices]
        pred_map = load_file(pred_paths_dict[split_name])
        mbrs = data["ids"]
        preds = np.array([pred_map[m] for m in mbrs])
        print(f"num members = {len(mbrs)}")
        # If list of groups is passed, only calculate shaps for those groups
        if groups:
#             np.random.seed(self.seed)
            indices = np.random.choice(len(mbrs), 1000, replace=False)
            # indices = np.argsort(preds)[::-1][:1000]
            mbrs = mbrs[indices]
            matrix = matrix[indices, :]
            preds = preds[indices]

        shap_matrix = shap.TreeExplainer(model).shap_values(matrix)
        original_to_agg_feats = _get_agg_features(
            feature_names, summary=True)
        shap_dict = _get_agg_shaps(
            shap_matrix, mbrs, preds, feature_names, original_to_agg_feats
        )
        output_path = output_paths_dict[split_name]
        print(f"writing shap values to {output_path}")
        save_npz(output_path, **shap_dict)

    return output_paths_dict


config = Config.from_file('gs://data-science-experiments-dev-env-0884/configs/wellmark/PIL-904.json')
dataset_name = 'wm'
model_name = 'ind_pmpm_model'
split_name = 'full'
data_config = config.data_config
modeling_config = config.modeling_config
dataset_paths_dict = modeling_config.get_dataset_split_path_dict(
    dataset_name)
pred_paths_dict = {
    split_name: modeling_config.get_prediction_split_path(
        model_name, dataset_name, split_name)
    for split_name in dataset_paths_dict
}
#load group_to_mems
membership_path = data_config.get_group_membership_path(dataset_name)
group_membership = load_file(membership_path)
model_path = modeling_config.get_model_path(model_name)
model_dict = load_file(model_path)
model = model_dict['model'].model
feature_selection_indices = model_dict["feature_selection_indices"]
split_name = 'full'
path = dataset_paths_dict[split_name]
data = _load_dataset(path)
matrix = data["matrix"]
feature_names = data["feature_names"]
original_to_agg_feats = _get_agg_features(
        feature_names, summary=True)
agg_lst_map = list(original_to_agg_feats.items())

# the highest state KY, 0.00512
#disease_name = 'heart_disease'
cancer_set = {
 'HCUP|Cancer of bladder',
 'HCUP|Cancer of bone and connective tissue',
 'HCUP|Cancer of brain and nervous system',
 'HCUP|Cancer of breast',
 'HCUP|Cancer of bronchus; lung',
 'HCUP|Cancer of cervix', # high prevalence 0.014
 'HCUP|Cancer of colon',
 'HCUP|Cancer of esophagus',
 'HCUP|Cancer of head and neck',
 'HCUP|Cancer of kidney and renal pelvis',
 'HCUP|Cancer of liver and intrahepatic bile duct',
 'HCUP|Cancer of other GI organs; peritoneum',
 'HCUP|Cancer of other female genital organs',
 'HCUP|Cancer of ovary',
 #'HCUP|Cancer of prostate',
 'HCUP|Cancer of rectum and anus',
 'HCUP|Cancer of testis',
 'HCUP|Cancer of thyroid',
 'HCUP|Cancer of uterus',
 'HCUP|Cancer; other and unspecified primary',
 'HCUP|Leukemias',
 'HCUP|Maintenance chemotherapy; radiotherapy',
 'HCUP|Malignant neoplasm without specification of site',
 'HCUP|Melanomas of skin',
 'HCUP|Multiple myeloma',
 #'HCUP|Nonmalignant breast conditions', # high prevalence Very common More than 3 million US cases per year
 #'HCUP|Other non-epithelial cancer of skin', # high prevalence > 0.02
 'HCUP|Secondary malignancies'
}

heart_disease_set = {
  'HCUP|Acute myocardial infarction',
 'HCUP|Aortic; peripheral; and visceral artery aneurysms',
 'HCUP|Cardiac and circulatory congenital anomalies',
 'HCUP|Cardiac arrest and ventricular fibrillation',
 'HCUP|Cardiac dysrhythmias',
 'HCUP|Congestive heart failure; nonhypertensive',
 'HCUP|Coronary atherosclerosis and other heart disease',
 'HCUP|Essential hypertension',
 'HCUP|Heart valve disorders',
 'HCUP|Hypertension complicating pregnancy; childbirth and the puerperium',
 'HCUP|Hypertension with complications and secondary hypertension',
 'HCUP|Other and ill-defined heart disease',
 'HCUP|Peri-; endo-; and myocarditis; cardiomyopathy (except that caused by tuberculosis or sexually transmitted disease)',
 'HCUP|Pulmonary heart disease'
        }

kidney_disease_set = {
 'HCUP|Acute and unspecified renal failure',
 'HCUP|Cancer of kidney and renal pelvis',
 'HCUP|Chronic kidney disease',
 'HCUP|Nephritis; nephrosis; renal sclerosis',
 'HCUP|Other diseases of kidney and ureters'
}

lung_disease_set = {
 'HCUP|Acute bronchitis',
 'HCUP|Asthma',
 'HCUP|Cancer of bronchus; lung',
 'HCUP|Chronic obstructive pulmonary disease and bronchiectasis',
 'HCUP|Other lower respiratory disease',
 'HCUP|Other upper respiratory disease',
 'HCUP|Other upper respiratory infections',
 'HCUP|Pleurisy; pneumothorax; pulmonary collapse',
 'HCUP|Pulmonary heart disease',
 'HCUP|Respiratory distress syndrome',
 'HCUP|Respiratory failure; insufficiency; arrest (adult)'
}

other_chronic_disease_set ={
 'HCUP|Blindness and vision defects',
 'HCUP|Epilepsy; convulsions',
 'HCUP|Infective arthritis and osteomyelitis (except that caused by tuberculosis or sexually transmitted disease)',
 'HCUP|Osteoarthritis',
 'HCUP|Rheumatoid arthritis and related disease'
}
edema_set = {'ATC|furosemide',
 'ATC|torasemide',
 'ICD10|R60.0',
 'ICD10|R60.1',
 'ICD10|R60.9',
 'RXNORM|Furosemide 20 MG Oral Tablet',
 'RXNORM|Furosemide 40 MG Oral Tablet',
 'RXNORM|Furosemide 80 MG Oral Tablet'}
 
substance_abuse_set = {
 'ATC|Drugs used in alcohol dependence',
 'ATC|Drugs used in nicotine dependence',
 'ATC|Drugs used in opioid dependence',
 'AlcoholDependencySyndromeCounter',
 'HCUP|Alcohol-related disorders',
 'HCUP|Substance-related disorders',
 'ICD10|K74.0',
 'ICD10|K74.60',
 'ICD10|K74.69'
}

gastrointestinal_set = {
'ATC|Belladonna alkaloids, tertiary amines for functional gastrointestinal disorders',
 'ATC|Enzyme preparations, digestives',
 'ATC|Other drugs for functional gastrointestinal disorders in ATC',
 'ATC|Other drugs for peptic ulcer and gastro-oesophageal reflux disease (GORD) in ATC',
 'ATC|Synthetic anticholinergics, esters with tertiary amino group for functional gastrointestinal disorders',
 'HCUP|Digestive congenital anomalies',
 'HCUP|Gastroduodenal ulcer (except hemorrhage)',
 'HCUP|Gastrointestinal hemorrhage',
 'HCUP|Noninfectious gastroenteritis',
 'HCUP|Other gastrointestinal disorders',
 'HCUP|Regional enteritis and ulcerative colitis'
}

psychiatric_set = {
'ATC|Diazepines, oxazepines, thiazepines and oxepines antipsychotic drugs',
 'ATC|Indole derivatives, antipsychotic',
 'ATC|Other antidepressants in ATC',
 'ATC|Other antipsychotics in ATC',
 'ATC|Phenothiazines with piperazine structure, antipsychotics',
 'ATC|Xanthine derivatives, psychostimulants, agents used for ADHD and nootropics',
 'ATC|prochlorperazine',
 'DepressionCounter',
 'HCUP|Mood disorders',
 'HCUP|Poisoning by psychotropic agents',
 'HCUP|Schizophrenia and other psychotic disorders',
 'HCUP|Suicide and intentional self-inflicted injury',
 'RXNORM|Prochlorperazine 10 MG Oral Tablet',
 'RXNORM|Prochlorperazine 5 MG Oral Tablet'
}

injury_set = {
'BackSurgeryCounter',
 'HCUP|Crushing injury or internal injury',
 'HCUP|Fracture of lower limb',
 'HCUP|Fracture of neck of femur (hip)',
 'HCUP|Fracture of upper limb',
 'HCUP|Intracranial injury',
 'HCUP|Joint disorders and dislocations; trauma-related',
 'HCUP|OB-related trauma to perineum and vulva',
 'HCUP|Other fractures',
 'HCUP|Other non-traumatic joint disorders',
 'HCUP|Pathological fracture',
 'HCUP|Skull and face fractures',
 'HCUP|Spinal cord injury',
 'HCUP|Spondylosis; intervertebral disc disorders; other back problems',
 'HCUP|Suicide and intentional self-inflicted injury',
 'HCUP|Superficial injury; contusion'
}

pain_management_set = {
 'ATC|Drugs used in opioid dependence',
 'ATC|OPIOID ANALGESICS',
 'ATC|Other opioids in ATC',
 'ATC|Peripheral opioid receptor antagonists',
 'ATC|Phenylpiperidine derivatives, opioid analgesics',
 'ATC|fentanyl',
 'HCUP|Abdominal pain',
 'HCUP|Nonspecific chest pain',
 'RXNORM|fentaNYL 25 MCG/HR 3 Day Transdermal Patch',
 'RXNORM|fentaNYL 50 MCG/HR 3 Day Transdermal Patch'
}

pregnancy_set = {
'HCUP|Birth trauma',
 'HCUP|Diabetes or abnormal glucose tolerance complicating pregnancy; childbirth; or the puerperium',
 'HCUP|Ectopic pregnancy',
 'HCUP|Female infertility',
 'HCUP|Hemorrhage during pregnancy; abruptio placenta; placenta previa',
 'HCUP|Hypertension complicating pregnancy; childbirth and the puerperium',
 'HCUP|Intrauterine hypoxia and birth asphyxia',
 'HCUP|Menstrual disorders',
 'HCUP|Other complications of birth; puerperium affecting management of mother',
 'HCUP|Other complications of pregnancy',
 'HCUP|Other pregnancy and delivery including normal',
 'HCUP|Prolonged pregnancy',
 'HCUP|Short gestation; low birth weight; and fetal growth retardation',
 'ICD10|Z32.00',
 'ICD10|Z32.01',
 'ICD10|Z32.02'
}
disease_name_lst = ['cancer', 'heart_disease', 'kidney_disease', 'lung_disease', 'other_chronic_disease',
'edema', 'substance_abuse', 'gastrointestinal', 'psychiatric', 'injury', 'pain_management', 'pregnancy']
some_set_lst = [cancer_set, heart_disease_set, kidney_disease_set, lung_disease_set, other_chronic_disease_set,
edema_set, substance_abuse_set, gastrointestinal_set, psychiatric_set, injury_set, pain_management_set, pregnancy_set]
#some_set_lst = [heart_disease_set]
#disease_name_lst = ['heart_disease']
def create_cols_indices(some_set, agg_lst_map):
    some_cols_set = {agg[0] for agg in agg_lst_map if agg[1] in some_set}
    some_index_lst = [ind for ind, feat_name in enumerate(feature_names) if feat_name in some_cols_set]
    return some_index_lst
def create_idx_indices(group_mems, mbrs):
    idx_select = [ind for ind, mbr in enumerate(mbrs) if mbr in set(group_mems)]
    return idx_select

for i in range(len(some_set_lst)):
    some_set = some_set_lst[i]
    disease_name = disease_name_lst[i]
    some_index_lst = create_cols_indices(some_set, agg_lst_map)
    mbrs = data["ids"]
    group_id_lst = list(group_membership.keys()) # sample 10 group
    result_group_id_lst = ['population'] + group_id_lst
    def create_each_group_prev(group_id):
        #print('group id: ', group_id)
        if group_id == 'population':
            prev = matrix[:,some_index_lst].sum(axis=1).astype(bool).mean()
        else:
            group_mems = group_membership[group_id]
            idx_select = create_idx_indices(group_mems, mbrs)
            mbr_mtx = matrix[idx_select]
            prev = mbr_mtx[:,some_index_lst].sum(axis=1).astype(bool).mean()
        return prev
    result_prev_lst = Pool().map(create_each_group_prev, result_group_id_lst)
    if i == 0 :
        details = {
            'group_id' : result_group_id_lst,
            '{}_prev'.format(disease_name) : result_prev_lst
        }
        df = pd.DataFrame(details)
    else:
        df['{}_prev'.format(disease_name)] = result_prev_lst
    print('disease : ',  disease_name)
df.to_csv('gs://data-science-experiments-dev-env-0884/wellmark_deliveries/disease_prevalence_pandas/prev_only_summary.csv', index=False)
print('------------------------finish--------------------------')
print('--------------------------------------------------------------------')
