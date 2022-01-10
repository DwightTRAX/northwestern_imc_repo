import numpy as np
from scipy.sparse import vstack
from pyspark import SparkContext
from pyspark.sql import SQLContext
from dscost.pipeline import Config
from dscost.pipeline import utils
from lumiata.ai.features import *
from lumiata.utils import save_npz


def feature_prev_filter(mat, prev_limit):
    """Return indices of all matrix columns that pass prevalence criteria"""

    # If prev_limit is a float, it means the fraction
    if prev_limit < 1:
        prev_limit = round(prev_limit * mat.shape[0])
    prevalences = mat.astype('bool').sum(axis=0).A.reshape(-1)
    indices = np.where(prevalences >= prev_limit)[0]

    print(f'{len(indices):,} of {mat.shape[1]:,} features have '
          f'prevalence above {prev_limit:,}')

    return indices.tolist()


def save_npz_data(path, data_dict, array):
    data_dict.update(array.__dict__)
    save_npz(path, **data_dict)

def find_target_extra_index(target_lst):
    for i in range(len(target_lst)):
        if target_lst[i].endswith('AllowedCost|reference'):
            cost_ref_idx = i
        elif target_lst[i].endswith('AllowedCost|proj_factor'):
            cost_fac_idx = i
        elif target_lst[i].endswith('PMPM_Cost|reference'):
            pmpm_ref_idx = i
        elif target_lst[i].endswith('PMPM_Cost|proj_factor'):
            pmpm_fac_idx = i
    return (cost_ref_idx, cost_fac_idx, pmpm_ref_idx, pmpm_fac_idx)

def parse_args():
    parser = utils.get_arg_parser(description="Create Matrices",
                                  model=True,
                                  datasets=True,
                                  overwriteable=True)
    parser.add_argument("--full_matrix", action="store_true",
                        help="Create unfiltered full matrices",
                        default=False)
    args = utils.parse_args(parser)
    config = Config.from_file(args.config_file)
    config.check_datasets_and_models(args.datasets)
    return config, args.model, args.datasets, args.full_matrix, args.overwrite


def run():
    config, model_name, datasets, full_matrix, overwrite = parse_args()
    sc = SparkContext()
    sql = SQLContext(sc)

    # Create full matrices for all datasets
    full_matrices = {}
    data_dicts = {}
    for dname in datasets:
        dset = config.datasets[dname]
        split_map = utils.load_file(dset.sha1id_to_split_path)
        covered_mbr = utils.load_file(dset.sha1ids_covered_on_left_censor_path)
        covered_mbr = set(covered_mbr)

        vector_fn = dset.feature_dicts_path
        target_fn = dset.target_values_path
        feature_df = sql.read.parquet(vector_fn)
        feature_df = feature_df.repartition(2000)
        target_df = sql.read.parquet(target_fn)
        target_df = target_df.repartition(200)
        featurizer = utils.load_file(vector_fn.replace('jsons', 'joblib'))
        target_featurizer = utils.load_file(target_fn
                                            .replace('jsons', 'joblib'))

        print("Processing dataset {}".format(dname))

        member_ids, feature_array = featurizer.to_array(feature_df)
        column_names = featurizer.get_array_column_names()
        target_member_ids, target_array = target_featurizer.to_array(target_df)
        assert (member_ids == target_member_ids).sum() == len(member_ids)

        target_array = target_array.toarray()
        member_ids = member_ids.reshape(-1)

        # add on column, need featurizer update
        expected_columns = set(
            ['MBR'] + list(target_featurizer.column_feature_map.keys()))
        feature_columns = set(target_df.columns)
        diff_columns = sorted(list(feature_columns - expected_columns))[::-1]
        (cost_ref_idx, cost_fac_idx, pmpm_ref_idx, pmpm_fac_idx) = \
            find_target_extra_index(diff_columns)
        target_array_adj = np.array(target_df.select(diff_columns
                                                     ).collect())
        data_dict = {
            "MBR": member_ids,
            "column_names": np.array(column_names),
            "column_indices": np.arange(len(column_names)),
            "costs": target_array[:, 0],
            "pmpm": target_array[:, 1],
            "costs_ref": target_array_adj[:, cost_ref_idx],
            "pmpm_ref": target_array_adj[:, cost_fac_idx],
            "costs_proj_fac": target_array_adj[:, pmpm_ref_idx],
            "pmpm_proj_fac": target_array_adj[:, pmpm_fac_idx],
            "covered_on_left_censor": np.array([m in covered_mbr for m in
                                                member_ids]),
            "splits": np.array([split_map[m] for m in member_ids])
        }
        full_matrices[dset] = feature_array
        data_dicts[dset] = data_dict

        matrix_path = dset.matrix_path
        if full_matrix and (overwrite or utils.validate_path(matrix_path)):
            save_npz_data(matrix_path, data_dict, feature_array)

    # select features based on feature prevalence of the training members
    model = config.models[model_name]
    feat_prev = model.indv_feature_prev if model.indv_feature_prev else 0.0002
    stack_matrix = None
    for ds in model.train_datasets:
        dset = config.datasets[ds.dataset_name]
        splits_to_use = ds.splits
        mask = (np.isin(data_dicts[dset]["splits"], splits_to_use) &
                data_dicts[dset]["covered_on_left_censor"])
        mat = full_matrices[dset][mask, :]
        if stack_matrix is None:
            stack_matrix = mat
        else:
            stack_matrix = vstack((stack_matrix, mat))
    cols = feature_prev_filter(stack_matrix, feat_prev)

    # save feature-selected matrices
    for dname in datasets:
        dset = config.datasets[dname]
        outpath = config.get_features_selected_matrix_path(dname, model_name)
        dd = data_dicts[dset]
        dd["column_names"] = dd["column_names"][cols]
        dd["column_indices"] = dd["column_indices"][cols]
        if overwrite or utils.validate_path(outpath):
            save_npz_data(outpath, dd, full_matrices[dset][:, cols])

    print("Finished creating matrices")


if __name__ == "__main__":
    run()
