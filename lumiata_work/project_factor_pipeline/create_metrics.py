import numpy as np
import pandas as pd
from matplotlib import rc
from matplotlib import pyplot as plt
from sklearn.metrics import *
import scipy.sparse as sps
import os
from dateutil.relativedelta import relativedelta

from dscost.pipeline import Config, utils
from dscost.pipeline.utils import cost_capture
from lumiata.utils import load_file

plt.switch_backend('agg')
rc('font', family='serif')
plt.style.use('seaborn')


def cost_capture_thresh(y_true, y_pred, thresh=50000):
    costs_pred = y_true[y_pred > thresh].sum()
    costs_true = y_true[y_true > thresh].sum()
    return costs_pred / costs_true


def newly_top_cost_capture(y_true, y_pred, y_prior, p=0.9):
    top = np.quantile(y_true, p)
    top_pred = np.quantile(y_pred, p)
    top_old = np.quantile(y_prior, p)
    costs_pred = y_true[(y_pred > top_pred) & (y_prior < top_old)].sum()
    costs_true = y_true[(y_true > top) & (y_prior < top_old)].sum()
    return costs_pred / costs_true


def make_pr_plot(y_true, y_prob, thresh, ap, dest_path):

    file_path = os.path.join(dest_path,
                             '{}k_precision_recall.png'
                             .format(int(thresh // 1000)))
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.xlabel('Recall', fontsize=15)
    plt.ylabel('Precision', fontsize=15)
    plt.ylim([0.0, 1.02])
    plt.xlim([-0.02, 1.02])
    plt.plot(recall, precision)
    plt.title('Average Precision = {0:0.3f}'.format(ap), fontsize=15)
    utils.save_plot(plt, file_path)
    plt.close()
    return


def get_overall_metrics(y_true, y_prior, y_pred):

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    cc = cost_capture(y_true, y_pred)

    top = np.quantile(y_true, 0.9)
    top_pred = np.quantile(y_pred, 0.9)
    prec = precision_score(y_true > top, y_pred > top_pred)
    recall = recall_score(y_true > top, y_pred > top_pred)
    f1 = f1_score(y_true > top, y_pred > top_pred)

    if y_prior is None:
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'r2': r2,
            'Top_Decile_Cost_Capture': cc,
            'Top_Decile_Precision': prec,
            'Top_Decile_Recall': recall,
            'Top_Decile_F1': f1
        }

    top_old = np.quantile(y_prior, 0.9)
    y_true_new_top = y_true[(y_true > top) & (y_prior < top_old)]
    y_pred_new_top = y_pred[(y_true > top) & (y_prior < top_old)]
    if len(y_true_new_top) > 0:
        rmse_new_top = np.sqrt(mean_squared_error(y_true_new_top,
                                                  y_pred_new_top))
    else:
        rmse_new_top = None

    cc_new_top = newly_top_cost_capture(y_true, y_pred, y_prior)

    y_true_filt = y_true[y_prior < top_old]
    y_pred_filt = y_pred[y_prior < top_old]
    prec_new_top = precision_score(y_true_filt > top, y_pred_filt > top_pred)
    recall_new_top = recall_score(y_true_filt > top, y_pred_filt > top_pred)
    f1_new_top = f1_score(y_true_filt > top, y_pred_filt > top_pred)

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'r2': r2,
        'Top_Decile_Cost_Capture': cc,
        'Top_Decile_Precision': prec,
        'Top_Decile_Recall': recall,
        'Top_Decile_F1': f1,
        'Newly_Top_Decile_RMSE': rmse_new_top,
        'Newly_Top_Decile_Cost_Capture': cc_new_top,
        'Newly_Top_Decile_Precision': prec_new_top,
        'Newly_Top_Decile_Recall': recall_new_top,
        'Newly_Top_Decile_F1': f1_new_top
    }


def get_thresh_dependent_metrics(y_true, y_prior, indv_preds, mbrs, dest_path):

    y_pred = np.array([indv_preds['reg'][s] for s in mbrs])
    y_prob = {}
    threshs = []
    for key, preds in indv_preds.items():
        if key[:3] == 'clf':
            threshs.append(int(key.split('_')[1][:-1]) * 1000)
            y_prob[key.split('_')[1]] = np.array([preds[s] for s in
                                                  mbrs])
    if not threshs:
        threshs = [50000, 100000, 200000]
    else:
        threshs.sort()

    rmse = {'metric': 'RMSE_above_thresh'}
    cc = {'metric': 'Cost_Capture_above_thresh'}
    prec = {'metric': 'Precision'}
    recall = {'metric': 'Recall'}
    f1 = {'metric': 'F1'}
    ap = {'metric': 'Average_Precision'}
    prec_new = {'metric': 'Newly_HiCC_Precision'}
    recall_new = {'metric': 'Newly_HiCC_Recall'}
    f1_new = {'metric': 'Newly_HiCC_F1'}
    ap_new = {'metric': 'Newly_HiCC_Average_Precision'}

    cols = []
    for thresh in threshs:

        key = str(thresh // 1000) + 'k'
        cols.append(key)

        if sum(y_true > thresh) > 0:
            rmse[key] = np.sqrt(mean_squared_error(y_true[y_true > thresh],
                                                   y_pred[y_true > thresh]))
        cc[key] = cost_capture_thresh(y_true, y_pred, thresh)
        prec[key] = precision_score(y_true > thresh, y_pred > thresh)
        recall[key] = recall_score(y_true > thresh, y_pred > thresh)
        f1[key] = f1_score(y_true > thresh, y_pred > thresh)

        if y_prob:
            y_score = y_prob[key]
        else:
            y_score = y_pred
        ap[key] = average_precision_score(y_true > thresh, y_score)
        make_pr_plot(y_true > thresh, y_score, thresh, ap[key], dest_path)
        
        if y_prior is None:
            continue

        y_true_filt = y_true[y_prior < thresh]
        y_score_filt = y_score[y_prior < thresh]
        y_pred_filt = y_pred[y_prior < thresh]
        prec_new[key] = precision_score(y_true_filt > thresh,
                                        y_pred_filt > thresh)
        recall_new[key] = recall_score(y_true_filt > thresh,
                                       y_pred_filt > thresh)
        f1_new[key] = f1_score(y_true_filt > thresh, y_pred_filt > thresh)
        ap_new[key] = average_precision_score(y_true_filt > thresh,
                                              y_score_filt)

    columns = ['metric'] + cols
    if y_prior is None:
        return pd.DataFrame([rmse, cc, prec, recall, f1, ap], columns=columns)

    return pd.DataFrame([rmse, cc, prec, recall, f1, ap, prec_new, recall_new,
                         f1_new, ap_new], columns=columns)


def load_matrix_data_from_npz(path, dense=False):
    npz_data = load_file(path)
    matrix = sps.csr_matrix(
        (npz_data["data"], npz_data["indices"], npz_data["indptr"]),
        shape=npz_data["_shape"]
    )
    if dense:
        matrix = matrix.toarray()
    del npz_data["data"], npz_data["indices"], npz_data["indptr"], npz_data[
        "_shape"]
    if 'maxprint' in npz_data:
        del npz_data['maxprint']

    npz_data.update({'matrix': matrix})

    return npz_data


def parse_args():
    parser = utils.get_arg_parser(description='Create metrics',
                                  datasets=True,
                                  model=True,
                                  overwriteable=True)
    args = utils.parse_args(parser)
    config = Config.from_file(args.config_file)
    config.check_datasets_and_models(args.datasets, [args.model])
    return config, args.model, args.datasets, args.overwrite


def run():
    config, model_name, datasets, overwrite = parse_args()

    for dname in datasets:
        dset = config.datasets[dname]
        path_train_mat = config.get_features_selected_matrix_path(
                dname, model_name)
        ind_preds_path = config.get_model_results_path(model_name,
                                                       dname,
                                                       'indv',
                                                       'json')
        base_dest_path = dset.plots_path
        is_gs = 'gs://' in base_dest_path
        mat_dict = load_matrix_data_from_npz(path_train_mat)
        ind_preds = utils.load_file(ind_preds_path)

        if dset.is_split:
            splits = ['train', 'test', 'evaluate']
        else:
            splits = ['all']

        for split in splits:
            if dset.is_split:
                dest_path = os.path.join(base_dest_path, model_name, split)
            else:
                dest_path = os.path.join(base_dest_path, model_name)

            if overwrite or utils.validate_path(dest_path):

                if not is_gs and not os.path.exists(dest_path):
                    os.makedirs(dest_path)

                mask = (mat_dict["splits"] == split)
                X, y_true, mbrs = (
                    mat_dict['matrix'][mask, :],
                    mat_dict['costs'][mask],
                    mat_dict['MBR'][mask])
                print('number of target values in {} split'.format(split),
                      len(y_true))
                y_pred = np.array([ind_preds['reg'][s] * ind_preds['reg_fac'][s] for s in mbrs])
                
                sd = dset.min_left_censor - relativedelta(
                    months=dset.blackout_months)
                sd_minus_1_yr = sd - relativedelta(years=1)
                right = sd.strftime('%Y-%m-%d')
                left = sd_minus_1_yr.strftime('%Y-%m-%d')
                # check min_date
                if left < dset.min_date.strftime('%Y-%m-%d'):
                    y_prior = None
                else:
                    feature_names = mat_dict['column_names']
                    col = f'Claims|{left}|{right}|AllowedCost'
                    idx = np.where(feature_names == col)[0][0]
                    y_prior = X[:, idx].toarray().reshape(-1)

                metrics = get_overall_metrics(y_true, y_prior, y_pred)
                df_metrics = get_thresh_dependent_metrics(
                    y_true,
                    y_prior,
                    ind_preds,
                    mbrs,
                    dest_path)
                utils.write_file(metrics,
                                 os.path.join(dest_path, 'metrics.json'))
                df_metrics.to_csv(
                    os.path.join(dest_path, 'thresh_dependent_metrics.csv'),
                    index=False)


if __name__ == '__main__':
    run()
