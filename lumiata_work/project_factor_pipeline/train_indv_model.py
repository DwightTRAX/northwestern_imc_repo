import os
import numpy as np
import scipy.sparse as sps

from dscost.pipeline import Config
from dscost.pipeline import utils
from lumiata.client import Client
from lumiata.utils import load_file
from lumiata.ai.models import GradientBoostedRegressionTree as GBRT
from lumiata.ai.models import GradientBoostedDecisionTree as GBDT


def get_features_targets(config, model, mat_dicts, target_name, train_sets=True):

    X_stacked = None
    y_stacked = None
    if train_sets:
        datasets = model.train_datasets
    else:
        datasets = model.validation_datasets

    for nds, ds in enumerate(datasets):
        dset = config.datasets[ds.dataset_name]
        mat_dict = mat_dicts[dset]
        splits_to_use = ds.splits
        mask = (np.isin(mat_dict["splits"], splits_to_use) &
                mat_dict["covered_on_left_censor"])
        # drop patients not enrolled on slice or not part of splits
        X, y = mat_dict["matrix"][mask, :], mat_dict[target_name][mask],

        if nds == 0:
            X_stacked, y_stacked = X, np.array(y)
        else:
            X_stacked = sps.vstack((X_stacked, X))
            y_stacked = np.concatenate((y_stacked, y))

    return X_stacked, y_stacked


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
    parser = utils.get_arg_parser(
        description="Train individual model",
        model=True,
        overwriteable=True)
    parser.add_argument(
        "--thresholds",
        nargs='+',
        type=float,
        help="list of HiCC thresholds in terms of dollar amount",
        default=[])
    args = utils.parse_args(parser)
    config = Config.from_file(args.config_file)
    config.check_datasets_and_models(models=[args.model])
    return config, args.model, args.thresholds, args.overwrite


def run():
    config, model_name, threshs, overwrite = parse_args()

    model = config.models[model_name]
    path_model_cache = config.get_model_cache_path(model_name, "indv")

    seen_dsets = set()
    mat_dicts = {}

    datasets = model.train_datasets
    if model.validation_datasets:
        datasets = model.train_datasets + model.validation_datasets

    for ds in datasets:
        dname = ds.dataset_name
        dset = config.datasets[dname]
        if dset not in seen_dsets:
            path_train_mat = config.get_features_selected_matrix_path(
                dname, model_name)
            if overwrite or utils.validate_path(path_model_cache):
                mat_dicts[dset] = load_matrix_data_from_npz(path_train_mat)
            seen_dsets.add(dset)

    feature_fn = list(seen_dsets)[0].feature_dicts_path
    featurizer = utils.load_file(feature_fn.replace('jsons', 'joblib'))
    feature_indices = list(mat_dicts.values())[0]["column_indices"]

    if overwrite or utils.validate_path(path_model_cache):

        # X,y for costs_ref
        X_train, y_train = get_features_targets(
            config, model, mat_dicts, 'costs_ref',train_sets=True)

        if model.validation_datasets:
            X_valid, y_valid = get_features_targets(
                config, model, mat_dicts, 'costs_ref',train_sets=False)
        # X,y for costs_proj_fac
        X_train_fac, y_train_fac = get_features_targets(
            config, model, mat_dicts, 'costs_proj_fac', train_sets=True)

        if model.validation_datasets:
            X_valid_fac, y_valid_fac = get_features_targets(
                config, model, mat_dicts, 'costs_proj_fac', train_sets=False)

        lgb_reg_params = {
            'objective': 'regression',
            'min_gain_to_split': 75,
            'min_data_in_leaf': 100,
            'num_leaves': 100,
            'num_iterations': 5000 if model.validation_datasets else 100
        }

        lgb_clf_params = {
            'objective': 'binary',
            'min_gain_to_split': 75,
            'learning_rate': .01,
            'min_data_in_leaf': 100,
            'num_leaves': 100,
            'num_iterations': 5000 if model.validation_datasets else 250
        }

        valid_params = {
            'early_stopping_rounds': 5,
            'verbose': 10
        }

        models = {}
        # train costs_ref model
        reg = GBRT("indv_model", model_package="lgbm", params=lgb_reg_params)
        if model.validation_datasets:
            reg.fit(features=X_train, targets=y_train,
                    validation_features=X_valid, validation_targets=y_valid,
                    **valid_params)
        else:
            reg.fit(features=X_train, targets=y_train)
        fimp = reg.model.feature_importances_
        print(f'Number of features found to be important: '
              f'{(fimp != 0).sum():,}')
        models['reg'] = reg

        # train costs_proj_fac
        reg_fac = GBRT("indv_model_fac", model_package="lgbm", params=lgb_reg_params)
        if model.validation_datasets:
            reg_fac.fit(features=X_train_fac, targets=y_train_fac,
                    validation_features=X_valid_fac, validation_targets=y_valid_fac,
                    **valid_params)
        else:
            reg_fac.fit(features=X_train_fac, targets=y_train_fac)
        fimp = reg_fac.model.feature_importances_
        print(f'Number of features found to be important: '
              f'{(fimp != 0).sum():,}')
        models['reg_fac'] = reg_fac

        client = Client(project_id='lumiata-internal-6f5a',
                        disable_rest_client=True)
        model_base_dir = os.path.dirname(path_model_cache)
        client.save_model_zip(
            output_dir=model_base_dir,
            model=reg,
            featurizer=featurizer,
            target_variable="allowed_cost",
            feature_selection_indices=feature_indices)

        for thresh in threshs:
            name = int(thresh // 1000)
            targets_train = y_train > thresh
            clf = GBDT(f"clf_model_{name}k", model_package="lgbm",
                       params=lgb_clf_params)
            if model.validation_datasets:
                targets_valid = y_valid > thresh
                clf.fit(features=X_train, targets=targets_train,
                        validation_features=X_valid,
                        validation_targets=targets_valid,
                        **valid_params)
            else:
                clf.fit(features=X_train, targets=targets_train)
            fimp = clf.model.feature_importances_
            print(f'Number of features found to be important: '
                  f'{(fimp != 0).sum():,}')
            models[f'clf_{name}k'] = clf

        utils.write_file(models, path_model_cache)


if __name__ == "__main__":
    run()
