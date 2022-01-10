from dscost.pipeline import Config
from dscost.pipeline import utils
import scipy.sparse as sps
from lumiata.utils import load_file


def get_preds_dict(mbrs, preds, digits=2):
    return {
        x[0]: round(float(max(0, x[1])), digits) for x in zip(mbrs, preds)
    }


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
        description="Apply individual model",
        datasets=True,
        model=True,
        overwriteable=True)
    args = utils.parse_args(parser)
    config = Config.from_file(args.config_file)
    config.check_datasets_and_models(args.datasets, [args.model])

    return config, args.model, args.datasets, args.overwrite


def main():
    config, model_name, dataset_names, overwrite = parse_args()

    model_cache_path = config.get_model_cache_path(model_name, "indv")
    cache = utils.load_file(model_cache_path)

    for dname in dataset_names:

        out_fn = config.get_model_results_path(
            model_name,
            dname,
            "indv",
            "json")
        if overwrite or utils.validate_path(out_fn):

            path_train_mat = config.get_features_selected_matrix_path(
                dname, model_name)
            orig_dict = load_matrix_data_from_npz(path_train_mat)
            X_test = orig_dict["matrix"]
            mbrs_test = orig_dict["MBR"]

            result_dict = {}
            preds = cache['reg'].predict(X_test)
            preds_fac = cache['reg_fac'].predict(X_test)
            result_dict['reg'] = get_preds_dict(mbrs_test, preds)
            result_dict['reg_fac'] = get_preds_dict(mbrs_test, preds_fac)
            for key in cache:
                if key[:3] != 'clf':
                    continue
                preds = cache[key].predict(X_test)
                result_dict[key] = get_preds_dict(
                    mbrs_test,
                    preds,
                    digits=3)

            print('writing results to', out_fn)
            utils.write_file(result_dict, out_fn)


if __name__ == "__main__":
    main()
