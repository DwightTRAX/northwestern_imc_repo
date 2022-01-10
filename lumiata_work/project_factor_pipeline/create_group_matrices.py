from lumiata.ai.segment import *
from lumiata.ai.features import *
from lumiata.ai.models import *

from copy import deepcopy
from dateutil.relativedelta import relativedelta

from dscost.pipeline import utils
from dscost.pipeline import Config


def get_featurizer(feat_params, target_params):

    agg_features = [
        AggregateAge(aggregators=['mean'], **feat_params),
        AggregateDaysCovered(aggregators=['mean'], **feat_params),
        AggregateHighCost(aggregators=['mean'], high_cost_threshold=8000,
                          **feat_params),
        AggregateMemberPrediction(aggregators=['mean', 'median', 'std'],
                                  **feat_params)
    ]

    four_months_params = deepcopy(feat_params)
    four_months_params['left_censor'] = four_months_params['right_censor'] - \
        relativedelta(months=4)

    group_features = [
        GroupPMPMCost(**feat_params),
        GroupMemberMonths(**feat_params),
        GroupAllowedAmount(**feat_params),
        GroupAllowedAmount(**four_months_params),

        # TODO: This is awkward. We actually want the member count on 1/1.
        # TODO: But because the class takes an "exclusive right censor" as
        # TODO: input, we need to pass 1/2 as the input
        GroupNumMembers(
            right_censor=feat_params['left_censor']+relativedelta(days=1)
        ),

        # Same as above. This time we want the count on 12/31
        GroupNumMembers(right_censor=feat_params['right_censor']),
        GroupPMPMCost(**target_params),
        GroupMemberMonths(**target_params),
    ]

    return GroupFeaturizer("group_features_targets",
                           function_set=agg_features+group_features,
                           global_params=feat_params)


def parse_args():
    parser = utils.get_arg_parser(description="Create group-level features and"
                                              " target values",
                                  model=True,
                                  datasets=True)
    args = utils.parse_args(parser)
    config = Config.from_file(args.config_file)
    config.check_datasets_and_models(args.datasets)
    return config, args.model, args.datasets


def main():
    config, model_name, datasets = parse_args()

    for dname in datasets:
        dset = config.datasets[dname]
        outpath = config.get_model_results_path(model_name,
                                                dname,
                                                "agg",
                                                "csv")

        max_date = dset.min_left_censor - relativedelta(
            months=dset.blackout_months)
        min_date = max_date - relativedelta(months=dset.projection_months)
        feature_params = {"left_censor": min_date, "right_censor": max_date}

        left_censor = dset.min_left_censor
        right_censor = left_censor + relativedelta(
            months=dset.projection_months)
        target_params = {"left_censor": left_censor,
                         "right_censor": right_censor}

        indv_pmpm_path = config.get_model_results_path(model_name,
                                                       dname,
                                                       "indv_pmpm",
                                                       "json")
        indv_pmpm = utils.load_file(indv_pmpm_path)
        mbrs, preds = zip(*indv_pmpm.items())

        # group-to-split map
        split_map = None
        if dset.is_split:
            split_map = utils.load_file(dset.group_to_split_path)

        seg = Segment.load_from_file(dset.ldm_path, "parquet")
        seg.repartition(2000)
        slice_date = (dset.min_left_censor
                      - relativedelta(months=dset.blackout_months)
                      - relativedelta(days=1)).date()

        print("Processing dataset {}".format(dname))
        group_seg = GroupSegment(df=seg.df,
                                 slice_date=slice_date)
        # append indv pmpm predictions
        group_seg.append_member_predictions(mbrs, preds)
        featurizer = get_featurizer(feature_params, target_params)
        df = featurizer.featurize(group_seg).toPandas()

        # add "trend" and "cost_fraction|last_4_month"
        begin = (min_date + relativedelta(days=1)).strftime('%Y-%m-%d')
        left = min_date.strftime('%Y-%m-%d')
        right = max_date.strftime('%Y-%m-%d')
        df["trend"] = (
            (df[f'Coverage|None|{right}|GroupNumMembers'] -
             df[f'Coverage|None|{begin}|GroupNumMembers']) /
            (df[f'Coverage|{left}|{right}|GroupMemberMonths'])
        )
        left_4m = (max_date - relativedelta(months=4)).strftime('%Y-%m-%d')
        df["cost_fraction|last_4_month"] = (
            df[f'Claims|{left_4m}|{right}|GroupAllowedAmount'] /
            (df[f'Claims|{left}|{right}|GroupAllowedAmount'] + 1)
        )

        if dset.is_split:
            df['split'] = df.GROUP.map(split_map)
        df.to_csv(outpath, index=False)


if __name__ == "__main__":
    main()
