import pandas as pd
import numpy as np
import os
import paths as pth

df = pd.read_csv(pth.CLEANED_REG_DF)

df = df.rename({'year': 'year_publication',
                'CostsDiscountRate': 'cost_disc_rate',
                'DiscountRate': 'burden_disc_rate',
               },
               axis=1)

# list of all sensitivity analysis covariates
sens_vars = ['log_vaccine_cost_2017usd', 'coverage',
             'both_sex', 'quadrivalent', 'qalys',
             'screen_comparator', 'not_lifetime',
             'cost_disc_rate', 'burden_disc_rate',
             'payer', 'ltd_societal', 'access_to_care_100',
             'no_booster'#
            ]
# dict of reference values used to select a single reference ratio
ref_vals = {'log_vaccine_cost_2017usd': np.nan,
            'coverage': 80, 'both_sex': 0, 'quadrivalent': 0, 'qalys': 0,
            'screen_comparator': 0, 'not_lifetime': 0,
            'cost_disc_rate': 3, 'burden_disc_rate': 3,
            'payer': 1, 'ltd_societal': 0, 'access_to_care_100': 0,
            'no_booster': 0
           }

sens_df = df[(df['sensitivity'] == 1)].copy()

# filter out cost-saving ratios
sens_df = sens_df[(~np.isinf(sens_df['log_icer_usd'])) &
    (sens_df['log_icer_usd'].notnull())]

# Create a dict of sensitivity-reference pairs and a dict mapping sensitivity
# analyses to the list of covariates for which they are sensitivity analyses
sens_r_ids = sens_df['RatioID']
pairs = {}
sens_var_assignments = {}

# Loop over unique article ids
for a_id in sens_df['ArticleID'].unique():
    # Subset the data frame, filtering out ratios from other articles
    article_df = df.loc[df['ArticleID'] == a_id].copy().set_index('RatioID')
    non_na_sens = (article_df['sensitivity'] == 1) &\
        (article_df['log_icer_usd'].notnull())
    for r_id, row in article_df[non_na_sens].iterrows():
        locn = row['lancet_label']
        # Create a df of candidate pairs by dropping the current ratio and
        # ratios from other locations
        pair_df = article_df.drop(r_id, axis=0).loc[
            (article_df['lancet_label'] == locn)].copy()
        
        # Find all candidate ratios that differ in exactly one covariate from
        # the current ratio
        pair_df = pair_df.assign(**{w + '_sens': row[w] for w in sens_vars})
        pair_df = pair_df.assign(
            **{w + '_unequal': pair_df[w] != pair_df[w + '_sens']
                for w in sens_vars}
            )
        pair_df['num_covs_unequal'] = pair_df[
            [w + '_unequal' for w in sens_vars]
            ].sum(axis=1)
        pair_df = pair_df.loc[pair_df['num_covs_unequal'] == 1]
        uneq_covs = pair_df.reset_index().melt(
            id_vars='RatioID',
            value_vars=[w + '_unequal' for w in sens_vars],
            var_name='covariate')
        uneq_covs['covariate'] = uneq_covs['covariate'].str.replace(
            pat='_unequal$', repl='', regex=True)
        uneq_covs = uneq_covs.loc[uneq_covs['value'] == True]
        # For each covariate covariates that is the only one that differs
        # between the current ratio and another, add that covariate to the
        # list of sensitivity covariates "assigned" to the current ratio.
        r_id_sens_covs = list(uneq_covs['covariate'].unique())
        sens_var_assignments.update({r_id: r_id_sens_covs})
        # Update the dictionary that maps ratios to dictionaries that map that
        # each ratio's sensitivity covariates to the list of ratios that differ
        # from the given ratio that covariate and no other.
        r_id_pairs = uneq_covs.groupby('covariate')['RatioID']
        r_id_pairs = r_id_pairs.apply(list).to_dict()
        pairs.update({r_id: r_id_pairs})

unpairable_rids = [
    key for key, val in sens_var_assignments.items()
    if len(val) == 0
    ]

# Create dicts mapping s.a. ratios to sensitivity covariates with possible
# reference ratios that all have NA/c-s icers and to possible reference ratios
# with non-NA icers.
pair_icer_na_inf = {r: [] for r in sens_r_ids}
valid_icer_pairs = {r: {} for r in sens_r_ids}
# Create a list of sensitivity analysis ratios that cannot be paired with any
# sensitivity covariate.
rids_unpairable_for_any_cov = []
# Loop over s.a. ratios and the dicts that map their sensitivity covariates to
# the possible reference analyses for that covariate.
for r_id, val in pairs.items():
    if len(val) > 0:
        one_rid_icer_pairs = {}
        # Loop over covariates and possible reference analyses for the given
        # ratio and covariate.
        for w, pair_ids in val.items():
            pair_df = df.set_index('RatioID').loc[pair_ids]
            r_id_df = sens_df.set_index('RatioID').loc[r_id]
            # Create a pd.Series indicating whether potential reference analyses
            # have NA/c-s icers
            icers_invalid = ((pair_df['log_icer_usd'].isnull()) |
                (np.isinf(pair_df['log_icer_usd'])))
            # If all possible reference icers are NA/c-s, add the sensitivity
            # covariate to the list of sensitivity covariates with only NA/c-s
            # reference ratios for the given s.a. ratio.
            # Otherwise add the sensitivity covariate and valid ratios to the
            # dictionary of valid reference ratios for the given s.a.
            if icers_invalid.all():
                pair_icer_na_inf[r_id].append(w)
            else:
                one_rid_icer_pairs.update(
                    {w: list(pair_df.loc[~icers_invalid].index)}
                    )
        # If all sensitivity covariates have reference analyses that are
        # exclusively NA/c-s for the given s.a., add the s.a. ratio to the
        # list of such s.a. ratios.
        if len(pair_icer_na_inf[r_id]) == len(list(val.keys())):
            rids_unpairable_for_any_cov.append(r_id)
        else:
            valid_icer_pairs.update({r_id: one_rid_icer_pairs})

# Create a dict mapping sensitivity covariates to dfs of s.a.-reference pairs.
paired_dfs = {w: [] for w in sens_vars}
# Create a list of s.a. RatioIDs that match another ratio in all covariates
# and one of those that differ from all other ratios in at least two covariates.
identical_covariate_rids = []
no_match_rids = []
df = df.set_index('RatioID')
for r_id, val in valid_icer_pairs.items():
    if not r_id in list(df.index):
        raise ValueError('df doesnt contain ratio')
    r_id_row = df.loc[r_id]
    # Loop over covariates and possible reference analyses for the current ratio.
    for w, pair_rids in val.items():
        r_id_w_val = r_id_row[w]
        pair_df = df.loc[pair_rids]
        if len(pair_rids) > 1:
            # If there are multiple possible pairs and one is specified as the
            # base-case select that one.
            spec_bc = r_id_row['base_case']
            if r_id_row.notnull()['base_case']:
                if spec_bc in pair_rids:
                    pair_df = pair_df.loc[[spec_bc]].copy()
            else:
                # Otherwise use the value of the covariate used for predictions.
                ref_val = ref_vals.get(w)
                if np.isnan(ref_val):
                    # If there is no value, e.g. vacc cost, select the Tufts-
                    # extracted ratio if possible.
                    if any(list(pair_df['sensitivity'] == 0)):
                        pair_df = pair_df[pair_df['sensitivity'] == 0].copy()
                    if pair_df.shape[0] != 1:
                        if (pair_df[sens_vars].nunique(axis=0) == 1).all():
                            if pair_df['log_icer_usd'].nunique() > 1:
                                # If all candidate pairs match r_id in all
                                # values of all covariates, but don't match
                                # its ICER, add r_id to the list of such ratios.
                                identical_covariate_rids.extend(
                                    list(pair_df['RatioID'])
                                    )
                            elif pair_df['log_icer_usd'].nunique() == 1:
                                # If all candidate pairs have all covariates and
                                # ICER equal to r_id's, 
                                message = ', '.join(list(pair_df['RatioID']))
                                message = r_id + ' matches' + message
                                raise ValueError(message)
                            else:
                                raise ValueError('All pairs were dropped somehow')
                        else:
                            # If the sensitivity covariate takes multiple values
                            # among candidate reference analyses, select the one
                            # closest to the average one extracted by Tufts.
                            merge_cols = ['ArticleID', 'lancet_label'] +\
                                [v for v in sens_vars if v != w]
                            pair_df = pair_df.reset_index().merge(
                                df[merge_cols + [w]],
                                on=merge_cols,
                                how='left', suffixes=['', '_base']
                                )
                            # select the one that is closest to the base case in absolute value
                            pair_df['abs_diff_to_base'] = np.abs(pair_df[w] - pair_df[w + '_base'])
                            pair_df = pair_df.sort_values('abs_diff_to_base').iloc[[0], :]
                            pair_df = pair_df.drop('abs_diff_to_base', axis=1)
                else:
                    # For sensitivity covariates with specified prediction
                    # values specified in ref_vals, select the closest one
                    # to the prediction value.
                    pair_df['abs_diff_to_base'] = np.abs(pair_df[w] - ref_val)
                    argmin_diff_to_base = np.argmin(pair_df['abs_diff_to_base'])
                    pair_df = pair_df.iloc[[argmin_diff_to_base], :]
                    pair_df = pair_df.drop('abs_diff_to_base', axis=1)
        if pair_df.shape[0] > 1:
            if (pair_df[sens_vars].nunique(axis=0) == 1).all():
                if pair_df['log_icer_usd'].nunique() > 1:
                    identical_covariate_rids.extend(list(pair_df['RatioID']))
                elif pair_df['log_icer_usd'].nunique() == 1:
                    pair_df = pair_df.iloc[[0], :]
                else:
                    raise ValueError('all pairs were dropped somehow')
            else:
                no_match_rids.append(r_id)
        else:
            r_id_df = pd.DataFrame(r_id_row).T
            r_id_df.index.name = 'RatioID'
            r_id_df = r_id_df.reset_index()
            merged_df = r_id_df.merge(pair_df.reset_index(),
                                      on=['ArticleID', 'lancet_label'] +\
                                          [v for v in sens_vars if v != w],
                                      suffixes=('_sens', '_ref'), how='left'
                                     )
            merged_df = merged_df.rename({w + '_sens': 'value_sens',
                                          w + '_ref': 'value_ref'},
                                          axis=1).assign(sens_variable=w)
            paired_dfs[w].append(merged_df)

identical_covariate_rids = list(set(identical_covariate_rids))
no_match_rids = list(set(no_match_rids))
if len(identical_covariate_rids) > 0:
    raise ValueError(
        str(len(identical_covariate_rids)) +
        ' ratios have identical covariates to another ratio.'
        )
if len(no_match_rids) > 0:
    raise ValueError(
        str(len(no_match_rids)) +
        ' sensitivity analysis ratios cannot be matched to another ratio.'
        )

paired_df = pd.concat(
    [pd.concat(y, axis=0) for v, y in paired_dfs.items() if len(y) > 0],
    axis=0
    )

failed_rids = list(set(identical_covariate_rids + no_match_rids))

# Exclude redundant pairs - e.g. if there are two ratio-pairs A-B and B-A,
# drop one and keep the other.
redundancies = []
for idx, row in paired_df.iterrows():
    redundancies.append(paired_df.loc[
        (paired_df['RatioID_ref'] == row['RatioID_sens']) &
        (paired_df['RatioID_sens'] == row['RatioID_ref']) &
        (paired_df['sens_variable'] == row['sens_variable']),
        ['RatioID_ref', 'RatioID_sens', 'sens_variable']
        ])

redundancies = [i for i in redundancies if i.shape[0] > 0]

redundancies = pd.concat(redundancies, axis=0, ignore_index=True)

redundancies['drop'] = np.nan
redundancies = redundancies.reset_index(drop=True)
for i in range(redundancies.shape[0]):
    if np.isnan(redundancies.loc[i, 'drop']):
        redundancies.loc[i, 'drop'] = 0
        x = (redundancies['RatioID_ref'] == redundancies.loc[i, 'RatioID_sens']) &\
            (redundancies['RatioID_sens'] == redundancies.loc[i, 'RatioID_ref']) &\
            (redundancies['sens_variable'] == redundancies.loc[i, 'sens_variable'])
        if x.sum() != 1:
            raise ValueError('No unique pair ratio')
        redundancies.loc[x, 'drop'] = 1

redundancies = redundancies.loc[redundancies['drop'] == 1]
paired_df = paired_df.merge(
    redundancies,
    on=['RatioID_sens', 'RatioID_ref', 'sens_variable'],
    how='left')
paired_df.loc[paired_df['drop'].isnull(), 'drop'] = 0
paired_df = paired_df.loc[paired_df['drop'] == 0].drop('drop', axis=1)

if not os.path.exists(pth.PAIRED_CWALK_DF):
    print('writing to ' + pth.PAIRED_CWALK_DF)
    paired_df.to_csv(pth.PAIRED_CWALK_DF, index=False)
