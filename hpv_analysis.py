import numpy as np
import pandas as pd
import os
import dill
import pickle as pkl

import mrtool
import paths as pth
import mr_functions
import plotting_functions
import matplotlib.pyplot as plt

output_dir = pth.MODEL_RESULTS_DIR
df = pd.read_csv(pth.CLEANED_REG_DF)
orig_df = df.copy()

response_name = 'log_icer_usd'
se_name = 'log_icer_se'
spline_cov = 'log_GDP_2017usd_per_cap'
study_id_name = 'ArticleID'
data_id_name = 'RatioID'

cwalk_params = pd.read_csv(pth.CWALK_PARAM_DRAWS)
cwalk_params = cwalk_params.rename(
    {'log_vaccine_cost': 'log_vaccine_cost_2017usd'},
    axis=1
    )

cwalk_covs = ['log_vaccine_cost_2017usd', 'coverage',
    'cost_disc_rate', 'burden_disc_rate']

cwalk_params = cwalk_params.set_index('draw')
cwalk_params = cwalk_params.drop(1000, axis=0)
cwalk_priors = cwalk_params.agg([np.mean, np.std]).T

df = df[
    (df[response_name].notnull()) &
    (~np.isinf(df[response_name])) &
    (df[spline_cov].notnull()) &
    (df['log_burden_variable'].notnull())
    ]
df = df[df[cwalk_covs].notnull().all(axis=1)].reset_index(drop=True)

cov_dict = {
    k: df[k].to_numpy()
    for k in [spline_cov, 'log_burden_variable'] + cwalk_covs
    }

cwalk_param_dict = {
    w: cwalk_priors.loc[w, :].to_numpy()
    for w in cwalk_covs
    }
# Need to create a deep copy because MRBRT functions are passed arguments
# by reference, and so values will be overwritten.
cwalk_prior_dict = {k: v for k, v in cwalk_param_dict.items()}

signal_mr_file = output_dir + 'signal_mr.pkl'
if not os.path.exists(signal_mr_file):
    signal_mr = mr_functions.fit_signal_model(
        df,
        resp_name=response_name, se_name=se_name, spline_cov=spline_cov,
        study_id_name=study_id_name, data_id_name=data_id_name,
        other_cov_names=cwalk_covs + ['log_burden_variable'],
        other_cov_gpriors=cwalk_param_dict,
        h=0.1, num_samples=20, deg=2, n_i_knots=2, knots_type='domain',
        knot_bounds=np.array([[0.1, 0.6], [0.4, 0.9]]),
        interval_sizes=np.array([[0.1, 0.7], [0.1, 0.7], [0.1, 0.7]])
        )
    with open(signal_mr_file, 'wb') as fl:
        dill.dump(signal_mr, fl)
else:
    with open(signal_mr_file, 'rb') as in_file:
        signal_mr = dill.load(in_file)

signal_df = mr_functions.create_signal(
    signal_mr, spline_cov, spline_cov_values=df[spline_cov].to_numpy(),
    data_id_name=data_id_name, data_ids=df[data_id_name].to_numpy()
    )

w_df = mr_functions.get_ws(signal_mr, data_id_name=data_id_name)
signal_df = signal_df.merge(w_df, on=[data_id_name])

df = df.merge(signal_df[[data_id_name, 'new_spline_cov', 'w']],
              on=[data_id_name])

sel_covs_file = output_dir + 'selected_covs.pkl'
if not os.path.exists(sel_covs_file):
    candidate_covs = [
        'log_burden_variable', 'not_lifetime',
        'quadrivalent', 'screen_comparator',
        'access_to_care_100', 'payer', 'qalys',
        'both_sex', 'no_booster'
        ]
    selected_covs = mr_functions.select_covariates(
        df=df[df['w'] >= 0.5],
        candidate_covs=candidate_covs,
        include_covs=['intercept', 'new_spline_cov'] + cwalk_covs,
        resp_name=response_name,
        se_name=se_name,
        study_id_name=study_id_name,
        beta_gprior=cwalk_param_dict
        )
    with open(sel_covs_file, 'wb') as handle:
        pkl.dump(selected_covs, handle, protocol=pkl.HIGHEST_PROTOCOL)
else:
    with open(sel_covs_file, 'rb') as handle:
        selected_covs = pkl.load(handle)

covs = ['log_vaccine_cost_2017usd',
        'cost_disc_rate',
        'burden_disc_rate',
        'coverage',
        'log_burden_variable',
        'not_lifetime',
        'quadrivalent',
        'screen_comparator',
        'access_to_care_100',
        'payer',
        'qalys',
        'both_sex'
       ]

df['null_study_id'] = df['RatioID']

outlier_df = df.copy()
df = df[df['w'] >= 0.5].copy().reset_index(drop=True)

df['null_study_id'] = df['RatioID']
cv_mse_file = output_dir + 'cv_mses.csv'
if os.path.exists(cv_mse_file):
    cv_results = pd.read_csv(cv_mse_file)
    cv_sds = cv_results['sd'].to_numpy()
    cv_mses = cv_results['mse'].to_numpy()
else:
    cv_sds, cv_mses = mr_functions.k_fold_cv_gaussian_prior(
        k=10,
        df=df,
        resp_name=response_name,
        se_name=se_name,
        study_id_name=study_id_name,
        data_id_name=data_id_name,
        covs=selected_covs,
        beta_gpriors=cwalk_prior_dict,
        initial_upper_prior_sd=1.0,
        num_sds_per_step=9,
        sd_tol=1e-5
        )
    cv_sds = cv_sds[np.argsort(cv_mses)]
    cv_mses = cv_mses[np.argsort(cv_mses)]
    
    cv_results = pd.DataFrame({'sd': cv_sds, 'mse': cv_mses})
    cv_results.to_csv(cv_mse_file, index=False)

upper_bound = np.quantile(np.log(cv_mses), 0.3)
lower_bound = np.log(cv_mses)[0]
upper_bound = upper_bound + (upper_bound - lower_bound) * 1.1
lower_bound = lower_bound - (upper_bound - lower_bound) * 1.1

msk = np.log(cv_mses) < upper_bound
xvals = cv_sds[msk]
yvals = np.log(cv_mses)[msk]

left_bound = np.min(xvals)
right_bound = np.max(xvals)
max_minus_min = right_bound - left_bound
left_bound = left_bound - max_minus_min * 1.1
right_bound = right_bound + max_minus_min * 1.1

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 8))
ax.scatter(x=xvals, y=yvals)
ax.set_ylim((lower_bound, upper_bound))
ax.set_xlim((left_bound, right_bound))

cv_mse_plot_file = output_dir + 'cv_mse_plot.png'
if not os.path.exists(cv_mse_plot_file):
    plt.savefig(cv_mse_plot_file)

prior_sd = cv_sds[np.argmin(cv_mses)]
# Adjust gaussian priors for selected non-crosswalk covariates.
# Prior SD should be equal-strength on a normalized-scale, which
# means they should be inversely proportional to the SD of covariate
# values.
gpriors = {v: np.array([0, prior_sd / df[v].std()])
           for v in selected_covs
           if v not in ['intercept'] + list(cwalk_prior_dict.keys())}

gpriors.update(cwalk_prior_dict)
gpriors.update({'intercept': [np.array([0, np.inf]),
                              np.array([0, np.inf])]})

out_file = output_dir + 'final_model.pkl'
if not os.path.exists(out_file):
    mr = mr_functions.fit_with_covs(df=df, covs=selected_covs,
                                resp_name=response_name, se_name=se_name,
                                study_id_name=study_id_name,
                                data_id_name=data_id_name,
                                z_covs=['intercept'],
                                trim_prop=0.0, spline_cov=None,
                                gprior_dict=gpriors,
                                inner_max_iter=2000, outer_max_iter=1000)

    with open(out_file, 'wb') as fl:
        dill.dump(mr, fl)
else:
    with open(out_file, 'rb') as in_file:
        mr = dill.load(in_file)

# Specify a random seed for reproducibility.
np.random.seed(5032198)
beta_samples = mrtool.core.other_sampling.sample_simple_lme_beta(1000, mr)
beta_samples_df = pd.DataFrame(beta_samples, columns=list(mr.fe_soln.keys()))
beta_samples_df.to_csv(output_dir + 'final_mr_beta_samples.csv', index=False)

mr_summary = mr_functions.summarize_parameters(mr, 'log_GDP_2017usd_per_cap')

cols_to_round = ['beta', 'beta_se', 'beta_variance', 'gamma']
mr_summary[cols_to_round] = np.round(mr_summary[cols_to_round], decimals=4)
mr_summary.to_csv(output_dir + 'final_model_parameter_summary.csv', index=False)

plotting_functions.visualize_spline(signal_mr,
                                    'log_GDP_2017usd_per_cap',
                                    df['log_GDP_2017usd_per_cap'].to_numpy(),
                                    x_on_log_scale=True,
                                    out_file_name=pth.PLOT_DIR + 'spline_transformation_plot.pdf',
                                    x_label='GDP per capita (USD)'
                                   )

fit_df = mr_functions.create_fit_df(mr=mr, df=df,
                                    resp_name=response_name,
                                    study_id_name=study_id_name,
                                    other_id_col_names=[],
                                    data_id_name=data_id_name)

r2s = mr_functions.r2(mr, fit_df, response_name)


fitted_vals_df = mr_functions.create_predictions(mr, signal_mr, df,
                                                 response_name, se_name, selected_covs,
                                                 study_id_name, data_id_name,
                                                 beta_samples=beta_samples,
                                                 seed=987432)
fitted_vals_df.to_csv(output_dir + 'fitted_df.csv')

## R^2 for only the base-case analyses
fit_df = mr_functions.create_fit_df(mr=mr, df=df[df['sensitivity'] == 0],
                                    resp_name=response_name,
                                    study_id_name=study_id_name,
                                    other_id_col_names=[], data_id_name=data_id_name)

r2_base_cases = mr_functions.r2(mr, fit_df, response_name)
r2_base_cases = r2_base_cases.rename({'Sample_size': 'Sample_size_base_cases',
                                      'R_squared': 'R_squared_base_cases',
                                      'RMSE': 'RMSE_base_cases'},
                                     axis=1)

r2s = r2s.join(r2_base_cases)

r2s.loc['fitted_fe_only', 'Sample_size_base_cases'] = fit_df.shape[0]
r2s.to_csv(output_dir + 'R2s.csv')

plotting_functions.plot_quartiles_with_ui(
    mr,
    y_axis_var='log_icer_usd',
    x_axis_var='log_vaccine_cost_2017usd',
    data_id_name='RatioID',
    group_var='log_burden_variable',
    spline_transform_df=None, spline_var=None,
    beta_samples=beta_samples,
    plot_title=None,
    group_var_name_display='Cervical cancer DALYs per capita',
    y_axis_var_display='ICER (2017 US$/DALY Averted)',
    x_axis_var_display='Vaccine Cost (2017 US$)',
    outliers=outlier_df.loc[outlier_df['w']< 0.5],
    out_dir=pth.PLOT_DIR,
    file_name='fit_vs_vacc_cost_by_burden_quartile.png'
    )

plotting_functions.plot_quartiles_with_ui(
    mr,
    y_axis_var='log_icer_usd',
    x_axis_var='log_burden_variable',
    data_id_name='RatioID',
    group_var='log_vaccine_cost_2017usd',
    spline_transform_df=None, spline_var=None,
    beta_samples=beta_samples,
    plot_title=None,
    group_var_name_display='Vaccine Cost',
    y_axis_var_display='ICER (2017 US$/DALY Averted)',
    x_axis_var_display='Cervical cancer DALYs per 100,000 population',
    outliers=outlier_df.loc[outlier_df['w'] < 0.5],
    x_decimals=-1, x_scale=1e5,
    out_dir=pth.PLOT_DIR,
    file_name='fit_vs_burden_by_vacc_cost_quartile.png'
    )

spline_transf_df = df[['RatioID', 'log_GDP_2017usd_per_cap', 'new_spline_cov']]
plotting_functions.plot_quartiles_with_ui(
    mr,
    y_axis_var='log_icer_usd',
    x_axis_var='log_GDP_2017usd_per_cap',
    data_id_name='RatioID',
    group_var='log_vaccine_cost_2017usd',
    spline_transform_df=spline_transf_df,
    spline_var='new_spline_cov',
    beta_samples=beta_samples,
    plot_title=None,
    group_var_name_display='Vaccine cost (2017 US$)',
    y_axis_var_display='ICER (2017 US$/DALY Averted)',
    x_axis_var_display='GDP per capita (2017 US$)',
    outliers=outlier_df.loc[outlier_df['w'] < 0.5],
    out_dir=pth.PLOT_DIR,
    file_name='fit_vs_gdp_by_vacc_cost_quartile.png'
    )

plotting_functions.plot_quartiles_with_ui(
    mr,
    y_axis_var='log_icer_usd',
    x_axis_var='log_vaccine_cost_2017usd',
    data_id_name='RatioID',
    group_var='log_GDP_2017usd_per_cap',
    spline_transform_df=spline_transf_df,
    spline_var='new_spline_cov',
    beta_samples=beta_samples,
    plot_title=None,
    x_axis_var_display='Vaccine cost (2017 US$)',
    y_axis_var_display='ICER (2017 US$/DALY Averted)',
    group_var_name_display='GDP per capita (2017 US$)',
    outliers=outlier_df.loc[outlier_df['w'] < 0.5],
    out_dir=pth.PLOT_DIR,
    file_name='fit_vs_vacc_cost_by_gdp_quartile.png'
    )

plotting_functions.plot_quartiles_with_ui(
    mr,
    y_axis_var='log_icer_usd',
    x_axis_var='log_burden_variable',
    data_id_name='RatioID',
    group_var='log_GDP_2017usd_per_cap',
    spline_transform_df=spline_transf_df,
    spline_var='new_spline_cov',
    beta_samples=beta_samples,
    group_var_name_display='GDP per capita',
    plot_title=None,
    y_axis_var_display='ICER (2017 US$/DALY Averted)',
    x_axis_var_display='Cervical cancer DALYs per 100,000 population',
    x_decimals=-1, x_scale=1e5,
    outliers=outlier_df.loc[outlier_df['w'] < 0.5],
    out_dir=pth.PLOT_DIR,
    file_name='fit_vs_burden_by_gdp_quartile.png'
    )

fitteds = mr.predict(mr.data, predict_for_study=False)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 8))
ax.scatter(x=fitteds, y=mr.data.obs - fitteds)
plt.savefig(pth.PLOT_DIR + 'fits_vs_obs_plot.png')

fitteds = mr.predict(mr.data, predict_for_study=True)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 8))
ax.scatter(x=fitteds, y=mr.data.obs - fitteds)
plt.savefig(pth.PLOT_DIR + 'fits_vs_resids_plot.png')
