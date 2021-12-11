import pandas as pd
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import paths as pth
import mrtool
import crosswalk_functions

df = pd.read_csv(pth.PAIRED_CWALK_DF)

df = df.rename({'log_icer_usd_sens': 'log_icer_sens',
                'log_icer_usd_ref': 'log_icer_ref',
                'log_vaccine_cost_2017usd_ref': 'log_vacc_cost_ref',
                'log_vaccine_cost_2017usd_sens': 'log_vacc_cost_sens'},
                axis=1)

param_estimate_df = pd.DataFrame({'draw': np.arange(1000)})
param_summary_df = pd.DataFrame({v: [] for v in ['beta', 'se_beta']},
                                index=[])

# Vaccine Cost
vacc_cost_df = df[df['sens_variable'] == 'log_vaccine_cost_2017usd']
vacc_cost_df = vacc_cost_df.rename(
    {'value' + i: 'log_vacc_cost' + i for i in ['_sens', '_ref']},
    axis=1)
vacc_cost_df = crosswalk_functions.create_diff_variables(
    vacc_cost_df,
    resp_name='log_icer',
    se_name='log_icer_se',
    cov_name='log_vacc_cost')

plt.scatter(x=vacc_cost_df['log_vacc_cost_diff'].to_numpy(),
            y=vacc_cost_df['log_icer_diff'].to_numpy(),
           )
plt.savefig(pth.PLOT_DIR + 'vacc_cost_cwalk_plot.png')
plt.clf()

mr = crosswalk_functions.cwalk(
    vacc_cost_df,
    'log_icer',
    'log_icer_se',
    'log_vacc_cost',
    'ArticleID'
    )

vacc_cost_beta_samples, vacc_cost_summary = crosswalk_functions.summarize_cwalk(mr, 'log_vacc_cost')

param_estimate_df['log_vaccine_cost'] = vacc_cost_beta_samples

param_summary_df = pd.concat([param_summary_df,
                              vacc_cost_summary],
                             axis=0)

# Coverage
cvg_df = df[df['sens_variable'] == 'coverage'].copy()
cvg_df = cvg_df.rename(
    {'value' + i: 'coverage' + i for i in ['_sens', '_ref']},
    axis=1)

cvg_df = crosswalk_functions.create_diff_variables(
    cvg_df,
    resp_name='log_icer',
    se_name='log_icer_se',
    cov_name='coverage')

plt.scatter(x=cvg_df['coverage_diff'].to_numpy(),
            y=cvg_df['log_icer_diff'].to_numpy(),
           )
plt.savefig(pth.PLOT_DIR + 'coverage_cwalk_plot.png')
plt.clf()

cvg_mr = crosswalk_functions.cwalk(
    cvg_df, resp_name='log_icer', se_name='log_icer_se',
    cov_name='coverage', study_id='ArticleID',
    monotonicity='increasing')

cvg_beta_samples, cvg_summary = crosswalk_functions.summarize_cwalk(cvg_mr, 'coverage')

param_estimate_df['coverage'] = cvg_beta_samples

param_summary_df = pd.concat([param_summary_df,
                              cvg_summary],
                             axis=0)

# Cost Discount Rate
cdr_df = df[df['sens_variable'] == 'cost_disc_rate'].copy()
cdr_df = cdr_df.rename(
    {'value' + i: 'cost_disc_rate' + i for i in ['_sens', '_ref']},
    axis=1)

cdr_df = crosswalk_functions.create_diff_variables(
    cdr_df,
    resp_name='log_icer',
    se_name='log_icer_se',
    cov_name='cost_disc_rate')

plt.scatter(x=cdr_df['cost_disc_rate_diff'].to_numpy(),
            y=cdr_df['log_icer_diff'].to_numpy(),
           )
plt.savefig(pth.PLOT_DIR + 'cost_disc_rate_cwalk_plot.png')
plt.clf()

cdr_mr = crosswalk_functions.cwalk(
    cdr_df, resp_name='log_icer', se_name='log_icer_se',
    cov_name='cost_disc_rate', study_id='ArticleID'
    )

cdr_beta_samples, cdr_summary = crosswalk_functions.summarize_cwalk(
    cdr_mr, 'cost_disc_rate'
    )

param_estimate_df['cost_disc_rate'] = cdr_beta_samples

param_summary_df = pd.concat([param_summary_df,
                              cdr_summary],
                             axis=0)

# # Q/DALYs Discount Rate
bdr_df = df[df['sens_variable'] == 'burden_disc_rate'].copy()
bdr_df = bdr_df.rename(
    {'value' + i: 'burden_disc_rate' + i for i in ['_sens', '_ref']},
    axis=1)

bdr_df = crosswalk_functions.create_diff_variables(
    bdr_df,
    resp_name='log_icer',
    se_name='log_icer_se',
    cov_name='burden_disc_rate')

plt.scatter(x=bdr_df['burden_disc_rate_diff'].to_numpy(),
            y=bdr_df['log_icer_diff'].to_numpy(),
           )
plt.savefig(pth.PLOT_DIR + 'burden_disc_rate_cwalk_plot.png')
plt.clf()

bdr_mr = crosswalk_functions.cwalk(
    bdr_df, resp_name='log_icer', se_name='log_icer_se',
    cov_name='burden_disc_rate', study_id='ArticleID'
    )

bdr_beta_samples, bdr_summary = crosswalk_functions.summarize_cwalk(
    bdr_mr, 'burden_disc_rate'
    )

param_estimate_df['burden_disc_rate'] = bdr_beta_samples

param_summary_df = pd.concat([param_summary_df,
                              bdr_summary],
                             axis=0)

mean_params = param_estimate_df.mean(axis=0)
mean_params.loc['draw'] = 1000
param_estimate_df = pd.concat([param_estimate_df, pd.DataFrame(mean_params).T])

param_summary_df = param_summary_df.reset_index().rename(
    {'index': 'covariate'},
    axis=1
    )

if not os.path.exists(pth.CWALK_PARAM_DRAWS):
    print('writing output to ' + pth.CWALK_PARAM_DRAWS)
    param_estimate_df.to_csv(pth.CWALK_PARAM_DRAWS, index=False)

if not os.path.exists(pth.CWALK_PARAM_SUMMARY):
    print('writing output to ' + pth.CWALK_PARAM_SUMMARY)
    param_summary_df.to_csv(pth.CWALK_PARAM_SUMMARY, index=False)
