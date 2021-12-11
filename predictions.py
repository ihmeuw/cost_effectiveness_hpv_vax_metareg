import numpy as np
import pandas as pd
import sys
import os
import dill
import paths as pth
import mr_functions

output_dir = pth.MODEL_RESULTS_DIR

response_name = 'log_icer_usd'
pred_icer_name = 'predicted_icer_usd'
spline_cov = 'log_GDP_2017usd_per_cap'

# Load the signal model and final metaregression model objects
signal_mr_file = output_dir + 'signal_mr.pkl'
with open(signal_mr_file, 'rb') as in_file:
    signal_mr = dill.load(in_file)
final_model_file = output_dir + 'final_model.pkl'
with open(final_model_file, 'rb') as in_file:
    mr = dill.load(in_file)
selected_covs = mr.cov_model_names
beta_samples_df = pd.read_csv(output_dir + 'final_mr_beta_samples.csv')
beta_samples = beta_samples_df.to_numpy()

preds_df = pd.read_csv(pth.CLEANED_PREDS_DF)

preds_df['log_vaccine_cost_2017usd'] = np.log(preds_df['vaccine_cost_2017usd'])
preds_df = preds_df[preds_df['log_vaccine_cost_2017usd'].notnull()]

# Generate predictions
preds_df = mr_functions.create_predictions(
    mr, signal_mr, preds_df,
    'log_icer_usd', 'log_icer_se', selected_covs,
    'ArticleID', 'RatioID',
    beta_samples=beta_samples,
    seed=8721)

preds_df['ratio_of_upper_to_lower_prediction'] =\
    preds_df[pred_icer_name + '_upper'] / preds_df[pred_icer_name + '_lower']
preds_df['lancet_label'] = preds_df['lancet_label'].str.lower()

preds_df[pred_icer_name + '_over_gdp_pc'] =\
    preds_df[pred_icer_name] / np.exp(preds_df[spline_cov])
preds_df[pred_icer_name + '_median_over_gdp_pc'] =\
    preds_df[pred_icer_name + '_median'] / np.exp(preds_df[spline_cov])

# Filter unnecessary columns
cols = [
    'country', 'location_id', 'lancet_label', 'ihme_loc_id',
    'region_id', 'region_name', 'super_region_id', 'super_region_name',
    'gavi_eligible',
    spline_cov, 'log_burden_variable', 'log_vaccine_cost_2017usd',
    'payer', 'intercept', 'burden_disc_rate', 'cost_disc_rate',
    'coverage', 'not_lifetime', 'screen_comparator', 'access_to_care_100',
    'quadrivalent', 'qalys', 'both_sex', 'new_spline_cov'
    ]
cols = cols + [pred_icer_name + i for i in ['', '_median', '_lower', '_upper']]
cols = cols + ['predicted_' + response_name + i for i in ['', '_lower', '_upper']]
cols = cols + [pred_icer_name + i + '_over_gdp_pc' for i in ['', '_median']]
cols = cols + ['ratio_of_upper_to_lower_prediction']
preds_df = preds_df[cols]

final_preds_path = output_dir + 'predictions.csv'
if not os.path.exists(final_preds_path):
    preds_df.to_csv(final_preds_path, index=False)
    print('Writing to ' + final_preds_path)
else:
    print('Output file already exists.')
    sys.exit()
