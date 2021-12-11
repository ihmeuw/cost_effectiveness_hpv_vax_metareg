import numpy as np
import pandas as pd
import warnings
import mrtool
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def plot_quartiles_with_ui(mr, y_axis_var='log_icer', x_axis_var='log_vaccine_cost',
                           data_id_name='RatioID', group_var='log_burden_variable',
                           plot_title='Log ICER vs. log GDP per capita by burden quartile',
                           spline_transform_df=None, spline_var=None,
                           beta_samples=None, gamma_samples=None,
                           # x_on_log_scale=False, y_on_log_scale=False,
                           x_decimals=-1, y_decimals=-1, x_scale=None, y_scale=None,
                           group_var_name_display=None, y_axis_var_display=None, x_axis_var_display=None,
                           outliers=None, out_dir=None, file_name=None):

    df = mr.data.to_df().rename({'obs': y_axis_var}, axis=1)
    df[data_id_name] = mr.data.data_id
    df['w'] = mr.w_soln if outliers is None else np.ones((df.shape[0],))

    if group_var_name_display is None:
        group_var_name_display = group_var
    if y_axis_var_display is None:
        y_axis_var_display = y_axis_var
    if x_axis_var_display is None:
        x_axis_var_display = x_axis_var

    if spline_transform_df is not None:
        var_to_transform = group_var if not group_var in mr.cov_model_names else x_axis_var
        df = df.merge(spline_transform_df[[data_id_name, var_to_transform]],
                      on=data_id_name, how='left')
    else:
        var_to_transform = ''

    quartiles = [1, 2, 3, 4]
    df['quartile'] = pd.qcut(df[group_var], 4, labels=quartiles)

    if var_to_transform == group_var:
        quartile_means = {q: df.loc[df['quartile'] == q, spline_var].mean() for q in quartiles}
    else:
        quartile_means = {q: df.loc[df['quartile'] == q, group_var].mean() for q in quartiles}

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 8))
    plt.title(plot_title, fontsize=24)
    plt.xlabel(x_axis_var_display, fontsize=20)
    plt.ylabel(y_axis_var_display, fontsize=20)
    sns.set_style('white')
    plt.rcParams['axes.edgecolor'] = '0.15'
    plt.rcParams['axes.linewidth'] = 0.5

    colors = ['#253494', '#2c7fb8', '#41b6c4', '#a1d99b']
    color_dict = {q: colors[i] for i, q in enumerate(quartiles)}

    pt_grp_labels = {quartiles[i]: v + ' ' + group_var_name_display + ' quartile'
                     for i, v in enumerate(['1st', '2nd', '3rd', '4th'])}

    if var_to_transform == x_axis_var:
        lines_df = spline_transform_df.sort_values(x_axis_var)
    else:
        x_padding = (np.max(df[x_axis_var]) - np.min(df[x_axis_var])) / 15
        lines_df = pd.DataFrame({x_axis_var: np.array([np.min(df[x_axis_var]) - x_padding,
                                                       np.max(df[x_axis_var]) + x_padding])})
    lines_df = lines_df.assign(**{v: mr.data.covs.get(v).mean()
                                  for v in mr.cov_names
                                  if v not in [x_axis_var, spline_var]})

    lines_df['row_id'] = np.arange(lines_df.shape[0])

    if var_to_transform == group_var:
        covs = {v: lines_df[v].to_numpy() for v in mr.cov_names if v != spline_var}
    else:
        covs = {v: lines_df[v].to_numpy()
                for v in mr.cov_names if v != group_var}

    for q, val in quartile_means.items():
        if var_to_transform == group_var:
            covs.update({spline_var: np.full((lines_df.shape[0],), val)})
        else:
            covs.update({group_var: np.full((lines_df.shape[0],), val)})
        lines_mrd = mrtool.MRData(covs=covs,
                                  data_id=lines_df['row_id'].to_numpy())
        lines_df['prediction_' + str(q)] = mr.predict(lines_mrd, predict_for_study=False,
                                                sort_by_data_id=True)

    if var_to_transform == group_var:
        spline_transform_df = spline_transform_df.sort_values(group_var).reset_index()
        spline_val_at_median = spline_transform_df.loc[spline_transform_df.shape[0] // 2,
                                                       spline_var]
        covs.update({spline_var: np.full((lines_df.shape[0],), spline_val_at_median)})
    else:
        covs.update({group_var: np.full((lines_df.shape[0],), df[group_var].mean())})
    lines_mrd = mrtool.MRData(covs=covs,
                              data_id=lines_df['row_id'].to_numpy())
    lines_df['prediction_mean'] = mr.predict(data=lines_mrd, predict_for_study=False, sort_by_data_id=True)
    draws = mr.create_draws(data=lines_mrd,
                            beta_samples=beta_samples,
                            gamma_samples=np.full((beta_samples.shape[0], 1),
                                                  mr.gamma_soln),
                            random_study=False)
    lines_df[['prediction_lower', 'prediction_upper']] = np.quantile(
        draws,
        [0.025, 0.975], axis=1).T

    mean_x_vals = lines_df[x_axis_var]
    mean_y_vals = lines_df['prediction_mean']
    lower_vals = lines_df['prediction_lower']
    upper_vals = lines_df['prediction_upper']
    ax.plot(mean_x_vals, mean_y_vals, '-', color='#000000', alpha=0.0)
    ax.fill_between(mean_x_vals,
                    lower_vals,
                    upper_vals,
                    color='grey', alpha=0.2,
                    label='95% UI'
                   )

    for q, color in color_dict.items():
        x_vals = df.loc[(df['quartile'] == q) & (df['w'] == 1), x_axis_var]
        y_vals = df.loc[(df['quartile'] == q) & (df['w'] == 1), y_axis_var]
        
        ax.scatter(
            x_vals,
            y_vals,
            s=15,
            facecolors=color_dict[q], linewidth=0.6,
            alpha=.7, label=pt_grp_labels[q])
        x_vals = lines_df[x_axis_var]
        y_vals = lines_df['prediction_' + str(q)]
        ax.plot(x_vals, y_vals, color=color, linewidth=3)

    if outliers is None:
        untrimmed = mr.w_soln
        x_vals = df.loc[untrimmed < 0.5, x_axis_var]
        y_vals = df.loc[untrimmed < 0.5, y_axis_var]
        ax.scatter(x_vals,
                   y_vals,
                   s=15,
                   facecolors='#aa0000',
                   linewidth=0.6, alpha=0.4,
                   label='Trimmed')
    else:
        untrimmed = np.ones((df.shape[0],))
        x_vals = outliers[x_axis_var]
        y_vals = outliers[y_axis_var]
        ax.scatter(x_vals,
                        y_vals,
                        s=15,
                        facecolors='#aa0000',
                        linewidth=0.6, alpha=0.4, label='Trimmed')

    ax.plot(mean_x_vals, mean_y_vals, '-', color='#000000', label='Mean Prediction')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    xlocs, xlabels = plt.xticks()
    if x_scale is not None:
        xlabs = np.exp(xlocs) * x_scale
        xlabs = np.round(xlabs, decimals=x_decimals)
    else:
        xlabs = np.round(np.exp(xlocs), decimals=x_decimals)
    if x_decimals <= 0:
        xlabs = xlabs.astype(int)
    xlabs = [f'{lab:,}' for lab in xlabs]
    plt.xticks(xlocs, xlabs, fontsize=15)
    ylocs, ylabs = plt.yticks()
    if y_scale is not None:
        ylabs = np.exp(xlocs) * y_scale
        ylabs = np.round(ylabs, decimals=y_decimals)
    else:
        ylabs = np.round(np.exp(ylocs), decimals=y_decimals)
    if y_decimals <= 0:
        ylabs = ylabs.astype(int)
    ylabs = [f'{lab:,}' for lab in ylabs]
    plt.yticks(ylocs, ylabs, fontsize=15)

    if outliers is not None:
        y_min = np.hstack([df[y_axis_var], outliers[y_axis_var]]).min()
        y_max = np.hstack([df[y_axis_var], outliers[y_axis_var]]).max()
        y_padding = (y_max - y_min) / 15
        y_lims=(y_min - y_padding, y_max + y_padding)
    else:
        y_padding = (df[y_axis_var].min() - df[y_axis_var].max()) / 15
        y_lims = (df[y_axis_var].min() - y_padding, df[y_axis_var].max() + y_padding) 
    plt.xlim((lines_df[x_axis_var].min(), lines_df[x_axis_var].max()))
    plt.ylim(y_lims)

    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')

    if out_dir is not None:
        plt.savefig(out_dir + file_name, bbox_inches='tight', format=file_name[-3:])

def visualize_spline(spline_mr, spline_cov, spline_cov_values, signal_name='Signal',
                     title='Spline Transformation',
                     x_label='GDP per capita',
                     y_label='Signal',
                     x_on_log_scale=False,
                     y_on_log_scale=False,
                     out_file_name=None):
    spline_cov_pred_values = np.linspace(spline_cov_values.min(),
                                         spline_cov_values.max(),
                                         100)
    pred_covs = {spline_cov: spline_cov_pred_values}
    other_covs = [v.name for v in spline_mr.cov_models if v != spline_cov]

    pred_covs.update({v: np.zeros(spline_cov_pred_values.shape[0]) for v in other_covs})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spline_mrd = mrtool.MRData(obs=np.full(spline_cov_pred_values.shape, 1.0),
                                   obs_se=np.full(spline_cov_pred_values.shape, 1.0),
                                   study_id=np.ones(spline_cov_pred_values.shape),
                                   covs=pred_covs)
    new_spline_cov = spline_mr.predict(spline_mrd, predict_for_study=False)

    spline_df = spline_mrd.to_df()[[spline_cov]].assign(**{'new_spline_cov': new_spline_cov})
    spline_df = spline_df.sort_values(spline_cov)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
    ax.plot(np.exp(spline_df[spline_cov]), spline_df['new_spline_cov'])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if x_on_log_scale:
        ax.set_xscale('log')
    if y_on_log_scale:
        ax.set_yscale('log')

    if out_file_name is not None:
        plt.savefig(out_file_name, bbox_inches='tight')
    
