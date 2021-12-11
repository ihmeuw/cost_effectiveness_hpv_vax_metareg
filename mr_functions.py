import warnings
import pickle as pkl
import sys
import numpy as np
import pandas as pd
import mrtool

def select_covariates(df, candidate_covs, include_covs,
                      resp_name, se_name, study_id_name,
                      beta_gprior=None):
    """
    Selects and covariates from candidate_covs and returns them as a list.

    Args:
        df (pd.DataFrame): the data.
        candidate_covs (list of strings): each of which is the column index
            of a column of df whose inclusion in the model will be determined
            by lasso.
        include_covs (list of strings): columns of a covariates automatically
            included in the model.
        resp_name (string): denotes column with the response.
        se_name (string): denotes column with the SE of the response.
        study_id_name (string): column that defines random effect groups
    Returns:
        list of strings denoting covariates selected by CovFinder.
    """
    if not any([v == 'intercept' for v in include_covs]):
        include_covs = include_covs + ['intercept']
    if not any([v == 'intercept' for v in df.columns.values]):
        df['intercept'] = np.ones(df.shape[0])

    # Standardize the covariates
    norm_df = df.copy()
    for cov in candidate_covs:
        norm_df[cov] = (df[cov] - df[cov].mean())/df[cov].std()

    all_covs = include_covs + candidate_covs

    mrd = mrtool.MRData(obs=norm_df[resp_name].to_numpy(),
                        obs_se=norm_df[se_name].to_numpy(),
                        covs={v: norm_df[v].to_numpy() for v in all_covs},
                        study_id=norm_df[study_id_name].to_numpy())

    # Note that it selects covariates without a spline even if the model will be fit
    # with a spline on an include_cov.
    cfinder = mrtool.CovFinder(data=mrd,
                               covs=candidate_covs,
                               pre_selected_covs=include_covs,
                               beta_gprior=beta_gprior,
                               normalized_covs=True,
                               num_samples=1000,
                               laplace_threshold=1e-5,
                               power_range=(-8, 8),
                               power_step_size=0.5)

    cfinder.select_covs()

    return cfinder.selected_covs


def fit_with_covs(df, covs,
                  resp_name, se_name, study_id_name,
                  data_id_name=None,
                  z_covs=['intercept'],
                  trim_prop=0.0, spline_cov=None, spline_degree=None,
                  spline_knots_type=None, spline_knots=None,
                  spline_monotonicity=None,
                  full_spline_basis=False,
                  gprior_dict=None,
                  uprior_dict=None,
                  inner_max_iter=2000, outer_max_iter=1000):
    """
    Fits a model using the specified covariates.

    Args:
        df (pd.DataFrame): the data.
        covs (list of strings): columns containing covariate values
        resp_name (string): column containing the response
        se_name (string): column containing the SE of the response
        study_id_name (string): column that defines random effect groups
        z_covs (list of strings): list of covariates with random slopes
        trim_prop (float between 0.0 and 1.0): proportion of data points to trim
        spline_cov (string or None): If a string, then the covariate to be
            fit with a spline. If None, then indicates no spline should be fit
        spline_degree (int > 0): degree of the spline. None if spline_cov is None
        spline_knots_type (string, either 'domain' or 'frequency'): whether knots
            should be evenly spaced values or quantiles
        spline_knots (np.ndarray): locations of knots either as quantiles or as
            proportions of the distance from min to the max observed value.
        gprior_dict (None or dictionary of np.arrays or lists of np.arrays):
            dictionary whose keys are the entries of covs and whose values are
            np.arrays specifying priors to be placed on parameters. Priors for
            spline_cov are specified using a numpy.array of shape (2, k) where k
            is the number of parameters estimated for spline_cov. Priors for covariates
            not in z_covs must be specified using a np.array of shape (2,) whose
            entries are the mean and sd of the gaussian prior, respectively.
            Priors for z_covs must be lists of length 2, whose first entry
            specifies the prior for the fixed effect, and whose second specifies
            the prior for the random effect.
        uprior_dict (None or dictionary of np.arrays or lists of np.arrays): dictionary
            whose keys are 
        inner_max_iter (int): maximum number of iterations for the inner loop.
        outer_max_iter (int): maximum number of iterations for the outer loop.
    Returns:
        MRBRT object

    """
    z_covs = list(set(z_covs + ['intercept']))
    if not any(df.columns.values == 'intercept'):
        df['intercept'] = 1.0
    elif not all(df['intercept'] == 1.0):
        sys.exit('Columns of df labeled intercept must only have entries of 1.0')

    if full_spline_basis:
        if uprior_dict is None:
            uprior_dict = {}
        uprior_dict.update({'intercept': np.array([0.0, 0.0])})

    if data_id_name is None:
        data_id_name = 'DataID'
        df = df.assign(DataID=np.arange(df.shape[0]))
    mrd = mrtool.MRData(obs=df[resp_name].to_numpy(),
                        obs_se=df[se_name].to_numpy(),
                        covs={v: df[v].to_numpy() for v in covs},
                        study_id=df[study_id_name].to_numpy(),
                        data_id=df[data_id_name].to_numpy()
                       )

    cov_model_args = {v: {'alt_cov': v, 'use_re': (v in z_covs)} for v in covs if v != spline_cov}

    if gprior_dict is not None:
        for key, val in gprior_dict.items():
            if key in z_covs:
                cov_model_args[key].update({'prior_beta_gaussian': val[0],
                                            'prior_gamma_gaussian': val[1]
                    })
            else:
                cov_model_args[key].update({'prior_beta_gaussian': val})

    if uprior_dict is not None:
        for key, val in uprior_dict.items():
            cov_model_args[key].update({'prior_beta_uniform': val})

    cov_model_list = [mrtool.LinearCovModel(**x) for x in cov_model_args.values()]
    if spline_cov is not None:
        n_knots = spline_knots.shape[0] if spline_degree is not None else None
        spline_degree = int(spline_degree)
        prior_spline_maxder = None if gprior_dict is None else gprior_dict.get(spline_cov)
        spline_cov_model = mrtool.LinearCovModel(spline_cov, use_re=False,
                                                 use_spline=True,
                                                 spline_degree=spline_degree,
                                                 spline_knots_type=spline_knots_type,
                                                 spline_knots=spline_knots,
                                                 spline_r_linear=(spline_degree > 1),
                                                 spline_l_linear=(spline_degree > 1),
                                                 use_spline_intercept=full_spline_basis,
                                                 prior_spline_monotonicity=spline_monotonicity,
                                                 prior_spline_maxder_gaussian=prior_spline_maxder)
        cov_model_list.append(spline_cov_model)

    mr = mrtool.MRBRT(data=mrd,
                      cov_models=cov_model_list,
                      inlier_pct=1-trim_prop)
    mr.fit_model(inner_print_level=5,
                 inner_max_iter=inner_max_iter,
                 outer_max_iter=outer_max_iter)
    return mr

def summarize_parameters(mr,
                         spline_cov='log_gdp_pc_17_ppp',
                         beta_samples=None,
                         num_draws=10000):
    cov_names = [v + '_' + str(i) for v in mr.fe_soln.keys() for i in range(mr.fe_soln.get(v).shape[0])]
    for i in range(len(cov_names)):
        if cov_names[i][:-2] != spline_cov:
            cov_names[i] = cov_names[i][:-2]
        betas = np.concatenate(list(mr.fe_soln.values()))
    use_draws_method = beta_samples is not None
    if use_draws_method:
        beta_ses = beta_samples.std(axis=0)
        beta_var = beta_ses**2
    else:
        lme_specs = mrtool.core.other_sampling.extract_simple_lme_specs(mr)
        hessn = mrtool.core.other_sampling.extract_simple_lme_hessian(lme_specs)
        
        beta_var = np.diag(np.linalg.inv(hessn))
        beta_ses = np.sqrt(beta_var)
    summary_df = pd.DataFrame([(mr.cov_model_names[i], bta, bta_se)
                               for i, x_var_idx in enumerate(mr.x_vars_indices)
                               for (bta, bta_se) in zip(mr.beta_soln[x_var_idx], beta_ses[x_var_idx])],
                              columns=['covariate', 'beta', 'beta_se'])
    summary_df['beta_variance'] = summary_df['beta_se']**2
    summary_df['gamma'] = np.concatenate([
        mr.gamma_soln[mr.z_vars_indices[mr.get_cov_model_index(cov_name)]]
        if mr.cov_models[mr.get_cov_model_index(cov_name)].use_re
        else np.repeat(np.nan, mr.cov_models[mr.get_cov_model_index(cov_name)].num_x_vars)
        for cov_name in mr.cov_model_names
    ])
    return summary_df

def predict(pred_df, mr):
    """
    Predicts response values for the data frame pred_df using the model fit mr.
    """
    pred_mrdata = mrtool.MRData(covs={v: pred_df[v].to_numpy() for v in mr.cov_model_names})
    preds = mr.predict(data=pred_mrdata, predict_for_study=False)
    return preds

# def mr_aic(mr):
    # ## This is flawed the penalty doesn't take into account the random effects variance
    # # The LimeTr objective is the -log-likelihood. We want 2*k_beta - 2*log-likelihood.
    # return 2 * mr.lt.objective(mr.lt.soln) + 2 * mr.lt.k_beta

def mr_r_squared(mr, conditional_on_random_fx=True):
    fits = mr.predict(data=mr.data, predict_for_study=conditional_on_random_fx)
    denominator = np.var(mr.data.obs)
    numerator = np.var(fits)
    return numerator / denominator

def k_fold_cv_gaussian_prior(k, df, resp_name, se_name, covs,
                             data_id_name, study_id_name,
                             constrained_covs=None,
                             beta_gpriors=None,
                             combine_gpriors=False,
                             fold_id_name=None,
                             initial_upper_prior_sd=1.0,
                             inner_max_iter=2000, outer_max_iter=1000,
                             sd_tol=1e-6, num_sds_per_step=10,
                             dev=False):
    df['fold_id'] = np.random.randn(df.shape[0])
    df['fold_id'] = pd.qcut(df['fold_id'], k, labels=list(range(k)))
    return cv_gaussian_prior(df, resp_name, se_name, covs,
                             data_id_name, study_id_name,
                             constrained_covs=constrained_covs,
                             beta_gpriors=beta_gpriors,
                             combine_gpriors=combine_gpriors,
                             initial_upper_prior_sd=initial_upper_prior_sd,
                             fold_id_name='fold_id',
                             inner_max_iter=inner_max_iter, outer_max_iter=outer_max_iter,
                             sd_tol=sd_tol, num_sds_per_step=num_sds_per_step,
                             dev=dev)

def cv_gaussian_prior(df, resp_name, se_name, covs,
                      data_id_name, study_id_name,
                      constrained_covs=None,
                      beta_gpriors=None,
                      combine_gpriors=False,
                      fold_id_name=None,
                      initial_upper_prior_sd=1.0,
                      inner_max_iter=2000, outer_max_iter=1000,
                      sd_tol=1e-6, num_sds_per_step=10,
                      dev=False):

    if not any(df.columns.values == 'intercept'):
        df['intercept'] = 1.0
    elif not all(df['intercept'] == 1.0):
        sys.exit('Columns of df labeled intercept must only have entries of 1.0')

    if fold_id_name is None:
        fold_id_name = data_id_name

    stdized_df = df[[fold_id_name, data_id_name, study_id_name, resp_name, se_name] + covs].copy()

    if constrained_covs is None:
        constrained_covs = {}

    unstdized_covs = ['intercept'] + list(constrained_covs.keys())
    if beta_gpriors is not None:
        if combine_gpriors:
            covs_to_stdize = list(set(covs) - set(unstdized_covs))
            for v in beta_gpriors.keys():
                if v in covs_to_stdize:
                    beta_gpriors[v] = beta_gpriors[v] * stdized_df[v].std()
        else:
            unstdized_covs = unstdized_covs + list(beta_gpriors.keys())
            unstdized_covs = list(set(unstdized_covs))
            covs_to_stdize = list(set(covs) - set(unstdized_covs))
    else:
        covs_to_stdize = list(set(covs) - set(unstdized_covs))
        beta_gpriors = {}

    stdized_df[covs_to_stdize] = (stdized_df[covs_to_stdize] - stdized_df[covs_to_stdize].mean(axis=0))
    stdized_df[covs_to_stdize] = stdized_df[covs_to_stdize] / stdized_df[covs_to_stdize].std(axis=0)
    # Create a list of pandas series for the train-test splitting.
    # The ith entry of the list is a boolean Series with value True wherever a row it is in the test
    # set for the ith unique value of the fold-id column and False wherever it is in the training set
    mask_list = [stdized_df[fold_id_name] == fid
            for fid in stdized_df[fold_id_name].unique()]
    # Use the list of masks to create a list of tuples of MRData objects.
    # The ith tuple has first entry that is the training data for the ith fold
    # and second entry for the test set data for the ith fold.
    train_test_mrd_list = [tuple(mrtool.MRData(obs=stdized_df.loc[m == truth_val, resp_name].to_numpy(),
                                               obs_se=stdized_df.loc[m == truth_val, se_name].to_numpy(),
                                               covs={v: stdized_df.loc[m == truth_val, v].to_numpy() for v in covs},
                                               study_id=stdized_df.loc[m == truth_val, study_id_name].to_numpy(),
                                               data_id=stdized_df.loc[m == truth_val, data_id_name].to_numpy())
                           for truth_val in [False, True]) for m in mask_list]
    if not combine_gpriors:
        gprior_cov_models = [mrtool.LinearCovModel(key, use_re=False, prior_beta_gaussian=val)
                             for key, val in beta_gpriors.items()]

    # Initialize empty numpy arrays for the prior SDs & mses
    prior_sds = np.array([], dtype=np.float64)
    mses = np.array([], dtype=np.float64)
    # Initialize bounds for the set of prior SDs for a single iteration
    lower_prior_sd = 1e-4
    upper_prior_sd = initial_upper_prior_sd
    # Create a copy of the number of prior SDs per step variable.
    n_sds_per_step = num_sds_per_step
    if dev:
        import time
    while upper_prior_sd - lower_prior_sd > sd_tol:
        if dev:
            tm = time.time()

        # Generate new values of the prior SD, evenly spaced on a log-scale
        new_prior_sds = np.geomspace(lower_prior_sd, upper_prior_sd, n_sds_per_step)
        # After the first iteration, increment the number of SDs variable by 2. This avoids
        # repeating the same value of SD in subsequent iterations, since the upper and lower
        # bounds of SD are defined by values of SD in the previous iteration.
        if n_sds_per_step == num_sds_per_step:
            n_sds_per_step += 2
        new_prior_sds = np.setdiff1d(new_prior_sds, prior_sds)

        if combine_gpriors:
            # Update the priors on covariates with gaussian priors specified in beta_gpriors by
            # taking a precision-weighted mean of the specified prior and the regularization prior
            # for the mean, and the standard deviation that results from adding the precisions of
            # the prior specified in beta_gpriors and the regularization prior.
            # Note that the result of this is that large amounts of regularization pull the
            # parameter estimates of all covariates towards 0 regardless of whether or not they
            # have priors specified in beta_gpriors.
            gprior_cov_model_lists = [
                    [mrtool.LinearCovModel(key, use_re=False,
                                           prior_beta_gaussian=np.array(
                                               [val[0] / (1 + (val[1]**2 / sd**2)),
                                               val[1] * sd / np.sqrt(val[1]**2 + sd**2)
                                               ]))
                     for key, val in beta_gpriors.items()]
                     for sd in new_prior_sds]
        else:
            gprior_cov_model_lists = [gprior_cov_models for sd in new_prior_sds]
       
        const_cov_model_lists = [[mrtool.LinearCovModel(key, use_re=False,
                                                        prior_beta_uniform=np.full((2,), val))
                                  for key, val in constrained_covs.items()]
                                 for sd in new_prior_sds]

        # Create a list of lists of covariate models. The ith entry of the outer list is a list
        # of all the covariate models for the ith value of SD.
        cov_model_lists = [[mrtool.LinearCovModel('intercept', use_re=True)] + \
                           [mrtool.LinearCovModel(v, use_re=False, prior_beta_gaussian=np.array([0, sd]))
                                for v in covs if v not in ['intercept'] +\
                                                            list(constrained_covs.keys()) +\
                                                            list(beta_gpriors.keys())] +\
                            gprior_cov_model_lists[i] + const_cov_model_lists[i]
                            for i, sd in enumerate(new_prior_sds)]
        new_mses = np.array(
                [
                    np.array([get_mse(train_mrd, test_mrd, cmod_list)
                        for train_mrd, test_mrd in train_test_mrd_list
                        ]).mean()
                    for cmod_list in cov_model_lists
                    ]
                )

        prior_sds = np.hstack([prior_sds, new_prior_sds])
        mses = np.hstack([mses, new_mses])
        ordr = np.argsort(prior_sds)
        prior_sds = prior_sds[ordr]
        mses = mses[ordr]

        min_index = np.argmin(mses)
        lower_prior_sd = prior_sds[min_index - 2 if min_index > 1 else 0]
        upper_prior_sd = prior_sds[min_index + 2
                if min_index < prior_sds.shape[0] - 2
                else prior_sds.shape[0] - 1]

        if dev:
            print('iteration took ' + str(time.time() - tm))
            print('just ran for ' + str(new_mses.shape[0]) + ' values of lambda')
    return (prior_sds, mses)

def get_mse(train_mrd, test_mrd,
            cov_model_list,
            inner_max_iter=1000,
            outer_max_iter=2000):

    mr = mrtool.MRBRT(data=train_mrd, cov_models=cov_model_list)
    mr.fit_model(inner_print_level=5,
                inner_max_iter=inner_max_iter,
                outer_max_iter=outer_max_iter)
    preds = mr.predict(test_mrd, predict_for_study=False)
    mse = ((preds - test_mrd.obs)**2).mean()
    return mse

def fit_signal_model(df,
                     resp_name, se_name, spline_cov, study_id_name, data_id_name,
                     other_cov_names, other_cov_gpriors=None,
                     h=0.1, num_samples=20, deg=2, n_i_knots=2, knots_type='domain',
                     knot_bounds = np.array([[0.1, 0.6], [0.4, 0.9]]),
                     interval_sizes = np.array([[0.1, 0.7], [0.1, 0.7], [0.1, 0.7]]),
                     ):

    cov_dict = {spline_cov: df[spline_cov].to_numpy()}
    cov_dict = {w: df[w].to_numpy() for w in other_cov_names + [spline_cov]}

    covs_with_prior = list(other_cov_gpriors.keys())

    cov_model_list = []
    for v in other_cov_names:
        if v in covs_with_prior:
            cov_model_list.append(mrtool.LinearCovModel(v, use_re=False,
                                                        prior_beta_gaussian=other_cov_gpriors[v])
                                 ) 
        else:
            cov_model_list.append(mrtool.LinearCovModel(v, use_re=False))

    mrd = mrtool.MRData(obs=df[resp_name].to_numpy(),
                        obs_se=df[se_name].to_numpy(),
                        study_id=df[study_id_name].to_numpy(),
                        data_id=df[data_id_name].to_numpy(),
                        covs=cov_dict)

    ensemble_cov_model = mrtool.LinearCovModel(alt_cov=spline_cov,
                                               use_spline=True,
                                               spline_degree=deg,
                                               spline_knots_type=knots_type,
                                               spline_knots=np.linspace(0.0, 1.0, n_i_knots + 2),
                                               spline_r_linear=True,
                                               spline_l_linear=True,
                                               use_spline_intercept=False,
    #                                            prior_spline_monotonicity='increasing',
                                               use_re=False)
    knot_samples = mrtool.core.utils.sample_knots(num_intervals=n_i_knots + 1,
                                                  knot_bounds=knot_bounds,
                                                  interval_sizes=interval_sizes,
                                                  num_samples=num_samples)


    signal_mr = mrtool.MRBeRT(data=mrd,
                       ensemble_cov_model=ensemble_cov_model,
                       ensemble_knots=knot_samples,
                       cov_models=cov_model_list,
                       inlier_pct=1.0 - h)

    signal_mr.fit_model(inner_max_iter=1000, outer_max_iter=2000)

    return signal_mr

def create_signal(signal_mr, spline_cov, spline_cov_values,
                  data_id_name, data_ids):

    pred_covs = {spline_cov: spline_cov_values}
    cov_model_names = [x.name for x in signal_mr.cov_models]
    pred_covs.update({v: np.zeros(spline_cov_values.shape[0])
                      for v in cov_model_names if v != spline_cov})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred_mrd = mrtool.MRData(obs=np.ones(data_ids.shape),
                                 obs_se=np.ones(data_ids.shape),
                                 study_id=np.arange(data_ids.shape[0]),
                                 data_id=data_ids,
                                 covs=pred_covs)

    new_spline_cov = signal_mr.predict(pred_mrd, predict_for_study=False)
    # w = np.hstack([mdl.w_soln[:, None] for mdl in signal_mr.sub_models]).dot(signal_mr.weights)
    
    # pred_df = pred_mrd.to_df().assign(new_spline_cov=new_spline_cov, data_id=pred_mrd.data_id, w=w)
    pred_df = pred_mrd.to_df().assign(new_spline_cov=new_spline_cov,
                                      data_id=pred_mrd.data_id)
    pred_df = pred_df[['data_id', 'new_spline_cov']]
    pred_df = pred_df.rename({'data_id': data_id_name,
                             }, axis=1)
    return pred_df

def get_ws(signal_mr, data_id_name):
    w = np.hstack([mdl.w_soln[:, None]
                   for mdl in signal_mr.sub_models
                  ]).dot(signal_mr.weights)
    pred_df = signal_mr.data.to_df().assign(**{data_id_name: signal_mr.data.data_id, 'w': w})
    pred_df = pred_df[[data_id_name, 'w']]
    pred_df = pred_df.rename({'data_id': data_id_name,
                             }, axis=1)
    return pred_df


def create_fit_df(mr, df, resp_name, study_id_name, other_id_col_names, data_id_name):

    fit_mrd = mrtool.MRData(obs=np.zeros((df.shape[0],)),
            obs_se=np.zeros((df.shape[0],)),
            covs={v: df[v].to_numpy() for v in mr.data.covs.keys()},
            study_id=df[study_id_name].to_numpy(),
            data_id=df[data_id_name].to_numpy(),
            )
    fit_df = fit_mrd.to_df().rename({'study_id': study_id_name}, axis=1)
    fit_df[data_id_name] = fit_mrd.data_id
    fit_df = fit_df.merge(df[[study_id_name, data_id_name, resp_name] + other_id_col_names],
            on=[study_id_name, data_id_name])

    fit_df['fitted_fe_only'] = mr.predict(data=fit_mrd, predict_for_study=False)
    fit_df['fitted_fe_and_re'] = mr.predict(data=fit_mrd, predict_for_study=True)

    fit_df = fit_df[[study_id_name, data_id_name] +\
            other_id_col_names +\
            [resp_name, 'fitted_fe_only', 'fitted_fe_and_re']]

    return fit_df


def r2(mr, fit_df, resp_name):
    rmses = fit_df[[resp_name, 'fitted_fe_and_re', 'fitted_fe_only']].copy()
    rmses['fitted_fe_and_re'] = rmses['fitted_fe_and_re'] - rmses[resp_name]
    rmses['fitted_fe_only'] = rmses['fitted_fe_only'] - rmses[resp_name]

    rmses = np.sqrt(rmses[['fitted_fe_and_re', 'fitted_fe_only']].var(axis=0))
    rmses.name = 'RMSE'

    r2s = (fit_df[[resp_name, 'fitted_fe_and_re', 'fitted_fe_only']].corr()**2)
    r2s = r2s.loc[resp_name, ['fitted_fe_and_re', 'fitted_fe_only']]
    r2s.name = 'R_squared'
    r2s = pd.DataFrame(r2s, columns=['R_squared'])
    r2s = r2s.join(rmses)
    r2s.loc['fitted_fe_only', 'Sample_size'] = fit_df.shape[0]

    return r2s

def create_predictions(mr, signal_mr, preds_df,
                       resp_name, se_name, selected_covs,
                       study_id_name, data_id_name,
                       beta_samples=None, n_samples=1000, seed=24601):

    spline_cov = signal_mr.ensemble_cov_model_name

    if 'new_spline_cov' not in preds_df.columns:
        preds_df['idx'] = np.arange(preds_df.shape[0])
        signal_preds_df = create_signal(signal_mr, spline_cov,
                                        preds_df[spline_cov].to_numpy(),
                                        'idx', preds_df['idx'].to_numpy())
        preds_df = preds_df.merge(signal_preds_df, on='idx')

    if any([v not in preds_df.columns for v in selected_covs]):
        print([v for v in selected_covs if v not in preds_df.columns])
        print(spline_cov)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds_mrd = mrtool.MRData(obs=np.zeros((preds_df.shape[0],)),
                                  obs_se=np.zeros((preds_df.shape[0],)),
                                  data_id=np.arange(preds_df.shape[0]),
                                  study_id=np.ones(preds_df.shape[0]),
                                  covs={v: preds_df[v].to_numpy()
                                        for v in selected_covs}
                                  )

    # Predictions on the log scale
    preds_df['predicted_' + resp_name] = mr.predict(preds_mrd,
                                                    predict_for_study=False,
                                                    sort_by_data_id=True)
    
    np.random.seed(seed)
    if beta_samples is None:
        beta_samples = mrtool.core.other_sampling.sample_simple_lme_beta(n_samples, mr)
    # Generate 1000 draws for UIs.
    preds_draws = mr.create_draws(data=preds_mrd, beta_samples=beta_samples,
                                  gamma_samples=np.full((beta_samples.shape[0], 1),
                                                        mr.gamma_soln))
    preds_df['log_icer_stdev'] = preds_draws.std(axis=1)
    # Calculate quantiles of the draws.
    ci_preds = np.quantile(preds_draws, [0.5, 0.025, 0.975, 0.05, 0.95], axis=1).T

    ci_suffixes = ['_median', '_lower',  '_upper', '_90_lower', '_90_upper']
    preds_df[['predicted_' + resp_name + v for v in ci_suffixes]] = ci_preds

    # Calculate the mean on the ICER scale by taking the average of the converted predictions.
    preds_df['predicted_' + resp_name.replace('log_', '')] = np.exp(preds_draws).mean(axis=1)
    # Convert quantiles from log scale
    log_columns = ['predicted_' + resp_name + v for v in ci_suffixes]
    lin_columns = [v.replace('log_', '') for v in log_columns]
    preds_df[lin_columns] = np.exp(preds_df[log_columns])

    preds_df = preds_df.drop('idx', axis=1)

    return preds_df
