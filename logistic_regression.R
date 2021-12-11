library(data.table)
library(lme4)

###############################################
# If running interactively, must set working directory to the directory containing this file.
paths <- readLines("./paths.py")
paths <- grep("=", paths, value=TRUE)
paths <- gsub("= ?", "= paste0(", paths)
paths <- gsub(" ?\\+", ",", paths)
paths <- paste0(paths, ")")
for(pth in paths){
  eval(parse(text=pth))
}
preds_df_path <- paste0(MODEL_RESULTS_DIR, "predictions.csv")
metareg_param_path <- paste0(MODEL_RESULTS_DIR, "final_model_parameter_summary.csv")
logit_reg_metrics_path <- paste0(MODEL_RESULTS_DIR, "logistic_regression_metrics.csv")
output_file_name <- paste0(MODEL_RESULTS_DIR, "predictions_with_adj.csv")
###############################################

glmer_wrapper <- function(frmla,
                          dat,
                          max_refits = 10,
                          fam = "binomial"
){
  ## Stubbornly tries to fit the model, specified by the lme4-format formula frmla, on the
  ## data.frame dat. It first fits the model, then checks convergence. If it did not
  ## converge, it then retries with lower tolerances both for the change in the log-likelihood
  ## and for the change in the parameter values, and starting where the previous attempt
  ## stopped.
  require(lme4)
  mdl <- suppressWarnings(
    glmer(frmla, family = fam, data = dat))
  converged <- is.null(mdl@optinfo$conv$lme4$code)
  if(!converged){
    i <- 1
    last_params <- mdl@theta
    last_step_sizes <- 1
    tol <- 10^(-8) # Lower tolerance will prevent stopping before the gradient is close enough to 0
    while(!converged){
      if(i > max_refits & !converged){
        
        print(paste0("model never converged."))
        return(NULL)
      }
      
      mdl <- suppressWarnings(glmer(frmla,
                                    family = fam,
                                    data = dat,
                                    start = mdl@theta,
                                    control = lmerControl(optCtrl = list("maxit" = 10000,
                                                                         "ftol_abs" = tol,
                                                                         "xtol_abs" = tol))))
      converged <- is.null(mdl@optinfo$conv$lme4$code)
      i <- i + 1
    }
  }
  return(mdl)
}

metareg_cvts <- fread(metareg_param_path)[, covariate]

resp_name <- "log_icer_usd"
spline_cov <- "log_GDP_2017usd_per_cap"

df <- fread(CLEANED_REG_DF)
df[, cost_saving := as.integer(is.na(get(resp_name)))]

metareg_cvts <- c(spline_cov,
                  setdiff(metareg_cvts, c("new_spline_cov", "intercept", "qalys")))
metareg_cvts <- setdiff(metareg_cvts, "screen_comparator")

frmla <- paste0("cost_saving ~ ", paste(metareg_cvts, collapse = " + "))
frmla <- paste0(frmla, " + (1 | ArticleID)")

mdl <- glmer_wrapper(as.formula(frmla), dat = df)

preds <- fread(preds_df_path)

preds[, ArticleID := "1"]
preds[, pred_prob := predict(mdl, newdata = preds,
                             type = "response",
                             allow.new.levels = TRUE)]
preds[, pred_val := pred_prob >= 0.5]
df[, pred_prob := predict(mdl, newdata = df,
                          type = "response",
                          re.form = ~0)]
df[, pred_prob_with_re := predict(mdl, newdata=df,
                                  type="response", re.form = ~(1 | ArticleID))]
df[, pred_val := pred_prob >= 0.5]

confus_mtx <- df[, table(cost_saving, pred_val)]
# This line shouldn't be necessary, but it ensures that rows & columns are in correct order (0 before 1, FALSE before TRUE)
confus_mtx <- confus_mtx[sort(rownames(confus_mtx)), sort(colnames(confus_mtx))]

metrics <- sum(diag(confus_mtx)) / sum(confus_mtx)
metrics <- c(metrics, diag(confus_mtx)/rowSums(confus_mtx))
names(metrics) <- c("accuracy", "sensitivity", "specificity")

preds[, adj_ICER := (1-pred_prob) * predicted_icer_usd]
preds[, adj_ICER_lower := (1-pred_prob) * predicted_icer_usd_lower]
preds[, adj_ICER_upper := (1-pred_prob) * predicted_icer_usd_upper]

# adding GDP cutoffs
gdp_cat <- "GDP_category"
preds[, GDP_per_cap := exp(get(spline_cov))]
preds[, GDP_category := cut(adj_ICER/GDP_per_cap, c(-Inf, 0.5, 1, 3, Inf))]
preds[, GDP_category := as.character(GDP_category)]
preds[, GDP_category := gsub("\\(|\\]|", "", GDP_category)]
preds[, GDP_category := gsub("-Inf,", "< ", GDP_category)]
preds[grepl(",Inf", GDP_category),
      GDP_category := paste0("> ", gsub(",Inf", "", GDP_category))]
preds[, GDP_category := gsub(",", " - ", GDP_category)]

if(!file.exists(output_file_name)){
  fwrite(preds, output_file_name)
  print(paste0("Wrote output to ", output_file_name))
}

metrics_df <- data.table(value=metrics)
metrics_df[, metric := names(..metrics)]
setcolorder(metrics_df, c("metric", "value"))

if(!file.exists(logit_reg_metrics_path)){
  fwrite(metrics_df, logit_reg_metrics_path)
  print("Wrote logistic regression metrics to ", logit_reg_metrics_path)
}
