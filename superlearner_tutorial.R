# ==============================================================================
# Super Learner: A Practical Tutorial - Complete Reproducible R Code
# ==============================================================================
# This script contains all R code from the Overleaf article "Super Learner: 
# A Practical Tutorial", organized by code boxes with their original labels.
# 
# Article: Super Learner: A Practical Tutorial
# Repository: https://git.overleaf.com/6abe3298dba216ce9b241bf0
# 
# To replicate the tutorial:
#   1. Install required packages (Box 1)
#   2. Run sections sequentially - each box builds on previous ones
#   3. Set seeds as shown for exact reproducibility
# ==============================================================================

# ---- PACKAGES ----
# Box: code:install - Installing required packages
# install.packages("SuperLearner")
# install.packages(c(
#   "glmnet", "randomForest", "ranger", "xgboost", "earth", "gam",
#   "gbm", "rpart", "nnet", "e1071", "ipred", "ada",
#   "ggplot2", "ROCR", "pROC", "caret", "data.table", "mlbench",
#   "RhpcBLASctl", "rmutil", "vcd", "nnls", "Matrix"
# ))

# Load required packages
library(SuperLearner)
library(glmnet)
library(randomForest)
library(ranger)
library(xgboost)
library(earth)
library(gam)
library(gbm)
library(rpart)
library(nnet)
library(e1071)
library(ipred)
library(ada)
library(ggplot2)
library(ROCR)
library(pROC)
library(caret)
library(data.table)
library(mlbench)
library(RhpcBLASctl)
library(rmutil)
library(vcd)
library(nnls)
library(Matrix)

# Box: code:load-sl - Loading the SuperLearner package
library(SuperLearner)


# ==============================================================================
# SECTION 1: SIMULATED EPIDEMIOLOGICAL DATA
# ==============================================================================

# Box: code:sim-covariates - Generating simulated covariates
set.seed(2743)
n <- 500

# Covariates (standardised to unit scale)
W1 <- runif(n, 0.5, 1)
W2 <- runif(n, 0, 1)
W3 <- runif(n, 0.25, 0.75)
W4 <- runif(n, 0, 1)

# A derived comorbidity indicator
W5 <- rbinom(n, 1, 1 / (1 + exp(1.5 * W2 - W3)))

# Box: code:sim-outcome - Generating the binary outcome
# True data-generating mechanism: interactions + non-linearity
logit_p <- -0.2 * W5 - 2 * W1 + 4 * W5 * W1 - 1.5 * W2 + sin(W4 * pi)
Y <- rbinom(n, 1, 1 / (1 + exp(-logit_p)))

dat <- data.frame(W1, W2, W3, W4, W5, Y)

# Box: code:descriptive - Descriptive summary of the simulated data
cat("Prevalence of the outcome:", round(mean(Y), 3), "\n")
summary(dat)


# ==============================================================================
# SECTION 2: BASIC SUPER LEARNER
# ==============================================================================

# Train / test split (80% / 20%) - used throughout
set.seed(123)
train_idx <- sample(1:n, 400)
X_train   <- dat[train_idx, 1:5]
Y_train   <- dat[train_idx, 6]
X_test    <- dat[-train_idx, 1:5]
Y_test    <- dat[-train_idx, 6]


# ---- 2.1 Single Algorithm ----
# Box: code:sl-single - Fitting Super Learner with a single algorithm
set.seed(123)

sl_lasso <- SuperLearner(
  Y = Y_train, X = X_train,
  family = binomial(),
  SL.library = "SL.glmnet"
)
sl_lasso

# Box: code:lasso-coef - Extracting LASSO coefficients
# Note: fit_lasso is the internal glmnet fit from SL.glmnet
# Access via the fit object returned by SuperLearner
fit_lasso <- sl_lasso$fitLibrary$SL.glmnet_All$object
round(as.matrix(coef(fit_lasso, s = "lambda.min")), 3)


# ---- 2.2 Two Algorithms ----
# Box: code:sl-two - Fitting Super Learner with two algorithms
set.seed(123)

sl_two <- SuperLearner(
  Y = Y_train, X = X_train,
  family = binomial(),
  SL.library = c("SL.glmnet", "SL.randomForest")
)
sl_two


# ---- 2.3 Full Ensemble with Reference Model ----
# Box: code:sl-reference - Including a reference model in the ensemble
set.seed(123)
sl <- SuperLearner(
  Y = Y_train, X = X_train,
  family = binomial(),
  SL.library = c("SL.mean", "SL.glmnet", "SL.randomForest")
)
sl

# Box: code:sl-time - Checking computational time
# Time it took to fit
sl$times$everything


# ==============================================================================
# SECTION 3: PREDICTIONS AND PERFORMANCE
# ==============================================================================

# ---- 3.1 Generating Predictions ----
# Box: code:predict - Generating predictions on the test set
# Keep ensemble and individual learner predictions
pred <- predict(sl, newdata = X_test, onlySL = FALSE)
str(pred)

# Box: code:head-pred - Examining ensemble predictions
# Ensemble predictions (probability of disease)
head(pred$pred)

# Box: code:head-library - Examining individual learner predictions
# Predictions from each individual learner
head(pred$library.predict)


# ---- 3.2 Manual Weighted Sum Reconstruction ----
# Box: code:manual-prediction - Manual weighted sum reconstruction
# Extract fitted ensemble weights from the Super Learner object
alpha_sl <- sl$coef
names(alpha_sl) <- c("SL.mean", "SL.glmnet", "SL.randomForest")
print(round(alpha_sl, 4))

# Extract individual learner predictions for test set
lib_pred <- pred$library.predict
colnames(lib_pred) <- c("SL.mean", "SL.glmnet", "SL.randomForest")

# Manual weighted sum: psi_SL = sum(alpha_k * psi_k)
manual_sl_pred <- as.vector(lib_pred %*% alpha_sl)

# Compare with predict() output
ensemble_pred <- pred$pred

# Show first 6 predictions
data.frame(
  SL.mean = round(lib_pred[1:6, "SL.mean"], 3),
  SL.glmnet = round(lib_pred[1:6, "SL.glmnet"], 3),
  SL.randomForest = round(lib_pred[1:6, "SL.randomForest"], 3),
  Manual_SL = round(manual_sl_pred[1:6], 3),
  Auto_SL = round(ensemble_pred[1:6], 3),
  Match = round(manual_sl_pred[1:6], 6) == round(ensemble_pred[1:6], 6)
)


# ---- 3.3 Area Under the ROC Curve ----
# Box: code:auc - Computing the AUC on the test set
pred_rocr <- ROCR::prediction(pred$pred, Y_test)

auc_sl <- ROCR::performance(
  pred_rocr,
  measure = "auc",
  x.measure = "cutoff"
)@y.values[[1]]

cat("Test AUC:", round(auc_sl, 4), "\n")

# Box: code:roc-curves - ROC curves for the ensemble and individual algorithms
# Column order of library.predict matches SL.library:
#  1 = SL.mean, 2 = SL.glmnet, 3 = SL.randomForest
auc_mean   <- pROC::auc(Y_test, pred$library.predict[, 1])
auc_glmnet <- pROC::auc(Y_test, pred$library.predict[, 2])
auc_rf     <- pROC::auc(Y_test, pred$library.predict[, 3])

roc_mean   <- pROC::roc(Y_test, pred$library.predict[, 1], quiet = TRUE)
roc_glmnet <- pROC::roc(Y_test, pred$library.predict[, 2], quiet = TRUE)
roc_rf     <- pROC::roc(Y_test, pred$library.predict[, 3], quiet = TRUE)
roc_sl     <- pROC::roc(Y_test, pred$pred, quiet = TRUE)

roc_df <- data.frame(
  FPR   = c(1 - roc_mean$specificities, 1 - roc_glmnet$specificities,
            1 - roc_rf$specificities, 1 - roc_sl$specificities),
  TPR   = c(roc_mean$sensitivities, roc_glmnet$sensitivities,
            roc_rf$sensitivities, roc_sl$sensitivities),
  Model = rep(c(
    paste0("Mean (AUC = ", round(auc_mean, 3), ")"),
    paste0("LASSO (AUC = ", round(auc_glmnet, 3), ")"),
    paste0("Random Forest (AUC = ", round(auc_rf, 3), ")"),
    paste0("SuperLearner (AUC = ", round(auc_sl, 3), ")")
  ), times = sapply(list(roc_mean, roc_glmnet, roc_rf, roc_sl),
                    function(x) length(x$sensitivities)))
)

# Plot ROC curves
ggplot(roc_df, aes(x = FPR, y = TPR, color = Model)) +
  geom_line(size = 1.1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed",
              color = "gray50") +
  scale_color_manual(values = c("#3498DB", "#E74C3C", "#9B59B6", "#2ECC71")) +
  labs(x = "1 - Specificity", y = "Sensitivity") +
  xlim(0, 1) + ylim(0, 1) +
  theme_minimal(base_size = 14) +
  theme(legend.position = c(0.65, 0.25))


# ==============================================================================
# SECTION 4: CROSS-VALIDATED SUPER LEARNER
# ==============================================================================

# ---- 4.1 Fitting CV.SuperLearner ----
# Box: code:cvsl - Fitting CV.SuperLearner
set.seed(123)

system.time({
  cv_sl <- CV.SuperLearner(
    Y = Y_train, X = X_train,
    family = binomial(),
    V = 5,
    SL.library = c("SL.mean", "SL.glmnet", "SL.randomForest")
  )
})

# Box: code:cvsl-summary - Summarizing CV.SuperLearner results
summary(cv_sl)


# ---- 4.2 Discrete Super Learner ----
# Box: code:discrete-sl - Identifying the Discrete Super Learner
table(simplify2array(cv_sl$whichDiscreteSL))


# ---- 4.3 Distribution of Ensemble Weights ----
# Box: code:review-weights - Distribution of ensemble weights across folds
review_weights <- function(cv_sl) {
  meta_weights <- coef(cv_sl)              # matrix: folds x learners
  means <- colMeans(meta_weights)
  sds   <- apply(meta_weights, 2, sd)
  mins  <- apply(meta_weights, 2, min)
  maxs  <- apply(meta_weights, 2, max)
  stats <- cbind("mean(weight)" = means, sd = sds, min = mins, max = maxs)
  stats[order(stats[, 1], decreasing = TRUE), ]
}

print(review_weights(cv_sl), digits = 3)


# ==============================================================================
# SECTION 5: HYPERPARAMETER TUNING
# ==============================================================================

# ---- 5.1 Algorithm Variants ----
# Box: code:rf-wrapper - Random Forest variant with custom wrapper
SL.rf.better <- function(...) {
  SL.randomForest(..., ntree = 3000)
}

set.seed(123)

cv_sl_rf <- CV.SuperLearner(
  Y = Y_train, X = X_train,
  family = binomial(),
  V = 5,
  SL.library = c(
    "SL.mean",
    "SL.glmnet",
    "SL.rf.better",
    "SL.randomForest"
  )
)

summary(cv_sl_rf)


# Box: code:create-learner - Generating a variant with create.Learner
learners <- create.Learner(
  "SL.randomForest",
  params = list(ntree = 3000)
)

learners$names


# ---- 5.2 Hyperparameter Grids ----
# Box: code:mtry-default - Reference value of mtry
floor(sqrt(ncol(X_train)))


# Box: code:mtry-grid - Generating mtry configurations
mtry_seq <- floor(
  sqrt(ncol(X_train)) * c(0.5, 1, 2)
)

learners_mtry <- create.Learner(
  "SL.randomForest",
  tune = list(mtry = mtry_seq)
)

learners_mtry$names


# Box: code:cvsl-mtry - Evaluating mtry configurations with CV.SuperLearner
set.seed(123)

cv_sl_mtry <- CV.SuperLearner(
  Y = Y_train, X = X_train,
  family = binomial(),
  V = 5,
  SL.library = c(
    "SL.mean",
    "SL.glmnet",
    learners_mtry$names,
    "SL.randomForest"
  )
)

summary(cv_sl_mtry)


# ---- 5.3 XGBoost Grid ----
# Box: code:xgb-grid - Generating XGBoost hyperparameters grid
tune_xgb <- list(
  ntrees    = c(200, 500, 1000, 2000, 5000),
  max_depth = 1:4,
  shrinkage = c(0.001, 0.01, 0.1)
)

learners_xgb <- create.Learner(
  "SL.xgboost",
  tune = tune_xgb,
  detailed_names = TRUE,
  name_prefix = "xgb"
)

cat("Number of XGBoost configurations:", length(learners_xgb$names), "\n")


# ==============================================================================
# SECTION 6: FEATURE SCREENING
# ==============================================================================

# ---- 6.1 Correlation-based Screening ----
# Box: code:screen-corp - Feature screening based on correlation
set.seed(123)

cv_sl_screen <- CV.SuperLearner(
  Y = Y_train, X = X_train,
  family = binomial(),
  V = 5,
  SL.library = list(
    "SL.mean",
    "SL.glmnet",
    c("SL.glmnet", "screen.corP")
  )
)

summary(cv_sl_screen)


# ---- 6.2 LASSO-based Screening ----
# Box: code:screen-glmnet - LASSO-based feature screening
set.seed(123)

cv_sl_screen2 <- CV.SuperLearner(
  Y = Y_train, X = X_train,
  family = binomial(),
  V = 5,
  SL.library = list(
    "SL.mean",
    "SL.glmnet",
    c("SL.glmnet", "screen.glmnet")
  )
)

summary(cv_sl_screen2)


# ==============================================================================
# SECTION 7: OPTIMIZING FOR AUC
# ==============================================================================

# ---- 7.1 AUC Optimization ----
# Box: code:cvsl-auc - Optimizing for AUC
set.seed(123)

cv_sl_auc <- CV.SuperLearner(
  Y = Y_train, X = X_train,
  family = binomial(),
  V = 5,
  method = "method.AUC",
  SL.library = list(
    "SL.mean",
    "SL.glmnet",
    c("SL.glmnet", "screen.corP")
  )
)

summary(cv_sl_auc)


# ---- 7.2 Negative Log-Likelihood ----
# Box: code:cvsl-nnloglik - Optimizing for Negative Log-Likelihood
set.seed(123)

cv_sl_nnloglik <- CV.SuperLearner(
  Y = Y_train, X = X_train,
  family = binomial(),
  V = 5,
  method = "method.NNloglik",
  SL.library = c(
    "SL.mean",
    "SL.glmnet",
    "SL.randomForest"
  )
)

summary(cv_sl_nnloglik)


# ==============================================================================
# SECTION 8: PARALLELIZATION
# ==============================================================================

# ---- 8.1 Multicore Execution ----
# Box: code:parallel-setup - Configuring parallel execution
num_cores <- RhpcBLASctl::get_num_cores()
cat("Cores available on this machine:", num_cores, "\n")

options(mc.cores = num_cores)
set.seed(123, "L'Ecuyer-CMRG")


# Box: code:cvsl-parallel - Parallel execution of CV.SuperLearner
system.time({
  cv_sl_par <- CV.SuperLearner(
    Y = Y_train, X = X_train,
    family = binomial(),
    V = 5,
    parallel = "multicore",
    SL.library = c(
      "SL.mean",
      "SL.glmnet",
      "SL.randomForest"
    )
  })
})

# Compare with sequential execution time (run sequentially first)
# summary(cv_sl_par)


# ---- 8.2 mcSuperLearner ----
# Box: code:mcsuperlearner - Parallelized Super Learner fit
set.seed(123, "L'Ecuyer-CMRG")

sl_mc <- mcSuperLearner(
  Y = Y_train, X = X_train,
  family = binomial(),
  SL.library = c(
    "SL.mean",
    "SL.glmnet",
    "SL.randomForest"
  )
)

sl_mc


# ---- 8.3 Cluster Parallel ----
# Box: code:cluster-parallel - Parallel execution using a cluster object
# cl <- parallel::makeCluster(4)
# parallel::clusterExport(cl, c("Y_train", "X_train"))
# 
# set.seed(123)
# 
# cv_sl_snow <- CV.SuperLearner(
#   Y = Y_train, X = X_train,
#   family = binomial(),
#   V = 10,
#   parallel = cl,
#   SL.library = c(
#     "SL.mean",
#     "SL.glmnet",
#     "SL.randomForest"
#   )
# )
# 
# parallel::stopCluster(cl)


# ==============================================================================
# SECTION 9: CUSTOM WRAPPERS
# ==============================================================================

# ---- 9.1 Custom Learner: GBM ----
# Box: code:gbm-wrapper - Custom wrapper for GBM
SL.gbm2 <- function(Y, X, newX, family, obsWeights,
                    gbm.trees = 5000, interaction.depth = 2,
                    shrinkage = 0.001, ...) {
  form <- as.formula(paste("Y ~", paste(colnames(X), collapse = " + ")))
  
  distribution <- switch(family$family,
    gaussian = "gaussian",
    binomial = "bernoulli"
  )
  
  fit.gbm <- gbm::gbm(formula = form, data = X,
    distribution = distribution,
    n.trees = gbm.trees,
    interaction.depth = interaction.depth,
    shrinkage = shrinkage,
    cv.folds = 5, keep.data = TRUE,
    weights = obsWeights, verbose = FALSE)
  
  best.iter <- gbm::gbm.perf(fit.gbm, method = "cv", plot.it = FALSE)
  pred <- predict(fit.gbm, newdata = newX, best.iter, type = "response")
  
  out <- list(pred = pred, fit = list(object = fit.gbm, n.trees = best.iter))
  class(out$fit) <- "SL.gbm"
  return(out)
}


# ---- 9.2 Custom Learner: MARS (earth) ----
# Box: code:earth-wrapper - Custom wrapper for MARS
SL.earth2 <- function(Y, X, newX, family, obsWeights, id,
                      degree = 2, penalty = 3,
                      nk = max(21, 2 * ncol(X) + 1),
                      pmethod = "backward", ...) {
  glm_arg <- NULL
  if (family$family == "binomial" && all(Y %in% c(0, 1))) {
    glm_arg <- list(family = binomial())
  }
  fit.earth <- earth::earth(x = X, y = Y, degree = degree,
    nk = nk, penalty = penalty,
    pmethod = pmethod, glm = glm_arg)
  pred <- predict(fit.earth, newdata = newX, type = "response")
  out  <- list(pred = pred, fit = list(object = fit.earth))
  class(out$fit) <- "SL.earth"
  return(out)
}


# ---- 9.3 Custom Screener ----
# Box: code:custom-screener - Custom variable screening procedure
screen.glmnet3 <- function(Y, X, family, alpha = 1, minscreen = 2,
                           nfolds = 10, nlambda = 200, ...) {
  myfamily <- if (family$family == "binomial" && !all(Y %in% c(0, 1)))
    "gaussian" else family$family
  
  if (!is.matrix(X)) X <- model.matrix(~ -1 + ., data = X)
  
  fitCV <- tryCatch(
    glmnet::cv.glmnet(x = X, y = Y, lambda = NULL,
      type.measure = "deviance", nfolds = nfolds,
      family = myfamily, alpha = alpha,
      nlambda = nlambda, keep = TRUE),
    error = function(e) NULL
  )
  
  whichVariable <- NULL
  if (!is.null(fitCV)) {
    si <- abs((max(fitCV$nzero) - ncol(X) / 2) - fitCV$nzero)
    w2 <- as.numeric(coef(fitCV$glmnet.fit, s = fitCV$lambda.min))[-1] != 0
    w3 <- as.numeric(glmnet::coef.glmnet(
      fitCV, s = fitCV$lambda[si == min(si)][1]))[-1] != 0
    if (sum(w2) > 1)      whichVariable <- w2
    else if (sum(w3) > 1 & sum(w3) < ncol(X) / 2) whichVariable <- w3
  }
  if (is.null(whichVariable)) whichVariable <- screen.cramersv(Y, X)
  whichVariable
}

# Cramer's V fallback screener (works with binary outcome)
screen.cramersv <- function(Y, X, nscreen = 6, cts.num = 10, ...) {
  var_cont <- apply(X, 2, function(x) length(unique(x)) > cts.num)
  cvdata   <- X
  cutf <- function(x) cut(x,
    unique(quantile(x, prob = seq(0, 1, 0.2))),
    include.lowest = TRUE)
  if (sum(var_cont) > 0) cvdata[, var_cont] <- apply(cvdata[, var_cont], 2, cutf)
  cvf <- function(xx) vcd::assocstats(table(Y, xx))$cramer
  whichVariable <- order(apply(cvdata, 2, cvf), decreasing = TRUE) <= nscreen
  whichVariable
}


# ---- 9.4 Incorporating Custom Wrappers ----
# Box: code:cvsl-custom - Evaluating custom wrappers
set.seed(123)

cv_sl_custom <- CV.SuperLearner(
  Y = Y_train, X = X_train,
  family = binomial(), V = 5,
  SL.library = c("SL.mean", "SL.glmnet", "SL.gbm2", "SL.earth2")
)
summary(cv_sl_custom)


# ==============================================================================
# SECTION 10: THEORETICAL FOUNDATION - NNLS META-LEARNER
# ==============================================================================

# Box: code:weighted-procedure - Implementing the weighted cross-validated 
# Super Learner procedure (illustrative with simulated Z matrix)
set.seed(123)
n <- 400
K <- 3
Z <- matrix(runif(n * K), nrow = n, ncol = K)
colnames(Z) <- c("SL.glmnet", "SL.randomForest", "SL.mean")

# Outcome vector
Y <- rbinom(n, 1, 0.44)

# Meta-learner: Non-negative least squares (NNLS) for convex combination
library(nnls)
fit_nnls <- nnls(Z, Y)
alpha <- fit_nnls$x / sum(fit_nnls$x)  # normalize to sum to 1
names(alpha) <- colnames(Z)

# Final Super Learner prediction on new data
# (each candidate refitted on full data)
# Note: This requires sl_glmnet_full, sl_rf_full, X_new defined elsewhere
# psi_glmnet <- predict(sl_glmnet_full, newx = as.matrix(X_new), type = "response", s = "lambda.min")
# psi_rf     <- predict(sl_rf_full, newdata = X_new, type = "prob")[, 2]
# psi_mean   <- rep(mean(Y_train), nrow(X_new))
# 
# # Weighted ensemble prediction
# psi_SL <- alpha["SL.glmnet"] * psi_glmnet + 
#           alpha["SL.randomForest"] * psi_rf + 
#           alpha["SL.mean"] * psi_mean

# Print NNLS weights
print(round(alpha, 3))


# ==============================================================================
# END OF SCRIPT
# ==============================================================================
# This script reproduces all computational steps from the article:
# "Super Learner: A Practical Tutorial"
# 
# For questions or issues, refer to the original Overleaf project.
# ==============================================================================