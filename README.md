# Super Learner: A Practical Tutorial - Reproducible Code

This repository contains the complete, reproducible R code for the article **"Super Learner: A Practical Tutorial"** originally hosted on Overleaf.

> **Scope note:** this GitHub repo keeps **only** `README.md` + `superlearner_tutorial.R` (+ `LICENSE`).
> All LaTeX source (`main.tex`, `references.bib`), figures (`image/`), and the compiled PDF live in the
> Overleaf project (`https://git.overleaf.com/6abe3298dba216ce9b241bf0`, local mirror: `SUPERLEARNER/SLoverleaf/`).
> Syncs from Overleaf update only these two files.

## Overview

The tutorial provides a comprehensive, practical guide to implementing **Super Learner** in R for applied researchers in epidemiology and biostatistics. Super Learner is an ensemble machine-learning method that uses cross-validation to combine multiple candidate algorithms into a single prediction model, ensuring performance asymptotically at least as good as the best candidate algorithm.

Synced to Overleaf commit `ad15a36` (2026-10-06, "Fix LaTeX errors, broken code blocks, and mismatched output boxes"): **58 code boxes**.

## Repository Contents

| File | Description |
|------|-------------|
| `superlearner_tutorial.R` | **Complete reproducible R script** - all 58 code boxes from the article, organized by section with original box labels |
| `README.md` | This file |
| `LICENSE` | MIT License |

No `.tex`, `.pdf`, or `image/` files are tracked here by design.

## Quick Start

### Prerequisites

- R (≥ 4.0.0)
- Required packages (installed automatically on first run)

### Installation

```r
# Core package
install.packages("SuperLearner")

# Additional packages used throughout the tutorial
install.packages(c(
  "glmnet", "randomForest", "ranger", "xgboost", "earth", "gam",
  "gbm", "rpart", "nnet", "e1071", "ipred", "ada",
  "ggplot2", "ROCR", "pROC", "caret", "data.table", "mlbench",
  "RhpcBLASctl", "rmutil", "vcd", "nnls", "Matrix", "tmle"
))
```

### Running the Tutorial

```r
# Source the complete script
source("superlearner_tutorial.R")

# Or run section by section - each box builds on previous ones
# The script is organized to run sequentially from top to bottom
```

## Tutorial Sections (Code Boxes)

The R script contains all code from the article, organized by the original box labels:

### 0. **Theoretical Foundation**
- `code:weighted-procedure` - NNLS meta-learner implementation with manual vs matrix verification (standalone simulated Z)

### 1. **Setup & Data Generation**
- `code:install` - Package installation
- `code:load-sl` - Load SuperLearner
- `code:sim-covariates` - Generate simulated covariates (W1-W5)
- `code:sim-outcome` - Generate binary outcome with interactions/non-linearity
- `code:descriptive` - Descriptive summary

### 2. **Basic Super Learner**
- `code:sl-single` - Single algorithm (SL.glmnet), includes 80/20 train/test split
- `code:lasso-coef` - Extract LASSO coefficients
- `code:sl-two` - Two algorithms (SL.glmnet + SL.randomForest)
- `code:sl-reference` - Full ensemble with reference model (SL.mean)
- `code:sl-time` - Computational time

### 3. **Predictions & Performance**
- `code:predict` - Generate predictions on test set
- `code:head-pred` - Examine ensemble predictions
- `code:head-library` - Examine individual learner predictions
- `code:manual-prediction` - **Manual weighted sum reconstruction** (verifies psi_SL = Σ αₖ ψₖ)
- `code:auc` - Compute AUC on test set
- `code:roc-curves` - ROC curves for ensemble and individual algorithms

### 4. **Cross-Validated Super Learner**
- `code:cvsl` - Fit CV.SuperLearner (V=5)
- `code:cvsl-summary` - Summarize CV results
- `code:discrete-sl` - Identify Discrete Super Learner
- `code:review-weights` - Distribution of ensemble weights across folds

### 5. **Hyperparameter Tuning**
- `code:rf-wrapper` - Random Forest variant (ntree=3000)
- `code:create-learner` - Generate variants with create.Learner()
- `code:mtry-default` - Reference mtry value
- `code:mtry-grid` - Generate mtry configurations
- `code:cvsl-mtry` - Evaluate mtry configurations
- `code:xgb-grid` - Generate XGBoost hyperparameter grid (60 configs)

### 6. **Feature Screening**
- `code:screen-corp` - Correlation-based screening (screen.corP)
- `code:screen-glmnet` - LASSO-based screening (screen.glmnet)

### 7. **Optimizing for AUC**
- `code:cvsl-auc` - AUC optimization (method.AUC, includes SL.randomForest + screened LASSO)
- Note: standalone `method.NNloglik` box was removed in Overleaf; NNloglik is demonstrated in the clinical application below.

### 8. **Parallelization**
- `code:parallel-setup` - Configure multicore execution
- `code:cvsl-parallel` - Parallel CV.SuperLearner
- `code:mcsuperlearner` - mcSuperLearner for prediction
- `code:cluster-parallel` - Cluster-based parallelization (active code, V=10)

### 9. **Custom Wrappers**
- `code:gbm-wrapper` - Custom GBM wrapper (SL.gbm2)
- `code:earth-wrapper` - Custom MARS wrapper (SL.earth2)
- `code:custom-screener` - Custom LASSO + Cramér's V screener
- `code:cvsl-custom` - Evaluate custom wrappers

### 10. **Full Epidemiological Application (CVD cohort, n=1000)**
- `code:clinical-cohort` - Simulate 9-covariate CVD cohort with interactions
- `code:clinical-data` - Organize data frame
- `code:clinical-library` - 70/30 split + 18-learner library (9 base + RF/XGB grids)
- `code:clinical-cv` - CV.SuperLearner with method.NNloglik, V=10, stratified
- `code:clinical-cv-summary` - Summarize CV performance
- `code:clinical-weights` - Ensemble weights across folds (reuses review_weights())
- `code:clinical-evaluation` - Final SL fit + test AUC with 95% CI
- `code:clinical-roc` - ROC on held-out test set
- `code:clinical-calibration` - Calibration plot (deciles + loess)

### 11. **Continuous Outcomes**
- `code:continuous-data` - Gaussian outcome with quadratic + interaction truth
- `code:continuous-sl` - SL with gaussian() (mean/glm/glmnet/RF/earth)
- `code:continuous-error` - MSE / RMSE

### 12. **Dose-Response Estimation**
- `code:dose-response` - Piecewise truth (sqrt < 2, quadratic ≥ 2) + Laplace noise
- `code:dose-sl-fit` - SL with method.NNLS (mean/glm/gam/earth)
- `code:dose-predict` - Predict on fine grid (onlySL=FALSE to keep learner curves)
- `code:dose-response-plot` - Truth vs SL vs glm vs gam

### 13. **Best Practices (templates)**
- `code:best-practice-tune` - RF + XGB grids for key algorithms
- `code:best-practice-screen` - Learner-screener pairs (unscreened/screen.corP/screen.corRank)
- `code:best-practice-parallel` - Two-line multicore setup with L'Ecuyer-CMRG

### 14. **Bridge to Causal Inference**
- `code:tmle-connection` - TMLE with SL for Q and g (supply own Y/A/W; `install.packages("tmle")`)

Plus in-article (no code box): SL vs caret comparison, SL package summary table, and SL quick-reference cheat sheet — see Overleaf PDF.

## Reproducibility

- **All random seeds are fixed** (`set.seed(123)` for SL, `set.seed(2743)` for small simulation, `set.seed(2024)` clinical, `set.seed(99)` continuous, `set.seed(12345)` dose-response)
- **Train/test splits are fixed** (80/20 small data via `train_idx`; 70/30 clinical via `idx`)
- **Cross-validation folds are controlled** via `V` parameter and `cvControl`
- **Parallel RNG** uses `"L'Ecuyer-CMRG"` for reproducible parallel execution

To reproduce exact results from the article, run the script sequentially without modification.

## Key Concepts Demonstrated

1. **Ensemble Learning**: Combining multiple algorithms via cross-validated weights
2. **Meta-Learning**: NNLS (default), NNloglik, AUC optimization
3. **Cross-Validation**: Internal (weight estimation) vs External (CV.SuperLearner)
4. **Hyperparameter Tuning**: Multiple configurations as distinct candidates
5. **Feature Screening**: Pre-processing variables before learning
6. **Parallelization**: Multicore, mcSuperLearner, and cluster execution
7. **Extensibility**: Custom learners and screeners via wrapper interface
8. **Clinical Prediction**: Full 18-learner CVD model with calibration + decision context
9. **Continuous & Dose-Response**: Gaussian SL, non-linear exposure-response curves
10. **TMLE Bridge**: SL as nuisance estimator for targeted learning

## Citation

If you use this code, please cite the original article:

> Super Learner: A Practical Tutorial
> [Overleaf Project](https://www.overleaf.com/read/6abe3298dba216ce9b241bf0)

Key references:
- van der Laan, M. J., Polley, E. C., & Hubbard, A. E. (2007). Super Learner. *Statistical Applications in Genetics and Molecular Biology*, 6(1).
- Phillips, R. V., van der Laan, M. J., Lee, H., & Gruber, S. (2023). Practical considerations for specifying a super learner. *International Journal of Epidemiology*, 52(4), 1276-1285.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Original Article

The article was developed on Overleaf: https://git.overleaf.com/6abe3298dba216ce9b241bf0

This repository extracts and organizes all executable R code from the LaTeX source for standalone reproducibility.
Last sync: 2026-10-06 (Overleaf `ad15a36`, 58 boxes).
