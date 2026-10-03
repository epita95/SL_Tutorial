# Super Learner: A Practical Tutorial - Reproducible Code

This repository contains the complete, reproducible R code for the article **"Super Learner: A Practical Tutorial"** originally hosted on Overleaf.

## Overview

The tutorial provides a comprehensive, practical guide to implementing **Super Learner** in R for applied researchers in epidemiology and biostatistics. Super Learner is an ensemble machine-learning method that uses cross-validation to combine multiple candidate algorithms into a single prediction model, ensuring performance asymptotically at least as good as the best candidate algorithm.

## Repository Contents

| File | Description |
|------|-------------|
| `superlearner_tutorial.R` | **Complete reproducible R script** - all code boxes from the article, organized by section with original box labels |
| `LICENSE` | MIT License |

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
  "RhpcBLASctl", "rmutil", "vcd", "nnls", "Matrix"
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

### 1. **Setup & Data Generation**
- `code:install` - Package installation
- `code:load-sl` - Load SuperLearner
- `code:sim-covariates` - Generate simulated covariates (W1-W5)
- `code:sim-outcome` - Generate binary outcome with interactions/non-linearity
- `code:descriptive` - Descriptive summary

### 2. **Basic Super Learner**
- `code:sl-single` - Single algorithm (SL.glmnet)
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
- `code:cvsl-auc` - AUC optimization (method.AUC)
- `code:cvsl-nnloglik` - Negative log-likelihood (method.NNloglik)

### 8. **Parallelization**
- `code:parallel-setup` - Configure multicore execution
- `code:cvsl-parallel` - Parallel CV.SuperLearner
- `code:mcsuperlearner` - mcSuperLearner for prediction
- `code:cluster-parallel` - Cluster-based parallelization (commented)

### 9. **Custom Wrappers**
- `code:gbm-wrapper` - Custom GBM wrapper (SL.gbm2)
- `code:earth-wrapper` - Custom MARS wrapper (SL.earth2)
- `code:custom-screener` - Custom LASSO + Cramér's V screener
- `code:cvsl-custom` - Evaluate custom wrappers

### 10. **Theoretical Foundation**
- `code:weighted-procedure` - NNLS meta-learner implementation (illustrative)

## Reproducibility

- **All random seeds are fixed** (`set.seed(123)` for SL, `set.seed(2743)` for data generation)
- **Train/test split is fixed** (80/20, indices stored in `train_idx`)
- **Cross-validation folds are controlled** via `V` parameter and `cvControl`
- **Parallel RNG** uses `"L'Ecuyer-CMRG"` for reproducible parallel execution

To reproduce exact results from the article, run the script sequentially without modification.

## Key Concepts Demonstrated

1. **Ensemble Learning**: Combining multiple algorithms via cross-validated weights
2. **Meta-Learning**: NNLS (default), NNloglik, AUC optimization
3. **Cross-Validation**: Internal (weight estimation) vs External (CV.SuperLearner)
4. **Hyperparameter Tuning**: Multiple configurations as distinct candidates
5. **Feature Screening**: Pre-processing variables before learning
6. **Parallelization**: Multicore and cluster execution
7. **Extensibility**: Custom learners and screeners via wrapper interface

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

This repository extracts and organizes all executable R code from the LaTeX/knitr source for standalone reproducibility.