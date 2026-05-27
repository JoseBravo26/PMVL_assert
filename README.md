[Readme.txt](https://github.com/user-attachments/files/28322246/Readme.txt)
# PMVL Estimation Accuracy Classifier
### CatBoost · Asymmetric Cost · Time-Series CV · MLflow · Optuna

A production-grade binary classification model that predicts whether a **PMVL (Plus/Minus
Valeur Latente) estimate at time *t*** will be sufficiently close to the **realized PMVL
at time *t+1***.

The model is purpose-built for financial position-level data:
it handles temporal ordering, class imbalance, categorical features (fund codes, ISINs,
entities), and optimizes a **business cost function** that penalizes errors proportionally
to the size of the PMVL position — not just raw accuracy.

---

## Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [Data Requirements](#2-data-requirements)
3. [Data Cleaning Notes](#3-data-cleaning-notes)
4. [Feature Engineering](#4-feature-engineering)
5. [Cost Function Design](#5-cost-function-design)
6. [Model Architecture](#6-model-architecture)
7. [Validation Strategy](#7-validation-strategy)
8. [Hyperparameter Tuning](#8-hyperparameter-tuning)
9. [Best Hyperparameters](#9-best-hyperparameters)
10. [Results](#10-results)
11. [MLflow Tracking](#11-mlflow-tracking)
12. [Usage Guide](#12-usage-guide)
13. [Project Structure](#13-project-structure)
14. [Key Design Decisions](#14-key-design-decisions)
15. [Limitations & Next Steps](#15-limitations--next-steps)

---

## 1. Problem Definition

For each portfolio position on a given holding date *t*, we define the target variable as:

```
error_rel = |PRMP_PMVL_future - PMVL_Estimé| / |PMVL_Estimé|

target = True   if error_rel <= 0.05   (estimate is accurate)
target = False  otherwise              (estimate deviates > 5%)
```

Where:
- `PMVL[PMVL Estimé]`  — PMVL estimate produced at time *t*
- `PRMP_PMVL_future`   — actual realized PMVL at time *t+1*
  (computed by shifting `PMVL[PRMP PMVL]` by -1 within each position group)

The threshold of 5% relative error is a business parameter and can be tuned.

### Class Distribution (after cleaning)

| Class | Count | Share |
|-------|-------|-------|
| `True`  — good estimate  | 7,667 | 74.0% |
| `False` — bad estimate   | 2,471 | 26.0% |

The dataset is **moderately imbalanced** (≈ 3:1 ratio). This is handled via
inverse-frequency class weights in CatBoost.

---

## 2. Data Requirements

### Input file

```
pmvl_cleaned_prepared_with_features.csv
```

### Required columns

| Column | Type | Description |
|--------|------|-------------|
| `PMVL[Holding date]`       | datetime | Trading/holding date — defines temporal order |
| `PMVL[PMVL Estimé]`        | float    | PMVL estimate at *t* |
| `PRMP_PMVL_future`          | float    | Realized PMVL at *t+1* (target numerator) |
| `PMVL[PRMP PMVL]`           | float    | Realized PMVL at *t* (used for group shifting) |
| `target`                    | bool/int | Binary label (True/1 = good estimate) |

### Optional grouping keys (used to define position identity)

| Column | Description |
|--------|-------------|
| `PMVL[ENTITE]`             | Legal entity |
| `PMVL[Selected Fund code]` | Fund code |
| `PMVL[ISIN]`               | ISIN of the instrument |
| `PMVL[Ref Unik Asset]`     | Unique asset reference |

All remaining columns are treated automatically as features (numerical or categorical).

### Dataset dimensions

| Dimension | Value |
|-----------|-------|
| Total rows          | 10,138 |
| Features            | 27 |
| Categorical features| 15 |
| Unique position groups (train) | 19 |
| Date range          | 2026-01-05 → 2026-03-25 |

---

## 3. Data Cleaning Notes

### Anomalous data in January–February 2026

During the initial data collection period (January and February 2026), **certain
trading days had their PMVL records triplicated** due to a data ingestion bug.
This caused artificially inflated PMVL values that contaminated:

- The p95 error cap (making it too permissive)
- The CV fold cost distributions (folds 2–4 collapsed the threshold)
- The `FN_WEIGHT` asymmetry ratio

**Fix applied:** rows from the affected days were removed before any feature
engineering or model training.

### Capped absolute error

To prevent residual outliers from dominating the cost function, the absolute
PMVL error is capped at the **95th percentile** of the training distribution:

```
error_abs_pmvl_capped = clip(|PRMP_PMVL_future - PMVL_Estimé|, upper=p95)
```

| Statistic | Value |
|-----------|-------|
| p95 cap value | 2,062,091 |
| Median (capped, train) | 9,420 |
| Mean (capped, train)   | 227,230 |

---

## 4. Feature Engineering

The feature matrix `X` is built by dropping only the columns that would cause
**temporal leakage** or are structural metadata:

| Dropped column | Reason |
|----------------|--------|
| `target`               | Label — not a feature |
| `PMVL[Holding date]`   | Used only for sorting and splitting |
| `PRMP_PMVL_future`     | Future information — direct leakage |
| `error_abs_pmvl`       | Derived from future — direct leakage |
| `error_abs_pmvl_capped`| Derived from future — used only in cost |

All 15 categorical columns (fund codes, ISINs, entities, etc.) are handled natively
by CatBoost without any manual encoding. Missing values in categoricals are filled
with the string `"MISSING"`.

---

## 5. Cost Function Design

### Motivation

Standard metrics (accuracy, F1) treat all errors equally. In the PMVL context:

- A **False Positive** (model says "good estimate" but it was bad) means the desk
  *trusts* a wrong PMVL figure → potential mispricing proportional to position size.
- A **False Negative** (model says "bad estimate" but it was good) means the desk
  *discards* a correct figure → unnecessary recalculation cost.

Both errors scale with the magnitude of the PMVL position, so a flat per-error
penalty is not appropriate.

### Formula

```
cost(FP) = error_abs_pmvl_capped * FP_WEIGHT
cost(FN) = error_abs_pmvl_capped * FN_WEIGHT

total_cost = Σ cost(FP) + Σ cost(FN)
cost_per_sample = total_cost / n
```

### Asymmetry calibration

`FN_WEIGHT` is **computed from the data at training time**, not hardcoded:

```python
FN_WEIGHT = mean(error_abs_pmvl_capped | target=False)
          / mean(error_abs_pmvl_capped | target=True)
```

This ensures that FP and FN have **comparable expected impact**, compensating for
the structural asymmetry (bad-estimate positions have larger PMVL errors by definition).

| Parameter | Value |
|-----------|-------|
| `FP_WEIGHT`              | 1.0 (fixed) |
| `FN_WEIGHT`              | **9.35** (computed from data) |
| Mean error (False class) | 624,387 |
| Mean error (True class)  | 66,776 |
| Asymmetry ratio          | 9.35× |

### Threshold selection

For each CV fold, the optimal probability threshold is found by scanning a grid
(0.05 → 0.95, step 0.05) and selecting the threshold that minimizes `cost_per_sample`,
subject to the business constraint **recall ≥ 0.70** (at most 30% of good estimates
can be discarded as false negatives).

---

## 6. Model Architecture

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Algorithm | CatBoostClassifier | Native categorical support, robust on tabular data, no encoding needed |
| Objective | Logloss | Standard binary cross-entropy |
| Eval metric | AUC | Threshold-independent during training |
| Class weights | Inverse frequency | Compensates 74/26 imbalance |
| Early stopping | Iter (od_wait=80) | Prevents overfitting on small folds |
| HPO | Optuna (TPE sampler) | In-memory, 50 trials |
| Threshold | Cost-based (grid search) | Aligned with business objective |

---

## 7. Validation Strategy

### Train / test split

A strict **chronological split** is applied: the most recent 20% of rows by date
form the final hold-out test set. The remaining 80% is used for cross-validation
and final training.

```
Train : 8,110 rows  (2026-01-05 → ~2026-03-06)
Test  :  2,028 rows  (~2026-03-06 → 2026-03-25)
```

### Cross-validation

`SimpleTimeSeriesSplit` with **5 folds** uses a **growing window** approach:
each fold adds more past data to the train split and evaluates on the next
contiguous time window. There is **no shuffling** — temporal order is strictly
preserved.

```
Fold 1: train=[t0 … t1]      val=[t1 … t2]
Fold 2: train=[t0 … t2]      val=[t2 … t3]
Fold 3: train=[t0 … t3]      val=[t3 … t4]
Fold 4: train=[t0 … t4]      val=[t4 … t5]
Fold 5: train=[t0 … t5]      val=[t5 … t6]
```

### Threshold selection rule

1. Discard folds where recall < `MIN_RECALL` (= 0.70).
2. Among remaining folds, select the one with the lowest `cost_per_sample`.
3. Use that fold's threshold as the **global operating threshold**.

All 5 folds met the recall constraint → **fold 5 selected** (lowest cost, threshold = 0.45).

---

## 8. Hyperparameter Tuning

Hyperparameter search is performed with **Optuna** using the TPE sampler and Median
pruner (in-memory, no SQLite). The objective is to minimize the **mean cost_per_sample
across all CV folds**.

| Parameter | Search space |
|-----------|-------------|
| `iterations`       | int [400, 1200] |
| `learning_rate`    | log-uniform [0.01, 0.10] |
| `depth`            | int [4, 8] |
| `l2_leaf_reg`      | uniform [1.0, 10.0] |
| `subsample`        | uniform [0.6, 1.0] |
| `random_strength`  | uniform [0.5, 2.0] |
| `border_count`     | categorical {64, 128, 255} |
| `min_data_in_leaf` | int [10, 100] |

**Trials:** 50  
**Best cost_per_sample (CV):** 117,233

---

## 9. Best Hyperparameters

| Parameter | Value |
|-----------|-------|
| `iterations`       | 456 |
| `learning_rate`    | 0.05505 |
| `depth`            | 6 |
| `l2_leaf_reg`      | 6.114 |
| `subsample`        | 0.699 |
| `random_strength`  | 0.806 |
| `border_count`     | 128 |
| `min_data_in_leaf` | 34 |
| `class_weight_0`   | 1.883 |
| `class_weight_1`   | 0.681 |

---

## 10. Results

### Cross-Validation (5 folds)

| Fold | Val period | Threshold | Cost/sample | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|------|-----------|-----------|-------------|-----------|--------|----|---------|--------|
| 1 | 2026-01-15 → 2026-01-23 | 0.65 | 214,812 | 0.740 | 0.908 | 0.815 | 0.804 | 0.778 |
| 2 | 2026-01-23 → 2026-02-03 | 0.25 | 153,875 | 0.775 | 0.987 | 0.868 | 0.839 | 0.873 |
| 3 | 2026-02-03 → 2026-02-17 | 0.20 |  88,743 | 0.876 | 0.970 | 0.921 | 0.881 | 0.964 |
| 4 | 2026-02-17 → 2026-03-03 | 0.20 |  79,667 | 0.888 | 0.949 | 0.918 | 0.867 | 0.952 |
| **5** | **2026-03-03 → 2026-03-11** | **0.45** | **49,066** | **0.914** | **0.929** | **0.921** | **0.874** | **0.963** |

**→ Selected fold: 5 · Global threshold: 0.45**

### CV Summary

| Metric | Value |
|--------|-------|
| Mean cost/sample   | 117,233 |
| Std cost/sample    | 66,584 |
| Mean F1            | 0.889 |
| Mean Recall        | 0.949 |
| Mean Precision     | 0.838 |
| Mean Balanced Acc  | 0.748 |
| Mean ROC-AUC       | 0.853 |
| Mean PR-AUC        | 0.906 |

### Test Set (hold-out — 2,028 rows)

| Metric | Value |
|--------|-------|
| Threshold           | 0.45 |
| Accuracy            | 0.854 |
| Balanced accuracy   | 0.730 |
| Precision           | 0.916 |
| Recall              | 0.910 |
| F1                  | 0.913 |
| MCC                 | 0.454 |
| ROC-AUC             | 0.836 |
| PR-AUC              | **0.955** |
| Brier score         | 0.168 |

### Cost breakdown (test)

| Component | Value | Share |
|-----------|-------|-------|
| FP cost total  | 59,736,304  | 38% |
| FN cost total  | 97,730,434  | 62% |
| **Total cost** | **157,466,738** | 100% |
| Cost/sample    | 77,646 | — |
| n_FP / n_FN    | 143 / 154 | — |
| Avg cost FP    | 417,736 | per error |
| Avg cost FN    | 67,870  | per error |

> The FN cost dominates (62%) because FN_WEIGHT=9.35 amplifies even low-error FNs.
> Individual FPs are more expensive on average (417k vs 68k) but there are fewer of them.

---

## 11. MLflow Tracking

### Experiment names

| Pipeline version | MLflow experiment |
|-----------------|-------------------|
| Full (Optuna + CV) | `pmvl_catboost_v3_asym_cost` |
| Fixed params (no Optuna) | `pmvl_catboost_v3_asym_cost_fixed` |

### Run structure

```
main_run  (catboost_v3_asym_cost_main)
├── fold_1  (nested)
├── fold_2  (nested)
├── fold_3  (nested)
├── fold_4  (nested)
├── fold_5  (nested)
└── final_model  (nested)
```

### What is logged

**Parameters (main run):**
- Data stats: n_rows, n_features, n_cat_features, class weights, date range
- Cost config: fp_weight, fn_weight_data, pmvl_cost_col, min_recall_constraint
- All Optuna best_params prefixed with `best_`
- optuna_best_cost_per_sample

**Metrics (per fold):**
- `val_threshold`, `val_cost_per_sample`, `val_total_cost`
- `val_fp_cost_total`, `val_fn_cost_total`, `val_n_fp`, `val_n_fn`
- `val_precision`, `val_recall`, `val_f1`, `val_balanced_accuracy`
- `val_roc_auc`, `val_pr_auc`, `val_brier`

**Metrics (main run — CV summary):**
- `cv_mean_*`, `cv_std_cost_per_sample`, `cv_selected_fold`, `cv_selected_threshold`

**Metrics (final model):**
- `test_*` — same set of metrics evaluated on the hold-out test set

**Artifacts (per fold):**
- `confusion_matrix.png`
- `proba_distribution.png`
- `threshold_curves.png` (cost + metrics vs threshold)
- `cost_breakdown.png` (FP vs FN stacked area)
- `feature_importance.csv` + `.png` (top 30)
- `threshold_grid.csv`

**Artifacts (final model):**
- `confusion_matrix_test.png`
- `test_threshold_sensitivity.csv` (all thresholds evaluated on test)
- `test_predictions_error_analysis.csv` (row-level: y_true, y_proba, error_type, business_cost)
- `feature_importance_final.csv` + `.png`
- `final_summary.json`
- Logged CatBoost model (reloadable via `mlflow.catboost.load_model`)

---

## 12. Usage Guide

### Prerequisites

```bash
pip install catboost scikit-learn pandas numpy mlflow optuna matplotlib seaborn
```

### Option A — Final model only (fixed hyperparameters, fastest)

Use this when you already have the best hyperparameters and just want to retrain
or score.

```python
import pandas as pd

df = pd.read_csv("pmvl_cleaned_prepared_with_features.csv")

# Train
final_model, global_threshold, train_metrics, test_metrics = train_final_model(df)

# Score new positions
# X_new must have the same columns as the training feature matrix (no target, no date, no future PMVL)
proba = final_model.predict_proba(X_new)[:, 1]
y_hat = (proba >= global_threshold).astype(int)
# y_hat = 1 → estimate expected to be accurate (True class)
# y_hat = 0 → estimate expected to be inaccurate (False class)
```

### Option B — Full pipeline with Optuna + MLflow (recommended for new data)

Use this when retraining from scratch or when the data distribution has changed.

```python
final_model, global_threshold, cv_results_df, test_metrics = run_pmvl_pipeline(df)
```

This will:
1. Prepare features and compute the asymmetry-corrected FN_WEIGHT from the data.
2. Run 50 Optuna trials to find the best hyperparameters.
3. Run 5-fold time-series CV to find the optimal threshold.
4. Train the final model on the full train set.
5. Log everything to MLflow.

### Adjustable parameters

| Parameter | Location | Default | When to change |
|-----------|----------|---------|----------------|
| `TEST_SIZE_RATIO` | CONFIG | 0.20 | More/less data for final evaluation |
| `N_SPLITS_CV`     | CONFIG | 5    | More folds if data grows |
| `MIN_RECALL`      | CONFIG | 0.70 | Raise to 0.90+ if FN is business-critical |
| `N_TRIALS`        | CONFIG | 50   | Increase for more thorough HPO |
| `GLOBAL_THRESHOLD`| CONFIG | 0.45 | Override if cost analysis recommends another |
| `FP_WEIGHT`       | CONFIG | 1.0  | Adjust if FP business cost is re-quantified |
| Target threshold  | `prepare_target()` | 0.05 | 5% relative error — tune per business rule |

---

## 13. Project Structure

```
pmvl_cleaned_prepared_with_features.csv     ← cleaned input data
pmvl_pipeline_v3.py                         ← full pipeline: Optuna + CV + MLflow
pmvl_final_model.py                         ← lightweight fixed-params script
README.md                                   ← this file

mlflow_artifacts_pmvl_v3_asym_cost/         ← local artifact store
  feature_schema.json
  train_class_balance.png
  test_class_balance.png
  error_distribution_by_class.png
  cv_fold_metrics.csv
  cv_threshold_grid_all_folds.csv
  fold_1/
    confusion_matrix.png
    proba_distribution.png
    threshold_curves.png
    cost_breakdown.png
    feature_importance.csv
    feature_importance.png
    threshold_grid.csv
  … fold_2 through fold_5 …
  final_model/
    confusion_matrix_test.png
    proba_distribution_test.png
    test_threshold_curves.png
    test_cost_breakdown.png
    test_threshold_sensitivity.csv
    test_predictions_error_analysis.csv
    feature_importance_final.csv
    feature_importance_final.png
    final_summary.json
```

---

## 14. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Asymmetric cost (FN_WEIGHT ≈ 9.35)** | Mean error of False class is 9.35× larger than True class; symmetric cost collapses threshold to 0.90+ and destroys recall |
| **FN_WEIGHT computed from data** | The ratio changes if the data changes (new cleaning, new period); hardcoding would silently mis-calibrate the cost |
| **p95 error cap** | Prevents any single extreme position from dominating the threshold search |
| **TimeSeriesSplit (no shuffle)** | Financial time series have temporal autocorrelation; shuffling would cause data leakage |
| **MIN_RECALL = 0.70** | Business constraint: the model cannot discard more than 30% of accurate estimates |
| **Threshold from fold 5** | Most recent fold has the lowest cost/sample and best generalization to unseen future data |
| **CatBoost with native categoricals** | 15 of 27 features are categorical; native handling avoids target-encoding leakage |
| **Class weights (inverse frequency)** | 74/26 split; without weighting the model would be biased toward the majority class |

---

## 15. Limitations & Next Steps

### Current limitations

- **Short time range:** data covers only Jan–Mar 2026 (≈3 months). CV folds are
  small (≈200–1,350 rows) which makes threshold estimates noisy in early folds.
- **FN dominates test cost (62%):** threshold 0.45 may be slightly too high for the
  test period. A lower threshold (0.35–0.40) could reduce FN cost if the recall
  constraint allows it.
- **No probability calibration:** CatBoost probabilities are not explicitly calibrated.
  Platt scaling or isotonic regression could improve threshold reliability.
- **Static FN_WEIGHT:** recomputed only at training time. If the position size
  distribution shifts significantly, FN_WEIGHT should be updated.

### Recommended next steps

1. **Extend dataset** — add more historical months to stabilize CV folds.
2. **Production scoring script** — load the final model from MLflow and score
   daily PMVL snapshots automatically.
3. **Threshold sensitivity analysis** — run `test_threshold_sensitivity.csv` through
   a business review to decide if 0.45 is the right operating point.
4. **Monitor drift** — track `cost_per_sample` and `recall` on each new month's
   data to detect model degradation.
5. **Recalibrate FN_WEIGHT** — when 6+ months of data are available, recompute
   the asymmetry ratio on a larger and more stable sample.
