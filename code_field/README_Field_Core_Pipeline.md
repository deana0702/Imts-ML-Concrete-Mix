# IMTS Field Core Modeling Pipeline

Place all five Python files in the root of the `imts-ML-Concrete-Mix` project. No command-line arguments are used.

## Folder flow

```text
project root/
├── 01_clean_field_core_dataset.py
├── 02_create_grouped_splits.py
├── 03_train_holdout_models.py
├── 04_run_grouped_cross_validation.py
├── 05_generate_model_report.py
└── data/
    ├── prepared_field_core_US/
    │   └── field_core_model_candidates.csv
    ├── field_core_clean/
    ├── field_core_splits/
    ├── field_core_models/
    ├── field_core_cv/
    └── field_core_report/
```

## Install packages

```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

## Run in order

```bash
python 01_clean_field_core_dataset.py
python 02_create_grouped_splits.py
python 03_train_holdout_models.py
python 04_run_grouped_cross_validation.py
python 05_generate_model_report.py
```

## Why the code is separated

1. **Cleaning** changes data values and creates a reusable model-ready dataset.
2. **Splitting** creates one fixed project-grouped train/test split so every model is compared fairly.
3. **Holdout training** compares Dummy, Ridge, Random Forest, and HistGradientBoosting on the same unseen projects.
4. **Grouped cross-validation** repeats project-based validation to measure model stability.
5. **Reporting** combines the results and creates charts and a Markdown report.

## Step 1 outputs

- `field_core_clean_base.csv`
- `field_core_clean_with_required.csv`
- `cleaning_exclusion_audit.csv`
- `cleaning_review_rows.csv`
- `cleaning_feature_adjustment_summary.csv`

Step 1 replaces clearly impossible feature values with missing values rather than deleting the entire test. Invalid targets, duplicated tests, invalid dates, and non-US unit rows are excluded.

## Step 2 outputs

- `comparison_train.csv`
- `comparison_test.csv`
- `field_only_full_train.csv`
- `field_only_full_test.csv`
- `split_assignments.csv`

The common comparison files contain required strength and are used to compare Field Only with Field + Required Strength on exactly the same tests.

## Step 3 outputs

- `holdout_model_metrics.csv`
- `holdout_predictions.csv`
- `best_holdout_model.joblib`
- one saved `.joblib` file for every model and feature set

## Step 4 outputs

- `grouped_cv_fold_metrics.csv`
- `grouped_cv_summary.csv`

This step can take longer because it trains each selected model five times for each feature set.

## Step 5 outputs

Open this file first:

```text
data/field_core_report/field_core_model_report.md
```

It summarizes the best holdout result, grouped cross-validation, and the value added by required strength.
