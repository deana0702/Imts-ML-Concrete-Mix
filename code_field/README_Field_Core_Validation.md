# IMTS Field Core Validation — Scripts 10–14

These scripts validate the strongest models created in experiments 06–09.
They use hardcoded repository paths and require no command-line arguments.

## Folder placement

Copy all files into the existing `code_field` folder:

```text
imts-ML-Concrete-Mix/
├── code_field/
│   ├── field_core_experiment_common.py
│   ├── field_core_validation_common.py
│   ├── 10_grouped_cv_regression.py
│   ├── 11_grouped_cv_risk_classification.py
│   ├── 12_unseen_supplier_mix_validation.py
│   ├── 13_prepare_external_2022_2026.py
│   └── 14_validate_external_2022_2026.py
└── data/
    └── field_core_outputs/
        └── field_core_clean/
            └── field_core_clean_with_required.csv
```

The scripts can also locate the older path:

```text
data/field_core_clean/field_core_clean_with_required.csv
```

## 10 — Project-grouped regression cross-validation

```powershell
python code_field/10_grouped_cv_regression.py
```

Validates these regression models across five project-group folds:

- Day-0 Context: Field + Required + Supplier/Plant/Mix/Test Subtype
- Day-7 Full Updated: Day-0 Context + 7-day strength
- Random Forest
- HistGradientBoosting

Important output:

```text
data/field_core_outputs/validation/10_grouped_cv_regression/
├── grouped_cv_regression_summary.csv
├── grouped_cv_regression_fold_metrics.csv
├── grouped_cv_regression_predictions.csv
└── grouped_cv_context_metadata.csv
```

## 11 — Project-grouped risk-classification cross-validation

```powershell
python code_field/11_grouped_cv_risk_classification.py
```

Validates:

- Day-0 Context risk
- Day-7 Full Updated risk
- Logistic Regression
- HistGradientBoosting

The reported threshold is fixed at 0.50 only for comparison. Do not select the final operational threshold from these validation folds.

Important output:

```text
data/field_core_outputs/validation/11_grouped_cv_risk/
├── grouped_cv_risk_summary.csv
├── grouped_cv_risk_fold_metrics.csv
├── grouped_cv_risk_predictions.csv
└── grouped_cv_risk_context_metadata.csv
```

## 12 — Unseen supplier and unseen supplier/plant/mix validation

```powershell
python code_field/12_unseen_supplier_mix_validation.py
```

Runs two holdout scenarios:

1. `UnseenSupplier`: the suppliers in test are not present in training.
2. `UnseenSupplierPlantMix`: complete Supplier + Plant + Mix combinations in test are not present in training.

It evaluates both strength regression and failure-risk classification for Day-0 and Day-7.

Important output:

```text
data/field_core_outputs/validation/12_unseen_supplier_mix/
├── unseen_context_regression_metrics.csv
├── unseen_context_risk_metrics.csv
├── unseen_context_split_summary.csv
├── unseen_context_split_assignments.csv
├── unseen_context_encoding_metadata.csv
└── unseen_context_predictions.csv
```

## 13 — Prepare current 2022–2026 production data

Export the **first result set** from the same final SQL query against the current production database and save it as:

```text
data/external_2022_2026/field_core_US_data_current.csv
```

Then run:

```powershell
python code_field/13_prepare_external_2022_2026.py
```

The script:

- keeps US-unit rows (`ConcreteTestUnitSystem = 0`);
- keeps cast years 2022 through 2026;
- requires valid actual and required 28-day strengths;
- excludes duplicate test IDs;
- changes impossible feature values to missing;
- preserves missing values for model imputation.

Prepared file:

```text
data/external_2022_2026/prepared/field_core_external_2022_2026_clean.csv
```

## 14 — True time-based external validation

```powershell
python code_field/14_validate_external_2022_2026.py
```

Training data:

```text
Old test database rows through 2021
```

External test data:

```text
Current production database rows from 2022 through 2026
```

No external rows are used to train the model or create the context mappings.
The script checks for overlapping `testId` values and stops if overlap exists.

Important output:

```text
data/field_core_outputs/validation/14_external_2022_2026/
├── external_regression_metrics.csv
├── external_regression_metrics_by_year.csv
├── external_risk_metrics_at_050.csv
├── external_risk_metrics_by_year.csv
├── external_regression_predictions.csv
├── external_risk_predictions.csv
└── external_context_encoding_metadata.csv
```

## Recommended execution order

```powershell
python code_field/10_grouped_cv_regression.py
python code_field/11_grouped_cv_risk_classification.py
python code_field/12_unseen_supplier_mix_validation.py
python code_field/13_prepare_external_2022_2026.py
python code_field/14_validate_external_2022_2026.py
```

Run 13 and 14 only after the current production SQL export is available.
