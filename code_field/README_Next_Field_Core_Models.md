# IMTS Field Core — Next Model Experiments

Place these files in the repository root, next to the existing Step 1–5 scripts.
No command-line arguments are used.

Expected input files:

```text
data/field_core_outputs/field_core_splits/comparison_train.csv
data/field_core_outputs/field_core_splits/comparison_test.csv
```

## Files

### `field_core_experiment_common.py`
Shared paths, features, models, metrics, 7-day feature preparation, and leakage-controlled context encoding. Do not run this file directly.

### `06_compare_required_only.py`
Compares:

- Required Strength Only
- Field Measurements + Required Strength

This determines how much additional value the field measurements provide beyond required strength alone.

Run:

```bash
python 06_compare_required_only.py
```

### `07_compare_day0_context.py`
Compares:

- Field + Required
- Field + Required + Supplier/Plant/Mix/Test Subtype context

Context is normalized and converted to numeric features with project-grouped out-of-fold target encoding. Test mappings are learned from training data only.

Run:

```bash
python 07_compare_day0_context.py
```

### `08_compare_day7_updated.py`
Uses only rows with a valid 7-day standard-cured strength and compares, on the same rows:

- Field + Required
- Field + Required + Context
- Field + Required + 7-Day
- Full Updated: Field + Required + Context + 7-Day

Run:

```bash
python 08_compare_day7_updated.py
```

### `09_early_risk_classification.py`
Creates the analytical failure target:

```text
FailureFlag28 = Actual 28-Day Strength < Applicable 28-Day Required/Design Strength
```

Compares Day-0 and Day-7 screening classifiers. Main metrics are PR-AUC, failure recall, precision, false negatives, ROC-AUC, and Brier score. Accuracy is intentionally not the primary metric because failures are uncommon.

Run:

```bash
python 09_early_risk_classification.py
```

## Execution order

```bash
python 06_compare_required_only.py
python 07_compare_day0_context.py
python 08_compare_day7_updated.py
python 09_early_risk_classification.py
```

## Context fields

The scripts resolve these alternatives automatically:

- Supplier: `SupplierId`, `supplierId`, or `SupplierName`
- Plant: `PlantNumber` or `plantNumber`
- Mix: `MixNumber` or `mixNumber`
- Test subtype: `testSubTypeId`, `TestSubTypeId`, or `testSubtypeId`

Supplier ID is preferred. Supplier name is used only when an ID is unavailable.

## Important interpretation

- Day-0 models can be used at or near placement time.
- Day-7 models are separate updated models and must not be described as placement-time predictions.
- The failure target is an analytical screening proxy, not the complete engineering acceptance rule.
- The risk classifier is decision support only and must not automatically accept or reject concrete.
