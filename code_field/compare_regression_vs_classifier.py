from __future__ import annotations

"""
IMTS Field Core
Compare Regression-Derived Pass/Fail vs Direct Classifier
using EXISTING 5-fold OOF prediction files.

This script does NOT retrain the models.

It uses:
1) Regression CV predictions:
   data/field_core_outputs/consolidated_three_model_cross_validation/cv_predictions.csv

2) Classification CV predictions:
   data/field_core_outputs/consolidated_risk_classification_cv/classification_oof_predictions.csv

3) Cleaned source data, only to retrieve the required 28-day strength:
   data/field_core_outputs/field_core_clean/field_core_clean_with_required.csv

Regression-derived failure:
    PredictedStrength28_psi < ApplicableSpecifiedStrength28

Direct classifier failure:
    FailureProbability >= 0.50

Run:
    python code_field/compare_regression_vs_classifier.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent


def find_repo_root() -> Path:
    for candidate in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        if (candidate / "data").exists():
            return candidate
    return SCRIPT_DIR.parent


REPO_ROOT = find_repo_root()
DATA_ROOT = REPO_ROOT / "data" / "field_core_outputs"

REGRESSION_PREDICTIONS = (
    DATA_ROOT
    / "consolidated_three_model_cross_validation"
    / "cv_predictions.csv"
)

CLASSIFICATION_PREDICTIONS = (
    DATA_ROOT
    / "consolidated_risk_classification_cv"
    / "classification_oof_predictions.csv"
)

CLEAN_DATA = (
    DATA_ROOT
    / "field_core_clean"
    / "field_core_clean_with_required.csv"
)

OUTPUT_DIR = (
    DATA_ROOT
    / "regression_vs_classifier_comparison"
)

REQUIRED_STRENGTH = "ApplicableSpecifiedStrength28"

CLASSIFIER_THRESHOLD = 0.50


# Use the best regression model from your latest regression CV.
BEST_REGRESSION_MODEL = {
    "Day0_FieldPlusRequired": "HistGradientBoosting",
    "Day7_FieldPlusRequired": "XGBoost",
    "Full_ContextPlusDay7": "HistGradientBoosting",
}

# Use the best PR-AUC classifier from your latest classification CV.
BEST_CLASSIFIER_MODEL = {
    "Day0_FieldPlusRequired": "HistGradientBoosting",
    "Day7_FieldPlusRequired": "HistGradientBoosting",
    "Full_ContextPlusDay7": "HistGradientBoosting",
}


# =============================================================================
# HELPERS
# =============================================================================

def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found:\n{path}")

    return pd.read_csv(path, low_memory=False)


def classification_metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict[str, float | int]:
    actual_array = actual.astype(int).to_numpy()
    predicted_array = predicted.astype(int).to_numpy()

    tn, fp, fn, tp = confusion_matrix(
        actual_array,
        predicted_array,
        labels=[0, 1],
    ).ravel()

    return {
        "Recall": float(
            recall_score(
                actual_array,
                predicted_array,
                zero_division=0,
            )
        ),
        "Precision": float(
            precision_score(
                actual_array,
                predicted_array,
                zero_division=0,
            )
        ),
        "F1": float(
            f1_score(
                actual_array,
                predicted_array,
                zero_division=0,
            )
        ),
        "TrueNegatives": int(tn),
        "FalsePositives": int(fp),
        "FalseNegatives": int(fn),
        "TruePositives": int(tp),
    }


def filter_best_model(
    df: pd.DataFrame,
    model_lookup: dict[str, str],
) -> pd.DataFrame:
    frames = []

    for feature_set, model_name in model_lookup.items():
        subset = df.loc[
            (df["FeatureSet"] == feature_set)
            & (df["Model"] == model_name)
        ].copy()

        if subset.empty:
            raise ValueError(
                f"No rows found for "
                f"FeatureSet={feature_set}, Model={model_name}"
            )

        frames.append(subset)

    return pd.concat(
        frames,
        ignore_index=True,
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Reading existing OOF prediction files...")

    reg = read_csv(
        REGRESSION_PREDICTIONS
    )

    clf = read_csv(
        CLASSIFICATION_PREDICTIONS
    )

    clean = read_csv(
        CLEAN_DATA
    )

    required_reg_columns = {
        "Fold",
        "FeatureSet",
        "Model",
        "ActualStrength28_psi",
        "PredictedStrength28_psi",
        "testId",
    }

    required_clf_columns = {
        "Fold",
        "FeatureSet",
        "Model",
        "ActualFailureFlag",
        "FailureProbability",
        "testId",
    }

    missing_reg = (
        required_reg_columns
        - set(reg.columns)
    )

    missing_clf = (
        required_clf_columns
        - set(clf.columns)
    )

    if missing_reg:
        raise KeyError(
            f"Regression prediction file is missing: "
            f"{sorted(missing_reg)}"
        )

    if missing_clf:
        raise KeyError(
            f"Classification prediction file is missing: "
            f"{sorted(missing_clf)}"
        )

    if "testId" not in clean.columns:
        raise KeyError(
            "Clean data must contain testId."
        )

    if REQUIRED_STRENGTH not in clean.columns:
        raise KeyError(
            f"Clean data must contain {REQUIRED_STRENGTH}."
        )

    # -------------------------------------------------------------------------
    # Keep only the selected best model for each feature set.
    # -------------------------------------------------------------------------
    reg = filter_best_model(
        reg,
        BEST_REGRESSION_MODEL,
    )

    clf = filter_best_model(
        clf,
        BEST_CLASSIFIER_MODEL,
    )

    # -------------------------------------------------------------------------
    # Required strength lookup.
    # -------------------------------------------------------------------------
    required_lookup = (
        clean[
            [
                "testId",
                REQUIRED_STRENGTH,
            ]
        ]
        .drop_duplicates(
            subset=["testId"]
        )
        .copy()
    )

    required_lookup[
        REQUIRED_STRENGTH
    ] = pd.to_numeric(
        required_lookup[
            REQUIRED_STRENGTH
        ],
        errors="coerce",
    )

    reg = reg.merge(
        required_lookup,
        on="testId",
        how="left",
        validate="many_to_one",
    )

    if reg[
        REQUIRED_STRENGTH
    ].isna().any():
        missing_count = int(
            reg[
                REQUIRED_STRENGTH
            ].isna().sum()
        )

        raise ValueError(
            f"{missing_count:,} regression OOF rows "
            f"could not be matched to required strength."
        )

    # -------------------------------------------------------------------------
    # Regression-derived PASS / FAIL.
    # -------------------------------------------------------------------------
    reg[
        "RegressionPredictedMargin_psi"
    ] = (
        reg[
            "PredictedStrength28_psi"
        ]
        - reg[
            REQUIRED_STRENGTH
        ]
    )

    reg[
        "RegressionDerivedFailure"
    ] = (
        reg[
            "PredictedStrength28_psi"
        ]
        < reg[
            REQUIRED_STRENGTH
        ]
    ).astype(int)

    # -------------------------------------------------------------------------
    # Direct classifier PASS / FAIL at 0.50.
    # -------------------------------------------------------------------------
    clf[
        "ClassifierFailure"
    ] = (
        clf[
            "FailureProbability"
        ]
        >= CLASSIFIER_THRESHOLD
    ).astype(int)

    # -------------------------------------------------------------------------
    # Merge identical OOF cases.
    # Fold + FeatureSet + testId guarantees the same validation observation.
    # -------------------------------------------------------------------------
    comparison = reg.merge(
        clf[
            [
                "Fold",
                "FeatureSet",
                "testId",
                "ActualFailureFlag",
                "FailureProbability",
                "ClassifierFailure",
            ]
        ],
        on=[
            "Fold",
            "FeatureSet",
            "testId",
        ],
        how="inner",
        validate="one_to_one",
    )

    if comparison.empty:
        raise ValueError(
            "Regression and classification OOF files "
            "did not match on Fold + FeatureSet + testId."
        )

    # -------------------------------------------------------------------------
    # Verify that the analytical failure flag matches the actual strength rule.
    # -------------------------------------------------------------------------
    comparison[
        "ActualFailureFromStrength"
    ] = (
        comparison[
            "ActualStrength28_psi"
        ]
        < comparison[
            REQUIRED_STRENGTH
        ]
    ).astype(int)

    target_mismatch = (
        comparison[
            "ActualFailureFlag"
        ].astype(int)
        != comparison[
            "ActualFailureFromStrength"
        ]
    )

    mismatch_count = int(
        target_mismatch.sum()
    )

    if mismatch_count:
        print(
            f"WARNING: {mismatch_count:,} rows have "
            "a mismatch between ActualFailureFlag "
            "and ActualStrength < RequiredStrength."
        )

    # -------------------------------------------------------------------------
    # Compare the two methods.
    # -------------------------------------------------------------------------
    summary_rows = []
    agreement_rows = []

    for feature_set, group in comparison.groupby(
        "FeatureSet"
    ):
        actual = (
            group[
                "ActualFailureFlag"
            ].astype(int)
        )

        reg_metrics = classification_metrics(
            actual,
            group[
                "RegressionDerivedFailure"
            ],
        )

        clf_metrics = classification_metrics(
            actual,
            group[
                "ClassifierFailure"
            ],
        )

        summary_rows.append(
            {
                "FeatureSet": feature_set,

                "RegressionModel": (
                    BEST_REGRESSION_MODEL[
                        feature_set
                    ]
                ),
                "RegressionRecall": (
                    reg_metrics["Recall"]
                ),
                "RegressionPrecision": (
                    reg_metrics["Precision"]
                ),
                "RegressionF1": (
                    reg_metrics["F1"]
                ),
                "RegressionFalseNegatives": (
                    reg_metrics[
                        "FalseNegatives"
                    ]
                ),
                "RegressionFalsePositives": (
                    reg_metrics[
                        "FalsePositives"
                    ]
                ),

                "ClassifierModel": (
                    BEST_CLASSIFIER_MODEL[
                        feature_set
                    ]
                ),
                "ClassifierThreshold": (
                    CLASSIFIER_THRESHOLD
                ),
                "ClassifierRecall": (
                    clf_metrics["Recall"]
                ),
                "ClassifierPrecision": (
                    clf_metrics["Precision"]
                ),
                "ClassifierF1": (
                    clf_metrics["F1"]
                ),
                "ClassifierFalseNegatives": (
                    clf_metrics[
                        "FalseNegatives"
                    ]
                ),
                "ClassifierFalsePositives": (
                    clf_metrics[
                        "FalsePositives"
                    ]
                ),

                "RecallDifference_ClassifierMinusRegression": (
                    clf_metrics["Recall"]
                    - reg_metrics["Recall"]
                ),
                "PrecisionDifference_ClassifierMinusRegression": (
                    clf_metrics["Precision"]
                    - reg_metrics["Precision"]
                ),
                "FalseNegativesSavedByClassifier": (
                    reg_metrics[
                        "FalseNegatives"
                    ]
                    - clf_metrics[
                        "FalseNegatives"
                    ]
                ),
                "AdditionalFalsePositivesFromClassifier": (
                    clf_metrics[
                        "FalsePositives"
                    ]
                    - reg_metrics[
                        "FalsePositives"
                    ]
                ),
            }
        )

        reg_flag = (
            group[
                "RegressionDerivedFailure"
            ].astype(int)
        )

        clf_flag = (
            group[
                "ClassifierFailure"
            ].astype(int)
        )

        classifier_only = (
            (reg_flag == 0)
            & (clf_flag == 1)
        )

        regression_only = (
            (reg_flag == 1)
            & (clf_flag == 0)
        )

        agreement_rows.append(
            {
                "FeatureSet": feature_set,
                "Rows": len(group),
                "AgreementPercent": float(
                    (
                        reg_flag
                        == clf_flag
                    ).mean()
                    * 100.0
                ),
                "DisagreementPercent": float(
                    (
                        reg_flag
                        != clf_flag
                    ).mean()
                    * 100.0
                ),
                "ClassifierOnlyAlerts": int(
                    classifier_only.sum()
                ),
                "ClassifierOnlyActualFailures": int(
                    actual[
                        classifier_only
                    ].sum()
                ),
                "RegressionOnlyAlerts": int(
                    regression_only.sum()
                ),
                "RegressionOnlyActualFailures": int(
                    actual[
                        regression_only
                    ].sum()
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    agreement = pd.DataFrame(
        agreement_rows
    )

    # -------------------------------------------------------------------------
    # Save detailed row-level comparison.
    # -------------------------------------------------------------------------
    comparison.to_csv(
        OUTPUT_DIR
        / "regression_vs_classifier_oof_detail.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "regression_vs_classifier_summary.csv",
        index=False,
    )

    agreement.to_csv(
        OUTPUT_DIR
        / "regression_vs_classifier_agreement.csv",
        index=False,
    )

    run_info = {
        "regression_predictions": str(
            REGRESSION_PREDICTIONS
        ),
        "classification_predictions": str(
            CLASSIFICATION_PREDICTIONS
        ),
        "clean_data": str(
            CLEAN_DATA
        ),
        "classifier_threshold": (
            CLASSIFIER_THRESHOLD
        ),
        "best_regression_models": (
            BEST_REGRESSION_MODEL
        ),
        "best_classifier_models": (
            BEST_CLASSIFIER_MODEL
        ),
        "matched_oof_rows": len(
            comparison
        ),
        "target_mismatch_count": (
            mismatch_count
        ),
        "regression_failure_rule": (
            "PredictedStrength28_psi "
            "< ApplicableSpecifiedStrength28"
        ),
        "classifier_failure_rule": (
            "FailureProbability >= 0.50"
        ),
    }

    (
        OUTPUT_DIR
        / "comparison_run_info.json"
    ).write_text(
        json.dumps(
            run_info,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Terminal report.
    # -------------------------------------------------------------------------
    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        220,
    )

    print()
    print(
        "=" * 120
    )

    print(
        "REGRESSION-DERIVED PASS/FAIL "
        "VS DIRECT CLASSIFIER"
    )

    print(
        "=" * 120
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 120
    )

    print(
        "AGREEMENT / DISAGREEMENT"
    )

    print(
        "=" * 120
    )

    print(
        agreement.to_string(
            index=False
        )
    )

    print()
    print(
        "How to interpret:"
    )

    print(
        "1. FalseNegativesSavedByClassifier > 0 means "
        "the direct classifier catches failures that "
        "the regression-derived rule misses."
    )

    print(
        "2. AdditionalFalsePositivesFromClassifier > 0 means "
        "that improved recall requires more unnecessary reviews."
    )

    print(
        "3. If classifier recall is materially higher while "
        "the false-positive increase is operationally acceptable, "
        "the separate risk classifier adds business value."
    )

    print(
        "4. If both methods are nearly identical, the regression "
        "model alone may be sufficient for production."
    )

    print()
    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
