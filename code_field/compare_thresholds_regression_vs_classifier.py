from __future__ import annotations

"""
IMTS Field Core
Threshold Sweep Comparison:
Regression Margin Threshold vs Direct Classifier Probability Threshold

This script uses EXISTING OOF prediction files and does NOT retrain models.

It compares:

A) Regression-derived risk rule
   Predict FAIL / REVIEW when:
       PredictedMarginPsi <= margin_threshold

   where:
       PredictedMarginPsi =
           PredictedStrength28_psi - ApplicableSpecifiedStrength28

B) Direct classifier risk rule
   Predict FAIL / REVIEW when:
       FailureProbability >= probability_threshold

The goal is to compare both methods at similar recall levels.

Run:
    python code_field/compare_thresholds_regression_vs_classifier.py
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
    / "regression_vs_classifier_threshold_sweep"
)

REQUIRED_STRENGTH = "ApplicableSpecifiedStrength28"

BEST_REGRESSION_MODEL = {
    "Day0_FieldPlusRequired": "HistGradientBoosting",
    "Day7_FieldPlusRequired": "XGBoost",
    "Full_ContextPlusDay7": "HistGradientBoosting",
}

BEST_CLASSIFIER_MODEL = {
    "Day0_FieldPlusRequired": "HistGradientBoosting",
    "Day7_FieldPlusRequired": "HistGradientBoosting",
    "Full_ContextPlusDay7": "HistGradientBoosting",
}

# Regression margin thresholds in psi.
# Positive thresholds mean "review even if predicted strength is somewhat
# above required strength."
REGRESSION_MARGIN_THRESHOLDS = np.arange(
    -500,
    1001,
    50,
)

# Direct classifier probability thresholds.
CLASSIFIER_PROBABILITY_THRESHOLDS = np.round(
    np.arange(
        0.05,
        0.951,
        0.025,
    ),
    3,
)

# Recall levels for apples-to-apples comparison.
TARGET_RECALL_LEVELS = [
    0.80,
    0.85,
    0.90,
    0.95,
]


# =============================================================================
# HELPERS
# =============================================================================

def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    return pd.read_csv(
        path,
        low_memory=False,
    )


def classification_metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict[str, float | int]:
    actual_array = (
        actual.astype(int).to_numpy()
    )

    predicted_array = (
        predicted.astype(int).to_numpy()
    )

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


def filter_best_models(
    df: pd.DataFrame,
    lookup: dict[str, str],
) -> pd.DataFrame:
    frames = []

    for feature_set, model_name in lookup.items():
        subset = df.loc[
            (df["FeatureSet"] == feature_set)
            & (df["Model"] == model_name)
        ].copy()

        if subset.empty:
            raise ValueError(
                f"No rows found for "
                f"{feature_set} / {model_name}"
            )

        frames.append(subset)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def choose_best_at_recall(
    sweep: pd.DataFrame,
    target_recall: float,
) -> pd.Series | None:
    """
    Among thresholds meeting the minimum recall target,
    choose the row with the highest precision.
    """
    eligible = sweep.loc[
        sweep["Recall"] >= target_recall
    ].copy()

    if eligible.empty:
        return None

    return (
        eligible.sort_values(
            [
                "Precision",
                "FalsePositives",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .iloc[0]
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Reading existing OOF predictions..."
    )

    reg = read_csv(
        REGRESSION_PREDICTIONS
    )

    clf = read_csv(
        CLASSIFICATION_PREDICTIONS
    )

    clean = read_csv(
        CLEAN_DATA
    )

    reg = filter_best_models(
        reg,
        BEST_REGRESSION_MODEL,
    )

    clf = filter_best_models(
        clf,
        BEST_CLASSIFIER_MODEL,
    )

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
        raise ValueError(
            "Some regression OOF rows could not "
            "be matched to required strength."
        )

    reg[
        "PredictedMargin_psi"
    ] = (
        reg[
            "PredictedStrength28_psi"
        ]
        - reg[
            REQUIRED_STRENGTH
        ]
    )

    comparison = reg.merge(
        clf[
            [
                "Fold",
                "FeatureSet",
                "testId",
                "ActualFailureFlag",
                "FailureProbability",
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
            "No matching OOF rows were found."
        )

    # -------------------------------------------------------------------------
    # Sweep thresholds.
    # -------------------------------------------------------------------------
    sweep_rows = []

    for feature_set, group in comparison.groupby(
        "FeatureSet"
    ):
        actual = (
            group[
                "ActualFailureFlag"
            ].astype(int)
        )

        # Regression margin thresholds
        for threshold in (
            REGRESSION_MARGIN_THRESHOLDS
        ):
            predicted = (
                group[
                    "PredictedMargin_psi"
                ]
                <= threshold
            ).astype(int)

            metrics = classification_metrics(
                actual,
                predicted,
            )

            sweep_rows.append(
                {
                    "FeatureSet": feature_set,
                    "Method": "RegressionMargin",
                    "Threshold": float(
                        threshold
                    ),
                    "ThresholdUnit": "psi",
                    **metrics,
                }
            )

        # Classifier probability thresholds
        for threshold in (
            CLASSIFIER_PROBABILITY_THRESHOLDS
        ):
            predicted = (
                group[
                    "FailureProbability"
                ]
                >= threshold
            ).astype(int)

            metrics = classification_metrics(
                actual,
                predicted,
            )

            sweep_rows.append(
                {
                    "FeatureSet": feature_set,
                    "Method": "DirectClassifier",
                    "Threshold": float(
                        threshold
                    ),
                    "ThresholdUnit": "probability",
                    **metrics,
                }
            )

    sweep = pd.DataFrame(
        sweep_rows
    )

    # -------------------------------------------------------------------------
    # Compare both methods at matched minimum recall levels.
    # -------------------------------------------------------------------------
    matched_rows = []

    for feature_set in (
        sweep[
            "FeatureSet"
        ].unique()
    ):
        feature_sweep = (
            sweep.loc[
                sweep[
                    "FeatureSet"
                ]
                == feature_set
            ]
        )

        regression_sweep = (
            feature_sweep.loc[
                feature_sweep[
                    "Method"
                ]
                == "RegressionMargin"
            ]
        )

        classifier_sweep = (
            feature_sweep.loc[
                feature_sweep[
                    "Method"
                ]
                == "DirectClassifier"
            ]
        )

        for recall_target in (
            TARGET_RECALL_LEVELS
        ):
            regression_best = (
                choose_best_at_recall(
                    regression_sweep,
                    recall_target,
                )
            )

            classifier_best = (
                choose_best_at_recall(
                    classifier_sweep,
                    recall_target,
                )
            )

            row = {
                "FeatureSet": feature_set,
                "MinimumRecallTarget": (
                    recall_target
                ),
            }

            if regression_best is not None:
                row.update(
                    {
                        "RegressionMarginThreshold_psi": (
                            regression_best[
                                "Threshold"
                            ]
                        ),
                        "RegressionRecall": (
                            regression_best[
                                "Recall"
                            ]
                        ),
                        "RegressionPrecision": (
                            regression_best[
                                "Precision"
                            ]
                        ),
                        "RegressionF1": (
                            regression_best[
                                "F1"
                            ]
                        ),
                        "RegressionFalseNegatives": (
                            regression_best[
                                "FalseNegatives"
                            ]
                        ),
                        "RegressionFalsePositives": (
                            regression_best[
                                "FalsePositives"
                            ]
                        ),
                    }
                )
            else:
                row.update(
                    {
                        "RegressionMarginThreshold_psi": np.nan,
                        "RegressionRecall": np.nan,
                        "RegressionPrecision": np.nan,
                        "RegressionF1": np.nan,
                        "RegressionFalseNegatives": np.nan,
                        "RegressionFalsePositives": np.nan,
                    }
                )

            if classifier_best is not None:
                row.update(
                    {
                        "ClassifierProbabilityThreshold": (
                            classifier_best[
                                "Threshold"
                            ]
                        ),
                        "ClassifierRecall": (
                            classifier_best[
                                "Recall"
                            ]
                        ),
                        "ClassifierPrecision": (
                            classifier_best[
                                "Precision"
                            ]
                        ),
                        "ClassifierF1": (
                            classifier_best[
                                "F1"
                            ]
                        ),
                        "ClassifierFalseNegatives": (
                            classifier_best[
                                "FalseNegatives"
                            ]
                        ),
                        "ClassifierFalsePositives": (
                            classifier_best[
                                "FalsePositives"
                            ]
                        ),
                    }
                )
            else:
                row.update(
                    {
                        "ClassifierProbabilityThreshold": np.nan,
                        "ClassifierRecall": np.nan,
                        "ClassifierPrecision": np.nan,
                        "ClassifierF1": np.nan,
                        "ClassifierFalseNegatives": np.nan,
                        "ClassifierFalsePositives": np.nan,
                    }
                )

            if (
                regression_best is not None
                and classifier_best is not None
            ):
                row[
                    "PrecisionDifference_ClassifierMinusRegression"
                ] = (
                    row[
                        "ClassifierPrecision"
                    ]
                    - row[
                        "RegressionPrecision"
                    ]
                )

                row[
                    "FalsePositiveDifference_ClassifierMinusRegression"
                ] = (
                    row[
                        "ClassifierFalsePositives"
                    ]
                    - row[
                        "RegressionFalsePositives"
                    ]
                )

                row[
                    "FalseNegativeDifference_ClassifierMinusRegression"
                ] = (
                    row[
                        "ClassifierFalseNegatives"
                    ]
                    - row[
                        "RegressionFalseNegatives"
                    ]
                )
            else:
                row[
                    "PrecisionDifference_ClassifierMinusRegression"
                ] = np.nan

                row[
                    "FalsePositiveDifference_ClassifierMinusRegression"
                ] = np.nan

                row[
                    "FalseNegativeDifference_ClassifierMinusRegression"
                ] = np.nan

            matched_rows.append(
                row
            )

    matched = pd.DataFrame(
        matched_rows
    )

    # -------------------------------------------------------------------------
    # Best F1 threshold for each method.
    # Useful as a secondary reference, not necessarily the business choice.
    # -------------------------------------------------------------------------
    best_f1_rows = []

    for (
        feature_set,
        method,
    ), group in sweep.groupby(
        [
            "FeatureSet",
            "Method",
        ]
    ):
        best = (
            group.sort_values(
                [
                    "F1",
                    "Precision",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .iloc[0]
        )

        best_f1_rows.append(
            {
                "FeatureSet": (
                    feature_set
                ),
                "Method": method,
                "Threshold": (
                    best["Threshold"]
                ),
                "ThresholdUnit": (
                    best[
                        "ThresholdUnit"
                    ]
                ),
                "Recall": (
                    best["Recall"]
                ),
                "Precision": (
                    best[
                        "Precision"
                    ]
                ),
                "F1": (
                    best["F1"]
                ),
                "FalseNegatives": (
                    best[
                        "FalseNegatives"
                    ]
                ),
                "FalsePositives": (
                    best[
                        "FalsePositives"
                    ]
                ),
            }
        )

    best_f1 = pd.DataFrame(
        best_f1_rows
    )

    # -------------------------------------------------------------------------
    # Save results.
    # -------------------------------------------------------------------------
    sweep.to_csv(
        OUTPUT_DIR
        / "threshold_sweep_all.csv",
        index=False,
    )

    matched.to_csv(
        OUTPUT_DIR
        / "matched_recall_comparison.csv",
        index=False,
    )

    best_f1.to_csv(
        OUTPUT_DIR
        / "best_f1_thresholds.csv",
        index=False,
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "threshold_comparison_oof_detail.csv",
        index=False,
    )

    run_info = {
        "regression_prediction_file": str(
            REGRESSION_PREDICTIONS
        ),
        "classification_prediction_file": str(
            CLASSIFICATION_PREDICTIONS
        ),
        "clean_data_file": str(
            CLEAN_DATA
        ),
        "regression_margin_thresholds_psi": (
            REGRESSION_MARGIN_THRESHOLDS.tolist()
        ),
        "classifier_probability_thresholds": (
            CLASSIFIER_PROBABILITY_THRESHOLDS.tolist()
        ),
        "target_recall_levels": (
            TARGET_RECALL_LEVELS
        ),
        "interpretation": (
            "Compare precision and false positives at approximately "
            "the same minimum recall. The method with higher precision "
            "at the same recall provides more efficient risk screening."
        ),
    }

    (
        OUTPUT_DIR
        / "threshold_sweep_run_info.json"
    ).write_text(
        json.dumps(
            run_info,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Terminal output.
    # -------------------------------------------------------------------------
    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        240,
    )

    print()
    print(
        "=" * 140
    )

    print(
        "MATCHED-RECALL COMPARISON"
    )

    print(
        "=" * 140
    )

    print(
        matched.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 140
    )

    print(
        "BEST F1 THRESHOLDS"
    )

    print(
        "=" * 140
    )

    print(
        best_f1.to_string(
            index=False
        )
    )

    print()
    print(
        "How to interpret:"
    )

    print(
        "- Compare methods at the SAME recall target."
    )

    print(
        "- Higher precision at the same recall is better."
    )

    print(
        "- Fewer false positives at the same recall is better."
    )

    print(
        "- For IMTS risk screening, the 90% and 95% recall rows "
        "are especially useful because missed failures are likely "
        "more costly than unnecessary reviews."
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
