from __future__ import annotations

"""
IMTS Field Core
Threshold Sweep:
Regression Margin vs Direct Classifier
FOUR FEATURE SETS

This script does NOT retrain models.

It reads:
- Regression OOF predictions from the FOUR-feature regression CV
- Classification OOF predictions from the FOUR-feature classification CV
- Clean data only to retrieve ApplicableSpecifiedStrength28

Feature sets
------------
1) Day0_FieldPlusRequired
2) Day0_Context
3) Day7_FieldPlusRequired
4) Full_ContextPlusDay7

Required strength is included in every feature set and is also used to
calculate the regression-predicted margin.

Regression risk:
    PredictedStrength28 - RequiredStrength28 <= margin threshold

Classifier risk:
    FailureProbability >= probability threshold

The most useful output is matched_recall_comparison.csv.
It compares precision and false positives at similar recall targets.

Run:
    python code_field/compare_thresholds_regression_vs_classifier_four_features.py
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# =============================================================================
# CONFIG
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
    / "consolidated_four_model_cross_validation"
    / "cv_predictions.csv"
)

CLASSIFICATION_PREDICTIONS = (
    DATA_ROOT
    / "consolidated_four_feature_risk_classification_cv"
    / "classification_oof_predictions.csv"
)

CLEAN_DATA = (
    DATA_ROOT
    / "field_core_clean"
    / "field_core_clean_with_required.csv"
)

OUTPUT_DIR = (
    DATA_ROOT
    / "regression_vs_classifier_threshold_sweep_four_features"
)

REQUIRED_STRENGTH = "ApplicableSpecifiedStrength28"

FEATURE_SETS = [
    "Day0_FieldPlusRequired",
    "Day0_Context",
    "Day7_FieldPlusRequired",
    "Full_ContextPlusDay7",
]

# Best regression models are read automatically from the latest four-feature
# regression CV summary when available.
REGRESSION_BEST_FILE = (
    DATA_ROOT
    / "consolidated_four_model_cross_validation"
    / "best_model_by_feature_set_cv.csv"
)

# Best classifier models are read automatically from the new four-feature
# classification CV summary.
CLASSIFIER_BEST_FILE = (
    DATA_ROOT
    / "consolidated_four_feature_risk_classification_cv"
    / "best_classifier_by_feature_set_cv.csv"
)

REGRESSION_MARGIN_THRESHOLDS = np.arange(
    -500,
    1501,
    50,
)

CLASSIFIER_PROBABILITY_THRESHOLDS = np.round(
    np.arange(0.05, 0.951, 0.025),
    3,
)

TARGET_RECALL_LEVELS = [0.80, 0.85, 0.90, 0.95]


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
            recall_score(actual_array, predicted_array, zero_division=0)
        ),
        "Precision": float(
            precision_score(actual_array, predicted_array, zero_division=0)
        ),
        "F1": float(
            f1_score(actual_array, predicted_array, zero_division=0)
        ),
        "TrueNegatives": int(tn),
        "FalsePositives": int(fp),
        "FalseNegatives": int(fn),
        "TruePositives": int(tp),
    }


def load_best_model_lookup(
    path: Path,
    model_column: str = "Model",
) -> dict[str, str]:
    df = read_csv(path)

    required = {"FeatureSet", model_column}
    missing = required - set(df.columns)

    if missing:
        raise KeyError(
            f"{path.name} is missing columns: {sorted(missing)}"
        )

    lookup = dict(
        zip(
            df["FeatureSet"].astype(str),
            df[model_column].astype(str),
        )
    )

    missing_sets = [
        fs for fs in FEATURE_SETS
        if fs not in lookup
    ]

    if missing_sets:
        raise ValueError(
            f"{path.name} does not contain all four feature sets: "
            f"{missing_sets}"
        )

    return {
        fs: lookup[fs]
        for fs in FEATURE_SETS
    }


def filter_best_models(
    df: pd.DataFrame,
    lookup: dict[str, str],
) -> pd.DataFrame:
    frames = []

    for feature_set in FEATURE_SETS:
        model_name = lookup[feature_set]

        subset = df.loc[
            (df["FeatureSet"] == feature_set)
            & (df["Model"] == model_name)
        ].copy()

        if subset.empty:
            raise ValueError(
                f"No OOF rows found for {feature_set} / {model_name}"
            )

        frames.append(subset)

    return pd.concat(frames, ignore_index=True)


def choose_best_at_recall(
    sweep: pd.DataFrame,
    target_recall: float,
) -> pd.Series | None:
    eligible = sweep.loc[
        sweep["Recall"] >= target_recall
    ].copy()

    if eligible.empty:
        return None

    # At the requested minimum recall, choose the most precise / efficient rule.
    return (
        eligible.sort_values(
            ["Precision", "FalsePositives"],
            ascending=[False, True],
        )
        .iloc[0]
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading four-feature OOF predictions...")

    reg = read_csv(REGRESSION_PREDICTIONS)
    clf = read_csv(CLASSIFICATION_PREDICTIONS)
    clean = read_csv(CLEAN_DATA)

    regression_models = load_best_model_lookup(
        REGRESSION_BEST_FILE
    )
    classifier_models = load_best_model_lookup(
        CLASSIFIER_BEST_FILE
    )

    print("\nBest regression models:")
    for feature_set, model in regression_models.items():
        print(f"  {feature_set}: {model}")

    print("\nBest classifier models:")
    for feature_set, model in classifier_models.items():
        print(f"  {feature_set}: {model}")

    reg = filter_best_models(reg, regression_models)
    clf = filter_best_models(clf, classifier_models)

    if "testId" not in clean.columns:
        raise KeyError("Clean data must contain testId.")

    if REQUIRED_STRENGTH not in clean.columns:
        raise KeyError(
            f"Clean data must contain {REQUIRED_STRENGTH}."
        )

    required_lookup = (
        clean[["testId", REQUIRED_STRENGTH]]
        .drop_duplicates(subset=["testId"])
        .copy()
    )

    required_lookup[REQUIRED_STRENGTH] = pd.to_numeric(
        required_lookup[REQUIRED_STRENGTH],
        errors="coerce",
    )

    reg = reg.merge(
        required_lookup,
        on="testId",
        how="left",
        validate="many_to_one",
    )

    if reg[REQUIRED_STRENGTH].isna().any():
        raise ValueError(
            "Some regression OOF rows could not be matched "
            "to ApplicableSpecifiedStrength28."
        )

    reg["PredictedMargin_psi"] = (
        reg["PredictedStrength28_psi"]
        - reg[REQUIRED_STRENGTH]
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
        on=["Fold", "FeatureSet", "testId"],
        how="inner",
        validate="one_to_one",
    )

    if comparison.empty:
        raise ValueError(
            "Regression and classifier OOF predictions did not match."
        )

    print(f"\nMatched OOF rows: {len(comparison):,}")

    # =========================================================================
    # THRESHOLD SWEEP
    # =========================================================================

    sweep_rows = []

    for feature_set, group in comparison.groupby("FeatureSet"):
        actual = group["ActualFailureFlag"].astype(int)

        for threshold in REGRESSION_MARGIN_THRESHOLDS:
            predicted = (
                group["PredictedMargin_psi"] <= threshold
            ).astype(int)

            sweep_rows.append(
                {
                    "FeatureSet": feature_set,
                    "Method": "RegressionMargin",
                    "Threshold": float(threshold),
                    "ThresholdUnit": "psi",
                    **classification_metrics(actual, predicted),
                }
            )

        for threshold in CLASSIFIER_PROBABILITY_THRESHOLDS:
            predicted = (
                group["FailureProbability"] >= threshold
            ).astype(int)

            sweep_rows.append(
                {
                    "FeatureSet": feature_set,
                    "Method": "DirectClassifier",
                    "Threshold": float(threshold),
                    "ThresholdUnit": "probability",
                    **classification_metrics(actual, predicted),
                }
            )

    sweep = pd.DataFrame(sweep_rows)

    # =========================================================================
    # MATCHED-RECALL COMPARISON
    # =========================================================================

    matched_rows = []

    for feature_set in FEATURE_SETS:
        feature_sweep = sweep.loc[
            sweep["FeatureSet"] == feature_set
        ]

        reg_sweep = feature_sweep.loc[
            feature_sweep["Method"] == "RegressionMargin"
        ]

        clf_sweep = feature_sweep.loc[
            feature_sweep["Method"] == "DirectClassifier"
        ]

        for recall_target in TARGET_RECALL_LEVELS:
            reg_best = choose_best_at_recall(
                reg_sweep,
                recall_target,
            )
            clf_best = choose_best_at_recall(
                clf_sweep,
                recall_target,
            )

            row = {
                "FeatureSet": feature_set,
                "MinimumRecallTarget": recall_target,
            }

            if reg_best is not None:
                row.update(
                    {
                        "RegressionMarginThreshold_psi": reg_best["Threshold"],
                        "RegressionRecall": reg_best["Recall"],
                        "RegressionPrecision": reg_best["Precision"],
                        "RegressionF1": reg_best["F1"],
                        "RegressionFalseNegatives": reg_best["FalseNegatives"],
                        "RegressionFalsePositives": reg_best["FalsePositives"],
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

            if clf_best is not None:
                row.update(
                    {
                        "ClassifierProbabilityThreshold": clf_best["Threshold"],
                        "ClassifierRecall": clf_best["Recall"],
                        "ClassifierPrecision": clf_best["Precision"],
                        "ClassifierF1": clf_best["F1"],
                        "ClassifierFalseNegatives": clf_best["FalseNegatives"],
                        "ClassifierFalsePositives": clf_best["FalsePositives"],
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

            if reg_best is not None and clf_best is not None:
                row[
                    "PrecisionDifference_ClassifierMinusRegression"
                ] = (
                    row["ClassifierPrecision"]
                    - row["RegressionPrecision"]
                )
                row[
                    "FalsePositiveDifference_ClassifierMinusRegression"
                ] = (
                    row["ClassifierFalsePositives"]
                    - row["RegressionFalsePositives"]
                )
                row[
                    "FalseNegativeDifference_ClassifierMinusRegression"
                ] = (
                    row["ClassifierFalseNegatives"]
                    - row["RegressionFalseNegatives"]
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

            matched_rows.append(row)

    matched = pd.DataFrame(matched_rows)

    # Best F1 as a secondary reference.
    best_f1_rows = []

    for (feature_set, method), group in sweep.groupby(
        ["FeatureSet", "Method"]
    ):
        best = (
            group.sort_values(
                ["F1", "Precision"],
                ascending=[False, False],
            )
            .iloc[0]
        )

        best_f1_rows.append(
            {
                "FeatureSet": feature_set,
                "Method": method,
                "Threshold": best["Threshold"],
                "ThresholdUnit": best["ThresholdUnit"],
                "Recall": best["Recall"],
                "Precision": best["Precision"],
                "F1": best["F1"],
                "FalseNegatives": best["FalseNegatives"],
                "FalsePositives": best["FalsePositives"],
            }
        )

    best_f1 = pd.DataFrame(best_f1_rows)

    # =========================================================================
    # SAVE
    # =========================================================================

    sweep.to_csv(
        OUTPUT_DIR / "threshold_sweep_all.csv",
        index=False,
    )

    matched.to_csv(
        OUTPUT_DIR / "matched_recall_comparison.csv",
        index=False,
    )

    best_f1.to_csv(
        OUTPUT_DIR / "best_f1_thresholds.csv",
        index=False,
    )

    comparison.to_csv(
        OUTPUT_DIR / "threshold_comparison_oof_detail.csv",
        index=False,
    )

    (OUTPUT_DIR / "threshold_sweep_run_info.json").write_text(
        json.dumps(
            {
                "feature_sets": FEATURE_SETS,
                "regression_models": regression_models,
                "classifier_models": classifier_models,
                "required_strength": REQUIRED_STRENGTH,
                "regression_prediction_file": str(REGRESSION_PREDICTIONS),
                "classification_prediction_file": str(CLASSIFICATION_PREDICTIONS),
                "target_recall_levels": TARGET_RECALL_LEVELS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 260)

    print("\nMATCHED-RECALL COMPARISON")
    print("=" * 150)
    print(matched.to_string(index=False))

    print("\nBEST F1 THRESHOLDS")
    print("=" * 150)
    print(best_f1.to_string(index=False))

    print(f"\nOutput: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
