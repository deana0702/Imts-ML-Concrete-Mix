from __future__ import annotations

"""
IMTS Field Core
Fixed-threshold comparison:
Regression margin rule vs direct classifier (FOUR feature sets)

This script does NOT retrain models.

It reads:
- Regression OOF predictions from four-feature regression CV
- Classification OOF predictions from four-feature classification CV
- Clean data to retrieve ApplicableSpecifiedStrength28

Rules:
- Regression risk: PredictedStrength28 - RequiredStrength28 <= 0.5 psi
- Classifier risk: FailureProbability >= 0.5

Run:
    python code_field/compare_regression_vs_classifier_fixed_threshold_four_features.py
"""

from pathlib import Path
import json

import pandas as pd


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

REGRESSION_BEST_FILE = (
    DATA_ROOT
    / "consolidated_four_model_cross_validation"
    / "best_model_by_feature_set_cv.csv"
)

CLASSIFIER_BEST_FILE = (
    DATA_ROOT
    / "consolidated_four_feature_risk_classification_cv"
    / "best_classifier_by_feature_set_cv.csv"
)

OUTPUT_DIR = (
    DATA_ROOT
    / "regression_vs_classifier_fixed_threshold_four_features"
)

REQUIRED_STRENGTH = "ApplicableSpecifiedStrength28"

FEATURE_SETS = [
    "Day0_FieldPlusRequired",
    "Day0_Context",
    "Day7_FieldPlusRequired",
    "Full_ContextPlusDay7",
]

REGRESSION_MARGIN_THRESHOLD_PSI = 0.5
CLASSIFIER_PROBABILITY_THRESHOLD = 0.5


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found:\n{path}")
    return pd.read_csv(path, low_memory=False)


def load_best_model_lookup(path: Path, model_column: str = "Model") -> dict[str, str]:
    df = read_csv(path)

    required = {"FeatureSet", model_column}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{path.name} is missing columns: {sorted(missing)}")

    lookup = dict(zip(df["FeatureSet"].astype(str), df[model_column].astype(str)))

    missing_sets = [feature_set for feature_set in FEATURE_SETS if feature_set not in lookup]
    if missing_sets:
        raise ValueError(f"{path.name} does not contain all feature sets: {missing_sets}")

    return {feature_set: lookup[feature_set] for feature_set in FEATURE_SETS}


def filter_best_models(df: pd.DataFrame, lookup: dict[str, str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for feature_set in FEATURE_SETS:
        model_name = lookup[feature_set]
        subset = df.loc[(df["FeatureSet"] == feature_set) & (df["Model"] == model_name)].copy()
        if subset.empty:
            raise ValueError(f"No OOF rows found for {feature_set} / {model_name}")
        frames.append(subset)
    return pd.concat(frames, ignore_index=True)


def compute_recall_precision(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    actual_int = actual.astype(int)
    predicted_int = predicted.astype(int)

    tp = int(((actual_int == 1) & (predicted_int == 1)).sum())
    fn = int(((actual_int == 1) & (predicted_int == 0)).sum())
    fp = int(((actual_int == 0) & (predicted_int == 1)).sum())

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    return {
        "Recall": float(recall),
        "Precision": float(precision),
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reg = read_csv(REGRESSION_PREDICTIONS)
    clf = read_csv(CLASSIFICATION_PREDICTIONS)
    clean = read_csv(CLEAN_DATA)

    regression_models = load_best_model_lookup(REGRESSION_BEST_FILE)
    classifier_models = load_best_model_lookup(CLASSIFIER_BEST_FILE)

    reg = filter_best_models(reg, regression_models)
    clf = filter_best_models(clf, classifier_models)

    if "testId" not in clean.columns:
        raise KeyError("Clean data must contain testId.")

    if REQUIRED_STRENGTH not in clean.columns:
        raise KeyError(f"Clean data must contain {REQUIRED_STRENGTH}.")

    required_lookup = clean[["testId", REQUIRED_STRENGTH]].drop_duplicates(subset=["testId"]).copy()
    required_lookup[REQUIRED_STRENGTH] = pd.to_numeric(required_lookup[REQUIRED_STRENGTH], errors="coerce")

    reg = reg.merge(required_lookup, on="testId", how="left", validate="many_to_one")

    if reg[REQUIRED_STRENGTH].isna().any():
        raise ValueError("Some regression OOF rows could not be matched to required strength.")

    reg["PredictedMargin_psi"] = reg["PredictedStrength28_psi"] - reg[REQUIRED_STRENGTH]
    reg["RegressionRiskFlag"] = (reg["PredictedMargin_psi"] <= REGRESSION_MARGIN_THRESHOLD_PSI).astype(int)

    clf["ClassifierRiskFlag"] = (
        clf["FailureProbability"] >= CLASSIFIER_PROBABILITY_THRESHOLD
    ).astype(int)

    comparison = reg.merge(
        clf[["Fold", "FeatureSet", "testId", "ActualFailureFlag", "ClassifierRiskFlag"]],
        on=["Fold", "FeatureSet", "testId"],
        how="inner",
        validate="one_to_one",
    )

    if comparison.empty:
        raise ValueError("Regression and classifier OOF predictions did not match.")

    rows = []
    for feature_set, group in comparison.groupby("FeatureSet"):
        actual = group["ActualFailureFlag"].astype(int)
        reg_metrics = compute_recall_precision(actual, group["RegressionRiskFlag"])
        clf_metrics = compute_recall_precision(actual, group["ClassifierRiskFlag"])

        rows.append(
            {
                "FeatureSet": feature_set,
                "RegressionModel": regression_models[feature_set],
                "RegressionMarginThreshold_psi": REGRESSION_MARGIN_THRESHOLD_PSI,
                "RegressionRecall": reg_metrics["Recall"],
                "RegressionPrecision": reg_metrics["Precision"],
                "ClassifierModel": classifier_models[feature_set],
                "ClassifierProbabilityThreshold": CLASSIFIER_PROBABILITY_THRESHOLD,
                "ClassifierRecall": clf_metrics["Recall"],
                "ClassifierPrecision": clf_metrics["Precision"],
            }
        )

    summary = pd.DataFrame(rows)

    summary.to_csv(OUTPUT_DIR / "fixed_threshold_recall_precision.csv", index=False)

    (OUTPUT_DIR / "fixed_threshold_run_info.json").write_text(
        json.dumps(
            {
                "feature_sets": FEATURE_SETS,
                "regression_models": regression_models,
                "classifier_models": classifier_models,
                "required_strength": REQUIRED_STRENGTH,
                "regression_margin_threshold_psi": REGRESSION_MARGIN_THRESHOLD_PSI,
                "classifier_probability_threshold": CLASSIFIER_PROBABILITY_THRESHOLD,
                "regression_prediction_file": str(REGRESSION_PREDICTIONS),
                "classification_prediction_file": str(CLASSIFICATION_PREDICTIONS),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    print("\nFIXED-THRESHOLD RECALL/PRECISION COMPARISON")
    print("=" * 120)
    print(summary.to_string(index=False))
    print(f"\nOutput: {OUTPUT_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())