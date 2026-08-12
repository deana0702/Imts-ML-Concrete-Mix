from __future__ import annotations

"""
IMTS Field Core - Risk Classification
5-Fold Project-Grouped Cross-Validation
FOUR FEATURE SETS

All records already have a valid ApplicableSpecifiedStrength28, and that
required 28-day strength is included in every feature set.

Feature sets
------------
1) Day0_FieldPlusRequired
   Day-0 field measurements + ApplicableSpecifiedStrength28

2) Day0_Context
   Day-0 field measurements + ApplicableSpecifiedStrength28
   + Supplier / Plant / Mix context

3) Day7_FieldPlusRequired
   Day-0 field measurements + ApplicableSpecifiedStrength28
   + Day-7 strength features

4) Full_ContextPlusDay7
   Day-0 field measurements + ApplicableSpecifiedStrength28
   + Supplier / Plant / Mix context
   + Day-7 strength features

Binary analytical target
------------------------
FailureFlag28 = 1 when:
    AverageActualStrength28_psi < ApplicableSpecifiedStrength28
otherwise 0.

Important
---------
- This is an analytical below-required-strength screening target, not the
  complete engineering acceptance rule.
- Outer CV is grouped by project.
- Supplier / Plant / Mix target encoding is rebuilt inside every outer fold.
- Validation outcomes never contribute to their own context encoding.
- All four feature sets use the same rows with valid Day-7 strength so the
  feature-set comparison is directly comparable.
- XGBoost is included.

Run:
    python code_field/field_core_four_feature_risk_classification_cv.py

Dependency:
    pip install xgboost
"""

from dataclasses import dataclass
from pathlib import Path
import json
import time
from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError(
        "xgboost is required. Install it with: pip install xgboost"
    ) from exc


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
DATA_DIR = REPO_ROOT / "data"
FIELD_CORE_OUTPUT_ROOT = DATA_DIR / "field_core_outputs"

INPUT_CANDIDATES = [
    FIELD_CORE_OUTPUT_ROOT / "field_core_clean" / "field_core_clean_with_required.csv",
    DATA_DIR / "field_core_clean" / "field_core_clean_with_required.csv",
]

OUTPUT_DIR = (
    FIELD_CORE_OUTPUT_ROOT
    / "consolidated_four_feature_risk_classification_cv"
)

TARGET_STRENGTH = "AverageActualStrength28_psi"
REQUIRED_STRENGTH = "ApplicableSpecifiedStrength28"
RISK_TARGET = "FailureFlag28"

RANDOM_STATE = 42
OUTER_CV_FOLDS = 5
TARGET_ENCODING_FOLDS = 5
TARGET_ENCODING_SMOOTHING = 20.0
REPORT_THRESHOLD = 0.50

FIELD_FEATURES = [
    "EffectiveSlump_in",
    "EffectiveAir_percent",
    "EffectiveUnitWeight_lb_ft3",
    "EffectiveConcreteTemp_F",
    "AmbientTemp_F",
    "WaterAdded_gal_per_yd3",
    "BatchToSampleMinutes",
    "BatchToCastMinutes",
    "HasWaterAdded",
    "HasAnyAfterSPMeasurement",
]

# Required 28-day strength is a baseline feature in EVERY feature set.
DAY0_FEATURES = FIELD_FEATURES + [REQUIRED_STRENGTH]

DAY7_AVERAGE_CANDIDATES = [
    "AverageActualStrength7_psi",
    "AverageActualStrength7",
]

DAY7_COUNT_CANDIDATES = [
    "ActualStrength7SpecimenCount",
    "StandardCuredStrength7SpecimenCount",
]

DAY7_FEATURES = [
    "Day7AverageStrength_psi",
    "Day7SpecimenCount",
    "Day7MarginToRequired_psi",
    "Day7ToRequiredRatio",
]

CONTEXT_COLUMN_CANDIDATES = {
    "Supplier": ["SupplierId", "supplierId", "SupplierName", "supplierName"],
    "Plant": ["PlantNumber", "plantNumber", "PlantNo", "plantNo"],
    "Mix": ["MixNumber", "mixNumber", "MixNo", "mixNo"],
}

GROUP_COLUMN_CANDIDATES = ["projectId", "projectNo", "ProjectId", "ProjectNo"]


@dataclass(frozen=True)
class ContextSources:
    supplier: str
    plant: str
    mix: str


# =============================================================================
# HELPERS
# =============================================================================

def first_existing_path(paths: Iterable[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find cleaned dataset.\n"
        + "\n".join(f"  - {p}" for p in paths)
    )


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            pass
    raise UnicodeError(f"Could not determine CSV encoding: {path}")


def resolve_first_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)

    s = df[column]
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    return pd.to_numeric(
        s.astype("string").str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    require_columns(df, columns)
    out = pd.DataFrame(index=df.index)
    for column in columns:
        out[column] = numeric_series(df, column)
    return out


def save_json(data: dict[str, object], path: Path) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


# =============================================================================
# GROUPING
# =============================================================================

def resolve_group_column(df: pd.DataFrame) -> str:
    column = resolve_first_column(df, GROUP_COLUMN_CANDIDATES)
    if column is None:
        raise KeyError(f"No project grouping column: {GROUP_COLUMN_CANDIDATES}")
    return column


def make_project_groups(df: pd.DataFrame, group_column: str) -> pd.Series:
    groups = df[group_column].astype("string").str.strip()
    missing = groups.isna() | groups.eq("")

    if "testId" in df.columns:
        fallback = "MISSING_PROJECT_TEST_" + df["testId"].astype("string")
    else:
        fallback = (
            "MISSING_PROJECT_ROW_"
            + pd.Series(df.index, index=df.index).astype("string")
        )

    return groups.mask(missing, fallback)


# =============================================================================
# DAY 7 + FAILURE TARGET
# =============================================================================

def add_day7_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    avg_col = resolve_first_column(result, DAY7_AVERAGE_CANDIDATES)
    count_col = resolve_first_column(result, DAY7_COUNT_CANDIDATES)

    if avg_col is None:
        raise KeyError(f"No Day-7 average column: {DAY7_AVERAGE_CANDIDATES}")

    result["Day7AverageStrength_psi"] = numeric_series(result, avg_col)

    if count_col is None:
        result["Day7SpecimenCount"] = np.nan
    else:
        result["Day7SpecimenCount"] = numeric_series(result, count_col)

    required = numeric_series(result, REQUIRED_STRENGTH)

    result["Day7MarginToRequired_psi"] = (
        result["Day7AverageStrength_psi"] - required
    )
    result["Day7ToRequiredRatio"] = (
        result["Day7AverageStrength_psi"] / required.replace(0, np.nan)
    )

    return result


def add_failure_target(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    actual = numeric_series(result, TARGET_STRENGTH)
    required = numeric_series(result, REQUIRED_STRENGTH)
    result[RISK_TARGET] = (actual < required).astype(int)
    return result


# =============================================================================
# CONTEXT ENCODING
# =============================================================================

def normalize_category(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("__MISSING__")
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
        .replace("", "__MISSING__")
    )


def resolve_context_sources(df: pd.DataFrame) -> ContextSources:
    resolved: dict[str, str] = {}

    for logical_name, candidates in CONTEXT_COLUMN_CANDIDATES.items():
        column = resolve_first_column(df, candidates)
        if column is None:
            raise KeyError(
                f"Context field '{logical_name}' not found. Expected: {candidates}"
            )
        resolved[logical_name] = column

    return ContextSources(
        supplier=resolved["Supplier"],
        plant=resolved["Plant"],
        mix=resolved["Mix"],
    )


def build_context_categories(
    df: pd.DataFrame,
    sources: ContextSources,
) -> pd.DataFrame:
    context = pd.DataFrame(index=df.index)

    context["SupplierCategory"] = normalize_category(df[sources.supplier])
    context["PlantCategory"] = normalize_category(df[sources.plant])
    context["MixCategory"] = normalize_category(df[sources.mix])

    context["SupplierPlantCategory"] = (
        context["SupplierCategory"] + "|" + context["PlantCategory"]
    )

    context["SupplierPlantMixCategory"] = (
        context["SupplierCategory"]
        + "|"
        + context["PlantCategory"]
        + "|"
        + context["MixCategory"]
    )

    return context


def smoothed_target_mapping(
    category: pd.Series,
    target: pd.Series,
    global_mean: float,
    smoothing: float,
) -> pd.Series:
    stats = pd.DataFrame(
        {"category": category, "target": target}
    ).groupby("category", dropna=False)["target"].agg(["sum", "count"])

    return (
        stats["sum"] + smoothing * global_mean
    ) / (
        stats["count"] + smoothing
    )


def cross_fitted_failure_target_encode(
    train_categories: pd.DataFrame,
    validation_categories: pd.DataFrame,
    y_train: pd.Series,
    train_groups: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Context target means represent historical failure rate.
    Outer validation labels never contribute to their own encoding.
    """
    groups = train_groups.astype("string").fillna("__MISSING_GROUP__")
    n_splits = min(TARGET_ENCODING_FOLDS, int(groups.nunique()))

    if n_splits < 2:
        raise ValueError("At least two project groups are required.")

    splitter = GroupKFold(n_splits=n_splits)
    global_failure_rate = float(y_train.mean())

    enc_train = pd.DataFrame(index=train_categories.index)
    enc_valid = pd.DataFrame(index=validation_categories.index)

    for column in train_categories.columns:
        train_values = normalize_category(train_categories[column])
        valid_values = normalize_category(validation_categories[column])

        oof = pd.Series(np.nan, index=train_categories.index, dtype=float)

        for fit_pos, encode_pos in splitter.split(
            train_categories,
            y_train,
            groups,
        ):
            fit_idx = train_categories.index[fit_pos]
            encode_idx = train_categories.index[encode_pos]

            mapping = smoothed_target_mapping(
                train_values.loc[fit_idx],
                y_train.loc[fit_idx],
                global_failure_rate,
                TARGET_ENCODING_SMOOTHING,
            )

            oof.loc[encode_idx] = (
                train_values.loc[encode_idx]
                .map(mapping)
                .fillna(global_failure_rate)
            )

        full_mapping = smoothed_target_mapping(
            train_values,
            y_train,
            global_failure_rate,
            TARGET_ENCODING_SMOOTHING,
        )

        counts = train_values.value_counts(dropna=False)
        unknown = ~valid_values.isin(full_mapping.index)

        enc_train[f"{column}_FailureRate"] = oof.fillna(global_failure_rate)
        enc_valid[f"{column}_FailureRate"] = (
            valid_values.map(full_mapping).fillna(global_failure_rate)
        )

        enc_train[f"{column}_LogCount"] = np.log1p(
            train_values.map(counts).fillna(0).astype(float)
        )
        enc_valid[f"{column}_LogCount"] = np.log1p(
            valid_values.map(counts).fillna(0).astype(float)
        )

        enc_train[f"{column}_Unknown"] = 0
        enc_valid[f"{column}_Unknown"] = unknown.astype(int)

    return enc_train, enc_valid


# =============================================================================
# MODELS
# =============================================================================

def build_classification_models() -> dict[str, object]:
    return {
        "DummyPrior": DummyClassifier(strategy="prior"),

        "LogisticRegression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "RandomForest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=5,
                        max_features=0.8,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "HistGradientBoosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=300,
                        learning_rate=0.05,
                        max_leaf_nodes=31,
                        min_samples_leaf=20,
                        l2_regularization=1.0,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "XGBoost": XGBClassifier(
            objective="binary:logistic",
            n_estimators=700,
            learning_rate=0.04,
            max_depth=6,
            min_child_weight=5,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="logloss",
        ),
    }


def fit_balanced(
    model: object,
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> object:
    weights = compute_sample_weight(class_weight="balanced", y=y_train)

    if isinstance(model, Pipeline):
        model.fit(x_train, y_train, model__sample_weight=weights)
    else:
        model.fit(x_train, y_train, sample_weight=weights)

    return model


# =============================================================================
# METRICS
# =============================================================================

def classification_metrics(
    actual: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predicted = (probability >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    ).ravel()

    return {
        "PR_AUC": float(average_precision_score(actual, probability)),
        "ROC_AUC": float(roc_auc_score(actual, probability)),
        "BrierScore": float(brier_score_loss(actual, probability)),
        "Recall": float(recall_score(actual, predicted, zero_division=0)),
        "Precision": float(precision_score(actual, predicted, zero_division=0)),
        "F1": float(f1_score(actual, predicted, zero_division=0)),
        "TrueNegatives": int(tn),
        "FalsePositives": int(fp),
        "FalseNegatives": int(fn),
        "TruePositives": int(tp),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_file = first_existing_path(INPUT_CANDIDATES)
    print(f"Input: {input_file}")

    df = read_csv(input_file)

    require_columns(
        df,
        [TARGET_STRENGTH, REQUIRED_STRENGTH, *FIELD_FEATURES],
    )

    actual = numeric_series(df, TARGET_STRENGTH)
    required = numeric_series(df, REQUIRED_STRENGTH)

    # Required 28-day strength is mandatory for this analysis.
    df = df.loc[actual.gt(0) & required.gt(0)].copy()

    df = add_day7_features(df)

    # Same rows for all four feature sets.
    df = df.loc[
        numeric_series(df, "Day7AverageStrength_psi").gt(0)
    ].copy()

    df = add_failure_target(df)

    print(f"Rows: {len(df):,}")
    print(f"Failures: {int(df[RISK_TARGET].sum()):,}")
    print(f"Failure rate: {df[RISK_TARGET].mean():.2%}")

    group_column = resolve_group_column(df)
    groups = make_project_groups(df, group_column)

    print(f"Projects: {groups.nunique():,}")

    context_sources = resolve_context_sources(df)
    outer = GroupKFold(n_splits=OUTER_CV_FOLDS)

    metric_rows = []
    prediction_frames = []

    for fold, (train_pos, valid_pos) in enumerate(
        outer.split(df, groups=groups),
        start=1,
    ):
        print(f"\nFold {fold}/{OUTER_CV_FOLDS}")

        train = df.iloc[train_pos].copy()
        valid = df.iloc[valid_pos].copy()

        train_groups = make_project_groups(train, group_column)

        y_train = train[RISK_TARGET].astype(int)
        y_valid = valid[RISK_TARGET].astype(int)

        day0_train = numeric_frame(train, DAY0_FEATURES)
        day0_valid = numeric_frame(valid, DAY0_FEATURES)

        day7_train = numeric_frame(train, DAY7_FEATURES)
        day7_valid = numeric_frame(valid, DAY7_FEATURES)

        train_context = build_context_categories(train, context_sources)
        valid_context = build_context_categories(valid, context_sources)

        enc_train, enc_valid = cross_fitted_failure_target_encode(
            train_context,
            valid_context,
            y_train,
            train_groups,
        )

        feature_sets = {
            "Day0_FieldPlusRequired": (
                day0_train,
                day0_valid,
            ),

            "Day0_Context": (
                pd.concat([day0_train, enc_train], axis=1),
                pd.concat([day0_valid, enc_valid], axis=1),
            ),

            "Day7_FieldPlusRequired": (
                pd.concat([day0_train, day7_train], axis=1),
                pd.concat([day0_valid, day7_valid], axis=1),
            ),

            "Full_ContextPlusDay7": (
                pd.concat([day0_train, enc_train, day7_train], axis=1),
                pd.concat([day0_valid, enc_valid, day7_valid], axis=1),
            ),
        }

        for feature_set, (x_train, x_valid) in feature_sets.items():
            for model_name, model in build_classification_models().items():
                print(f"  {feature_set} | {model_name}")

                start = time.perf_counter()
                model = fit_balanced(model, x_train, y_train)
                probability = model.predict_proba(x_valid)[:, 1]
                elapsed = time.perf_counter() - start

                metrics = classification_metrics(
                    y_valid.to_numpy(),
                    probability,
                    REPORT_THRESHOLD,
                )

                metric_rows.append(
                    {
                        "Fold": fold,
                        "FeatureSet": feature_set,
                        "Model": model_name,
                        "TrainingSeconds": elapsed,
                        **metrics,
                    }
                )

                pred = pd.DataFrame(
                    {
                        "Fold": fold,
                        "FeatureSet": feature_set,
                        "Model": model_name,
                        "ActualFailureFlag": y_valid.to_numpy(),
                        "FailureProbability": probability,
                        "PredictedFailureAt050": (
                            probability >= REPORT_THRESHOLD
                        ).astype(int),
                    },
                    index=valid.index,
                )

                for identifier in [
                    "testId",
                    "projectId",
                    "projectNo",
                    "officeId",
                    "OfficeName",
                ]:
                    if identifier in valid.columns:
                        pred[identifier] = valid[identifier].to_numpy()

                prediction_frames.append(pred.reset_index(drop=True))

    fold_metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    summary = (
        fold_metrics.groupby(["FeatureSet", "Model"], as_index=False)
        .agg(
            MeanCV_PR_AUC=("PR_AUC", "mean"),
            StdCV_PR_AUC=("PR_AUC", "std"),
            MeanCV_ROC_AUC=("ROC_AUC", "mean"),
            StdCV_ROC_AUC=("ROC_AUC", "std"),
            MeanCV_Recall=("Recall", "mean"),
            StdCV_Recall=("Recall", "std"),
            MeanCV_Precision=("Precision", "mean"),
            StdCV_Precision=("Precision", "std"),
            MeanCV_F1=("F1", "mean"),
            MeanCV_Brier=("BrierScore", "mean"),
            MeanCV_FalseNegatives=("FalseNegatives", "mean"),
            MeanCV_FalsePositives=("FalsePositives", "mean"),
        )
        .sort_values(
            ["MeanCV_PR_AUC", "MeanCV_ROC_AUC"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )

    best = (
        summary.sort_values(
            ["MeanCV_PR_AUC", "MeanCV_ROC_AUC"],
            ascending=[False, False],
        )
        .groupby("FeatureSet", as_index=False)
        .first()
        .sort_values("MeanCV_PR_AUC", ascending=False)
        .reset_index(drop=True)
    )

    fold_metrics.to_csv(
        OUTPUT_DIR / "classification_cv_fold_metrics.csv",
        index=False,
    )
    summary.to_csv(
        OUTPUT_DIR / "classification_cv_summary.csv",
        index=False,
    )
    best.to_csv(
        OUTPUT_DIR / "best_classifier_by_feature_set_cv.csv",
        index=False,
    )
    predictions.to_csv(
        OUTPUT_DIR / "classification_oof_predictions.csv",
        index=False,
    )

    save_json(
        {
            "input_file": str(input_file),
            "rows_used": len(df),
            "failure_rate": float(df[RISK_TARGET].mean()),
            "group_column": group_column,
            "feature_sets": [
                "Day0_FieldPlusRequired",
                "Day0_Context",
                "Day7_FieldPlusRequired",
                "Full_ContextPlusDay7",
            ],
            "required_strength_note": (
                "ApplicableSpecifiedStrength28 is mandatory for eligibility "
                "and is included in every feature set."
            ),
        },
        OUTPUT_DIR / "classification_cv_run_summary.json",
    )

    print("\nBEST CLASSIFIER BY FEATURE SET")
    print(
        best[
            [
                "FeatureSet",
                "Model",
                "MeanCV_PR_AUC",
                "MeanCV_ROC_AUC",
                "MeanCV_Recall",
                "MeanCV_Precision",
                "MeanCV_FalseNegatives",
                "MeanCV_FalsePositives",
            ]
        ].to_string(index=False)
    )

    print(f"\nOutput: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
