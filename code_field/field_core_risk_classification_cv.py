from __future__ import annotations

"""
IMTS Field Core - Pass/Fail Risk Classification with 5-Fold Project-Grouped CV

Feature sets:
1) Day0_FieldPlusRequired
2) Day7_FieldPlusRequired
3) Full_ContextPlusDay7

Binary target:
    FailureFlag28 = 1 if AverageActualStrength28_psi < ApplicableSpecifiedStrength28
    else 0

Notes:
- This is an analytical failure proxy, not the complete engineering acceptance standard.
- Outer CV is grouped by project.
- Supplier/Plant/Mix failure-rate target encoding is rebuilt inside each outer fold.
- XGBoost is included.
- Rank models primarily by PR-AUC because failures are rare.

Run:
    python code_field/field_core_risk_classification_cv.py

Dependency:
    pip install xgboost
"""

from dataclasses import dataclass
from pathlib import Path
import math
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
    raise ImportError("Install xgboost first: pip install xgboost") from exc


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

OUTPUT_DIR = FIELD_CORE_OUTPUT_ROOT / "consolidated_risk_classification_cv"

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
DAY0_FEATURES = FIELD_FEATURES + [REQUIRED_STRENGTH]

DAY7_AVERAGE_CANDIDATES = ["AverageActualStrength7_psi", "AverageActualStrength7"]
DAY7_COUNT_CANDIDATES = ["ActualStrength7SpecimenCount", "StandardCuredStrength7SpecimenCount"]
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


def first_existing_path(paths: Iterable[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find cleaned Field Core dataset.")


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            pass
    raise UnicodeError(f"Could not determine CSV encoding: {path}")


def resolve_first_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    s = df[column]
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(s.astype("string").str.replace(",", "", regex=False).str.strip(), errors="coerce")


def numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    require_columns(df, columns)
    return pd.DataFrame({c: numeric_series(df, c) for c in columns}, index=df.index)


def resolve_group_column(df: pd.DataFrame) -> str:
    col = resolve_first_column(df, GROUP_COLUMN_CANDIDATES)
    if col is None:
        raise KeyError(f"Expected one of group columns: {GROUP_COLUMN_CANDIDATES}")
    return col


def make_project_groups(df: pd.DataFrame, group_column: str) -> pd.Series:
    groups = df[group_column].astype("string").str.strip()
    missing = groups.isna() | groups.eq("")
    fallback = (
        "MISSING_PROJECT_TEST_" + df["testId"].astype("string")
        if "testId" in df.columns
        else "MISSING_PROJECT_ROW_" + pd.Series(df.index, index=df.index).astype("string")
    )
    return groups.mask(missing, fallback)


def add_day7_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    avg_col = resolve_first_column(result, DAY7_AVERAGE_CANDIDATES)
    count_col = resolve_first_column(result, DAY7_COUNT_CANDIDATES)
    if avg_col is None:
        raise KeyError(f"Missing 7-day strength column: {DAY7_AVERAGE_CANDIDATES}")

    result["Day7AverageStrength_psi"] = numeric_series(result, avg_col)
    result["Day7SpecimenCount"] = numeric_series(result, count_col) if count_col else np.nan
    required = numeric_series(result, REQUIRED_STRENGTH)
    result["Day7MarginToRequired_psi"] = result["Day7AverageStrength_psi"] - required
    result["Day7ToRequiredRatio"] = result["Day7AverageStrength_psi"] / required.replace(0, np.nan)
    return result


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
    resolved = {}
    for logical, candidates in CONTEXT_COLUMN_CANDIDATES.items():
        col = resolve_first_column(df, candidates)
        if col is None:
            raise KeyError(f"Missing context field {logical}; expected {candidates}")
        resolved[logical] = col
    return ContextSources(resolved["Supplier"], resolved["Plant"], resolved["Mix"])


def build_context_categories(df: pd.DataFrame, sources: ContextSources) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["SupplierCategory"] = normalize_category(df[sources.supplier])
    out["PlantCategory"] = normalize_category(df[sources.plant])
    out["MixCategory"] = normalize_category(df[sources.mix])
    out["SupplierPlantCategory"] = out["SupplierCategory"] + "|" + out["PlantCategory"]
    out["SupplierPlantMixCategory"] = (
        out["SupplierCategory"] + "|" + out["PlantCategory"] + "|" + out["MixCategory"]
    )
    return out


def smoothed_target_mapping(category: pd.Series, target: pd.Series, global_mean: float, smoothing: float) -> pd.Series:
    stats = pd.DataFrame({"category": category, "target": target}).groupby("category")["target"].agg(["sum", "count"])
    return (stats["sum"] + smoothing * global_mean) / (stats["count"] + smoothing)


def cross_fitted_failure_target_encode(
    train_categories: pd.DataFrame,
    validation_categories: pd.DataFrame,
    y_train: pd.Series,
    train_groups: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = train_groups.astype("string").fillna("__MISSING_GROUP__")
    n_splits = min(TARGET_ENCODING_FOLDS, int(groups.nunique()))
    inner = GroupKFold(n_splits=n_splits)
    global_rate = float(y_train.mean())

    encoded_train = pd.DataFrame(index=train_categories.index)
    encoded_valid = pd.DataFrame(index=validation_categories.index)

    for column in train_categories.columns:
        train_values = normalize_category(train_categories[column])
        valid_values = normalize_category(validation_categories[column])
        oof = pd.Series(np.nan, index=train_categories.index, dtype=float)

        for fit_pos, enc_pos in inner.split(train_categories, y_train, groups):
            fit_idx = train_categories.index[fit_pos]
            enc_idx = train_categories.index[enc_pos]
            mapping = smoothed_target_mapping(
                train_values.loc[fit_idx], y_train.loc[fit_idx], global_rate, TARGET_ENCODING_SMOOTHING
            )
            oof.loc[enc_idx] = train_values.loc[enc_idx].map(mapping).fillna(global_rate)

        full_mapping = smoothed_target_mapping(
            train_values, y_train, global_rate, TARGET_ENCODING_SMOOTHING
        )
        counts = train_values.value_counts()

        encoded_train[f"{column}_FailureRate"] = oof.fillna(global_rate)
        encoded_valid[f"{column}_FailureRate"] = valid_values.map(full_mapping).fillna(global_rate)
        encoded_train[f"{column}_LogCount"] = np.log1p(train_values.map(counts).fillna(0).astype(float))
        encoded_valid[f"{column}_LogCount"] = np.log1p(valid_values.map(counts).fillna(0).astype(float))
        encoded_train[f"{column}_Unknown"] = 0
        encoded_valid[f"{column}_Unknown"] = (~valid_values.isin(full_mapping.index)).astype(int)

    return encoded_train, encoded_valid


def build_classification_models() -> dict[str, object]:
    return {
        "DummyPrior": DummyClassifier(strategy="prior"),
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=5,
                max_features=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=RANDOM_STATE,
            )),
        ]),
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


def fit_balanced(model: object, x: pd.DataFrame, y: pd.Series) -> object:
    weights = compute_sample_weight(class_weight="balanced", y=y)
    if isinstance(model, Pipeline):
        model.fit(x, y, model__sample_weight=weights)
    else:
        model.fit(x, y, sample_weight=weights)
    return model


def metrics(actual: np.ndarray, probability: np.ndarray, threshold: float = REPORT_THRESHOLD) -> dict[str, float | int]:
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    return {
        "PR_AUC": float(average_precision_score(actual, probability)),
        "ROC_AUC": float(roc_auc_score(actual, probability)),
        "BrierScore": float(brier_score_loss(actual, probability)),
        "Recall": float(recall_score(actual, predicted, zero_division=0)),
        "Precision": float(precision_score(actual, predicted, zero_division=0)),
        "F1": float(f1_score(actual, predicted, zero_division=0)),
        "FalseNegatives": int(fn),
        "FalsePositives": int(fp),
        "TruePositives": int(tp),
        "TrueNegatives": int(tn),
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_file = first_existing_path(INPUT_CANDIDATES)
    df = read_csv(input_file)
    require_columns(df, [TARGET_STRENGTH, REQUIRED_STRENGTH, *FIELD_FEATURES])

    actual28 = numeric_series(df, TARGET_STRENGTH)
    required28 = numeric_series(df, REQUIRED_STRENGTH)
    df = df.loc[actual28.gt(0) & required28.gt(0)].copy()

    df = add_day7_features(df)
    df = df.loc[numeric_series(df, "Day7AverageStrength_psi").gt(0)].copy()

    df[RISK_TARGET] = (
        numeric_series(df, TARGET_STRENGTH) < numeric_series(df, REQUIRED_STRENGTH)
    ).astype(int)

    print(f"Rows used: {len(df):,}")
    print(f"Failures: {int(df[RISK_TARGET].sum()):,}")
    print(f"Failure rate: {df[RISK_TARGET].mean():.2%}")

    group_column = resolve_group_column(df)
    groups = make_project_groups(df, group_column)
    print(f"Grouping column: {group_column}")
    print(f"Unique project groups: {groups.nunique():,}")

    context_sources = resolve_context_sources(df)
    outer = GroupKFold(n_splits=OUTER_CV_FOLDS)

    fold_rows = []
    prediction_rows = []

    for fold, (train_pos, valid_pos) in enumerate(outer.split(df, groups=groups), start=1):
        train = df.iloc[train_pos].copy()
        valid = df.iloc[valid_pos].copy()

        y_train = train[RISK_TARGET].astype(int)
        y_valid = valid[RISK_TARGET].astype(int)
        train_groups = make_project_groups(train, group_column)

        day0_train = numeric_frame(train, DAY0_FEATURES)
        day0_valid = numeric_frame(valid, DAY0_FEATURES)
        day7_train = numeric_frame(train, DAY7_FEATURES)
        day7_valid = numeric_frame(valid, DAY7_FEATURES)

        day7_full_train = pd.concat([day0_train, day7_train], axis=1)
        day7_full_valid = pd.concat([day0_valid, day7_valid], axis=1)

        train_context = build_context_categories(train, context_sources)
        valid_context = build_context_categories(valid, context_sources)
        enc_train, enc_valid = cross_fitted_failure_target_encode(
            train_context, valid_context, y_train, train_groups
        )

        full_train = pd.concat([day0_train, enc_train, day7_train], axis=1)
        full_valid = pd.concat([day0_valid, enc_valid, day7_valid], axis=1)

        feature_sets = {
            "Day0_FieldPlusRequired": (day0_train, day0_valid),
            "Day7_FieldPlusRequired": (day7_full_train, day7_full_valid),
            "Full_ContextPlusDay7": (full_train, full_valid),
        }

        print(f"\nFold {fold}/{OUTER_CV_FOLDS} | train={len(train):,}, valid={len(valid):,}, valid_fail_rate={y_valid.mean():.2%}")

        for feature_set, (x_train, x_valid) in feature_sets.items():
            for model_name, model in build_classification_models().items():
                print(f"  {feature_set} | {model_name}")
                start = time.perf_counter()
                model = fit_balanced(model, x_train, y_train)
                prob = model.predict_proba(x_valid)[:, 1]
                elapsed = time.perf_counter() - start
                m = metrics(y_valid.to_numpy(), prob)

                fold_rows.append({
                    "Fold": fold,
                    "FeatureSet": feature_set,
                    "Model": model_name,
                    "ValidationRows": len(valid),
                    "ValidationFailureRate": float(y_valid.mean()),
                    "TrainingSeconds": elapsed,
                    **m,
                })

                pred = pd.DataFrame({
                    "Fold": fold,
                    "FeatureSet": feature_set,
                    "Model": model_name,
                    "ActualFailureFlag": y_valid.to_numpy(),
                    "FailureProbability": prob,
                })
                if "testId" in valid.columns:
                    pred["testId"] = valid["testId"].to_numpy()
                prediction_rows.append(pred)

    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)

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
            MeanCV_Brier=("BrierScore", "mean"),
            MeanCV_FalseNegatives=("FalseNegatives", "mean"),
            MeanCV_FalsePositives=("FalsePositives", "mean"),
        )
        .sort_values(["MeanCV_PR_AUC", "MeanCV_ROC_AUC"], ascending=[False, False])
        .reset_index(drop=True)
    )

    best = (
        summary.sort_values(["MeanCV_PR_AUC", "MeanCV_ROC_AUC"], ascending=[False, False])
        .groupby("FeatureSet", as_index=False)
        .first()
        .sort_values("MeanCV_PR_AUC", ascending=False)
        .reset_index(drop=True)
    )

    # OOF threshold analysis for non-dummy models.
    threshold_rows = []
    thresholds = np.round(np.arange(0.05, 0.951, 0.025), 3)
    for (feature_set, model_name), g in predictions.groupby(["FeatureSet", "Model"]):
        actual = g["ActualFailureFlag"].to_numpy()
        prob = g["FailureProbability"].to_numpy()
        for threshold in thresholds:
            m = metrics(actual, prob, float(threshold))
            threshold_rows.append({
                "FeatureSet": feature_set,
                "Model": model_name,
                "Threshold": threshold,
                **m,
            })
    threshold_table = pd.DataFrame(threshold_rows)

    fold_metrics.to_csv(OUTPUT_DIR / "classification_cv_fold_metrics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "classification_cv_summary.csv", index=False)
    best.to_csv(OUTPUT_DIR / "best_classifier_by_feature_set_cv.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "classification_oof_predictions.csv", index=False)
    threshold_table.to_csv(OUTPUT_DIR / "classification_oof_threshold_table.csv", index=False)

    print("\nBEST CLASSIFIER BY FEATURE SET")
    print("=" * 110)
    print(best[[
        "FeatureSet",
        "Model",
        "MeanCV_PR_AUC",
        "StdCV_PR_AUC",
        "MeanCV_ROC_AUC",
        "MeanCV_Recall",
        "MeanCV_Precision",
        "MeanCV_FalseNegatives",
        "MeanCV_Brier",
    ]].to_string(index=False))

    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("Important: 0.50 is only a comparison threshold. Use OOF threshold analysis before external validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
