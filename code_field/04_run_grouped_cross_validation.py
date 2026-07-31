from __future__ import annotations

from pathlib import Path
import math
import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline


# Run with:
#     python 04_run_grouped_cross_validation.py
ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = (
    ROOT_DIR / "data/field_core_outputs/field_core_clean/field_core_clean_with_required.csv"
)
OUTPUT_DIR = ROOT_DIR / "data/field_core_outputs/field_core_cv"

TARGET = "AverageActualStrength28_psi"
RANDOM_STATE = 42
REQUESTED_FOLDS = 5
GROUP_COLUMN_CANDIDATES = ["projectId", "projectNo"]

FIELD_ONLY_FEATURES = [
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
FIELD_PLUS_REQUIRED_FEATURES = FIELD_ONLY_FEATURES + [
    "ApplicableSpecifiedStrength28"
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\nRun 01_clean_field_core_dataset.py first."
        )
    return pd.read_csv(path, low_memory=False)


def resolve_group_column(df: pd.DataFrame) -> str:
    for column in GROUP_COLUMN_CANDIDATES:
        if column in df.columns and df[column].notna().any():
            return column
    raise KeyError(f"Expected one of the grouping columns: {GROUP_COLUMN_CANDIDATES}")


def make_groups(df: pd.DataFrame, group_column: str) -> pd.Series:
    groups = df[group_column].astype("string").str.strip()
    missing = groups.isna() | groups.eq("")
    fallback = "MISSING_PROJECT_TEST_" + df["testId"].astype("string")
    return groups.mask(missing, fallback)


def numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Required model columns are missing: {missing}")
    result = pd.DataFrame(index=df.index)
    for column in columns:
        result[column] = pd.to_numeric(df[column], errors="coerce")
    return result


def build_models() -> dict[str, object]:
    return {
        "DummyMean": DummyRegressor(strategy="mean"),
        "RandomForest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=200,
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
                    HistGradientBoostingRegressor(
                        max_iter=250,
                        learning_rate=0.05,
                        max_leaf_nodes=31,
                        min_samples_leaf=20,
                        l2_regularization=1.0,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - actual
    absolute_error = np.abs(residual)
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "MedianAE": float(median_absolute_error(actual, predicted)),
        "RMSE": float(math.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
        "MeanBias": float(np.mean(residual)),
        "Within300PsiPercent": float(np.mean(absolute_error <= 300.0) * 100.0),
        "Within500PsiPercent": float(np.mean(absolute_error <= 500.0) * 100.0),
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Step 4 started: grouped cross-validation")

    df = read_csv(INPUT_FILE)
    group_column = resolve_group_column(df)
    groups = make_groups(df, group_column)
    unique_group_count = groups.nunique()
    n_splits = min(REQUESTED_FOLDS, unique_group_count)
    if n_splits < 2:
        raise ValueError("At least two unique groups are required for grouped CV.")

    y = pd.to_numeric(df[TARGET], errors="coerce")
    if y.isna().any():
        raise ValueError("The CV dataset contains missing target values.")

    feature_sets = {
        "FieldOnly": FIELD_ONLY_FEATURES,
        "FieldPlusRequired": FIELD_PLUS_REQUIRED_FEATURES,
    }
    splitter = GroupKFold(n_splits=n_splits)
    fold_rows: list[dict[str, object]] = []

    for feature_set_name, features in feature_sets.items():
        x = numeric_frame(df, features)

        for model_name, model in build_models().items():
            print(f"Cross-validating {feature_set_name} / {model_name} ...")
            for fold_number, (train_idx, test_idx) in enumerate(
                splitter.split(x, y, groups=groups),
                start=1,
            ):
                start = time.perf_counter()
                model.fit(x.iloc[train_idx], y.iloc[train_idx])
                predicted = model.predict(x.iloc[test_idx])
                elapsed = time.perf_counter() - start
                fold_rows.append(
                    {
                        "FeatureSet": feature_set_name,
                        "Model": model_name,
                        "Fold": fold_number,
                        "TrainRows": len(train_idx),
                        "TestRows": len(test_idx),
                        "TrainGroups": groups.iloc[train_idx].nunique(),
                        "TestGroups": groups.iloc[test_idx].nunique(),
                        "TrainingSeconds": elapsed,
                        **metrics(y.iloc[test_idx].to_numpy(), predicted),
                    }
                )

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(OUTPUT_DIR / "grouped_cv_fold_metrics.csv", index=False)

    metric_columns = [
        "MAE",
        "MedianAE",
        "RMSE",
        "R2",
        "MeanBias",
        "Within300PsiPercent",
        "Within500PsiPercent",
    ]
    aggregations: dict[str, list[str]] = {
        metric: ["mean", "std", "min", "max"] for metric in metric_columns
    }
    summary = folds.groupby(["FeatureSet", "Model"], as_index=False).agg(aggregations)
    summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    summary = summary.rename(
        columns={"FeatureSet_": "FeatureSet", "Model_": "Model"}
    ).sort_values("MAE_mean")
    summary.to_csv(OUTPUT_DIR / "grouped_cv_summary.csv", index=False)

    print("Step 4 completed.")
    print(f"Grouping column: {group_column}")
    print(f"Folds: {n_splits}")
    print(summary[["FeatureSet", "Model", "MAE_mean", "MAE_std", "RMSE_mean", "R2_mean"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
