from __future__ import annotations

import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Run with:
#     python code_field/test.py
ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = (
    ROOT_DIR
    / "data"
    / "field_core_outputs"
    / "field_core_clean"
    / "field_core_clean_with_required.csv"
)
OUTPUT_DIR = ROOT_DIR / "data" / "field_core_outputs" / "test_field_features_only"
MODEL_DIR = OUTPUT_DIR / "saved_models"

TARGET = "AverageActualStrength28_psi"
RANDOM_STATE = 42
TEST_SIZE = 0.20
GROUP_COLUMN = "projectId"

# Only these numeric features are used for ML training and prediction.
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
    "ApplicableSpecifiedStrength28",
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\\n"
            "Run code_field/01_clean_field_core_dataset.py first."
        )
    return pd.read_csv(path, low_memory=False)


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if TARGET not in df.columns:
        raise KeyError(f"Target column is missing: {TARGET}")
    if GROUP_COLUMN not in df.columns:
        raise KeyError(f"Grouping column is missing: {GROUP_COLUMN}")

    groups = df[GROUP_COLUMN].astype("string").str.strip()
    if groups.isna().any() or groups.eq("").any():
        raise ValueError(
            f"Grouping column '{GROUP_COLUMN}' contains missing values. "
            "Please clean projectId before splitting."
        )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_positions, test_positions = next(
        splitter.split(df, y=df[TARGET], groups=groups)
    )
    train = df.iloc[train_positions].copy()
    test = df.iloc[test_positions].copy()
    return train, test


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    series = df[column]
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(
        series.astype("string").str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Required model columns are missing: {missing}")
    result = pd.DataFrame(index=df.index)
    for column in columns:
        result[column] = numeric_series(df, column)
    return result


def build_regression_models() -> dict[str, object]:
    return {
        "DummyMean": DummyRegressor(strategy="mean"),
        "Ridge": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
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
    }


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
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


def save_json(data: dict[str, object], path: Path) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Test experiment started: FIELD_FEATURES-only holdout modeling")
    print("Step 1/6: Load with-required dataset")
    all_rows = read_csv(INPUT_FILE)
    print(f"  Input rows: {len(all_rows):,}")

    print("Step 2/6: Split train/test by projectId with grouped shuffle")
    train, test = split_train_test(all_rows)
    print(f"  Train rows: {len(train):,}; Test rows: {len(test):,}")
    print(f"  Group column: {GROUP_COLUMN}")

    print("Step 3/6: Convert target to numeric")
    y_train = numeric_series(train, TARGET)
    y_test = numeric_series(test, TARGET)
    if y_train.isna().any() or y_test.isna().any():
        raise ValueError("The clean split contains missing target values.")
    print(
        f"  Target non-null: train={int(y_train.notna().sum()):,}, "
        f"test={int(y_test.notna().sum()):,}"
    )

    print("Step 4/6: Build numeric matrices from FIELD_FEATURES only")
    x_train = numeric_frame(train, FIELD_FEATURES)
    x_test = numeric_frame(test, FIELD_FEATURES)
    print(f"  Feature count: {x_train.shape[1]}")

    print("Step 5/6: Train models and score holdout predictions")
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    model_paths: dict[str, Path] = {}

    for model_name, model in build_regression_models().items():
        print(f"  Training FieldFeaturesOnly / {model_name} ...")
        start = time.perf_counter()
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        elapsed = time.perf_counter() - start

        metric_rows.append(
            {
                "FeatureSet": "FieldFeaturesOnly",
                "Model": model_name,
                "TrainRows": len(train),
                "TestRows": len(test),
                "FeatureCount": x_train.shape[1],
                "TrainingSeconds": elapsed,
                **regression_metrics(y_test.to_numpy(), predicted),
            }
        )

        prediction_frame = pd.DataFrame(
            {
                "FeatureSet": "FieldFeaturesOnly",
                "Model": model_name,
                "ActualStrength28_psi": y_test.to_numpy(),
                "PredictedStrength28_psi": predicted,
                "ResidualPsi": predicted - y_test.to_numpy(),
                "AbsoluteErrorPsi": np.abs(predicted - y_test.to_numpy()),
            }
        )
        for identifier in ["testId", "projectId", "projectNo", "OfficeName"]:
            if identifier in test.columns:
                prediction_frame[identifier] = test[identifier].to_numpy()
        prediction_frames.append(prediction_frame)

        model_path = MODEL_DIR / f"FieldFeaturesOnly__{model_name}.joblib"
        joblib.dump(
            {
                "model": model,
                "numeric_features": list(x_train.columns),
                "target": TARGET,
                "feature_set": "FieldFeaturesOnly",
                "model_name": model_name,
            },
            model_path,
        )
        model_paths[model_name] = model_path

    print("Step 6/6: Save artifacts and summary")
    metrics = pd.DataFrame(metric_rows).sort_values(["MAE", "RMSE"])
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics.to_csv(OUTPUT_DIR / "field_features_only_model_metrics.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "field_features_only_predictions.csv", index=False)

    best_row = metrics.iloc[0]
    best_model_name = str(best_row["Model"])
    best_source = model_paths[best_model_name]
    best_destination = OUTPUT_DIR / "best_field_features_only_model.joblib"
    best_destination.write_bytes(best_source.read_bytes())

    save_json(
        {
            "input_file": str(INPUT_FILE),
            "target": TARGET,
            "feature_set": "FieldFeaturesOnly",
            "feature_columns": FIELD_FEATURES,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "group_column": GROUP_COLUMN,
            "model_count": len(metric_rows),
            "train_rows": len(train),
            "test_rows": len(test),
            "best_model": best_model_name,
            "best_mae": float(best_row["MAE"]),
            "best_rmse": float(best_row["RMSE"]),
            "best_r2": float(best_row["R2"]),
        },
        OUTPUT_DIR / "field_features_only_summary.json",
    )

    print("Test experiment completed.")
    print(metrics[["FeatureSet", "Model", "MAE", "RMSE", "R2"]].to_string(index=False))
    print(f"Best model: {best_model_name}")
    print(f"Output directory: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())