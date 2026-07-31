from __future__ import annotations

from pathlib import Path
import json
import math
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Run with:
#     python 03_train_holdout_models.py
ROOT_DIR = Path(__file__).resolve().parent.parent
TRAIN_FILE = ROOT_DIR / "data/field_core_outputs/field_core_splits" / "comparison_train.csv"
TEST_FILE = ROOT_DIR / "data/field_core_outputs/field_core_splits" / "comparison_test.csv"
OUTPUT_DIR = ROOT_DIR / "data/field_core_outputs/field_core_models"

TARGET = "AverageActualStrength28_psi"
RANDOM_STATE = 42

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
            f"Input file not found: {path}\nRun 02_create_grouped_splits.py first."
        )
    return pd.read_csv(path, low_memory=False)


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


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
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
    MODEL_DIR = OUTPUT_DIR / "saved_models"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 3 started: train holdout models")
    train = read_csv(TRAIN_FILE)
    test = read_csv(TEST_FILE)

    y_train = pd.to_numeric(train[TARGET], errors="coerce")
    y_test = pd.to_numeric(test[TARGET], errors="coerce")
    if y_train.isna().any() or y_test.isna().any():
        raise ValueError("The clean split contains missing target values.")

    feature_sets = {
        "FieldOnly": FIELD_ONLY_FEATURES,
        "FieldPlusRequired": FIELD_PLUS_REQUIRED_FEATURES,
    }

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    fitted_models: dict[tuple[str, str], object] = {}

    for feature_set_name, features in feature_sets.items():
        x_train = numeric_frame(train, features)
        x_test = numeric_frame(test, features)

        for model_name, model in build_models().items():
            print(f"Training {feature_set_name} / {model_name} ...")
            start = time.perf_counter()
            model.fit(x_train, y_train)
            predicted = model.predict(x_test)
            elapsed = time.perf_counter() - start

            metrics = calculate_metrics(y_test.to_numpy(), predicted)
            metric_rows.append(
                {
                    "FeatureSet": feature_set_name,
                    "Model": model_name,
                    "TrainRows": len(train),
                    "TestRows": len(test),
                    "FeatureCount": len(features),
                    "TrainingSeconds": elapsed,
                    **metrics,
                }
            )

            prediction_data = {
                "FeatureSet": feature_set_name,
                "Model": model_name,
                "ActualStrength28_psi": y_test.to_numpy(),
                "PredictedStrength28_psi": predicted,
                "ResidualPsi": predicted - y_test.to_numpy(),
                "AbsoluteErrorPsi": np.abs(predicted - y_test.to_numpy()),
            }
            for identifier in ["testId", "projectId", "projectNo", "OfficeName"]:
                if identifier in test.columns:
                    prediction_data[identifier] = test[identifier].to_numpy()
            prediction_frames.append(pd.DataFrame(prediction_data))

            model_path = MODEL_DIR / f"{feature_set_name}__{model_name}.joblib"
            joblib.dump(
                {
                    "model": model,
                    "features": features,
                    "target": TARGET,
                    "feature_set": feature_set_name,
                    "model_name": model_name,
                },
                model_path,
            )
            fitted_models[(feature_set_name, model_name)] = model_path

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        ["MAE", "RMSE"],
        ascending=[True, True],
    )
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    metrics_df.to_csv(OUTPUT_DIR / "holdout_model_metrics.csv", index=False)
    predictions_df.to_csv(OUTPUT_DIR / "holdout_predictions.csv", index=False)

    best_row = metrics_df.iloc[0]
    best_key = (str(best_row["FeatureSet"]), str(best_row["Model"]))
    best_source = fitted_models[best_key]
    best_destination = OUTPUT_DIR / "best_holdout_model.joblib"
    best_destination.write_bytes(best_source.read_bytes())

    configuration = {
        "target": TARGET,
        "random_state": RANDOM_STATE,
        "train_file": str(TRAIN_FILE),
        "test_file": str(TEST_FILE),
        "feature_sets": feature_sets,
        "best_feature_set": best_key[0],
        "best_model": best_key[1],
        "best_mae": float(best_row["MAE"]),
        "best_rmse": float(best_row["RMSE"]),
        "best_r2": float(best_row["R2"]),
    }
    (OUTPUT_DIR / "model_run_configuration.json").write_text(
        json.dumps(configuration, indent=2),
        encoding="utf-8",
    )

    print("Step 3 completed.")
    print(metrics_df[["FeatureSet", "Model", "MAE", "RMSE", "R2"]].to_string(index=False))
    print(f"Best holdout model: {best_key[0]} / {best_key[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
