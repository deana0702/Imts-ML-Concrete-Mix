"""Train and compare 28-day concrete-strength regression models.

Edit all paths and options in config.py, then run:
    python 02_train_regression_models.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

import config


REQUIRED_CONFIG_NAMES = [
    "REGRESSION_INPUT_PATH",
    "REGRESSION_OUTPUT_DIR",
    "TEST_SIZE",
    "RANDOM_STATE",
    "N_JOBS",
    "TRAIN_FEATURE_SET_NAMES",
    "FEATURE_SETS",
    "GROUP_COLUMN",
    "REGRESSION_TARGET",
]


def validate_config() -> None:
    missing = [name for name in REQUIRED_CONFIG_NAMES if not hasattr(config, name)]
    if missing:
        raise RuntimeError(
            "Your config.py is older than this regression script. Missing: "
            + ", ".join(missing)
            + ". Replace config.py with the version delivered with this script."
        )
    if not 0 < config.TEST_SIZE < 1:
        raise ValueError("TEST_SIZE must be between 0 and 1.")
    unknown = [n for n in config.TRAIN_FEATURE_SET_NAMES if n not in config.FEATURE_SETS]
    if unknown:
        raise ValueError("Unknown TRAIN_FEATURE_SET_NAMES: " + ", ".join(unknown))


def read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists() and path.suffix.lower() == ".csv":
        parquet_path = path.with_suffix(".parquet")
        if parquet_path.exists():
            path = parquet_path
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("REGRESSION_INPUT_PATH must be a CSV or Parquet file.")


def grouped_train_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.GROUP_COLUMN not in df.columns:
        raise ValueError(f"Missing split group column: {config.GROUP_COLUMN}")
    groups = df[config.GROUP_COLUMN].fillna(df["testId"].astype(str))
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    train_index, test_index = next(splitter.split(df, groups=groups))
    return df.iloc[train_index].copy(), df.iloc[test_index].copy()


def available_features(df: pd.DataFrame, feature_set_name: str) -> list[str]:
    requested = config.FEATURE_SETS[feature_set_name]
    features = [name for name in requested if name in df.columns]
    if not features:
        raise ValueError(f"No usable columns found for {feature_set_name}.")
    return features


def model_candidates() -> dict[str, object]:
    return {
        "DummyMedian": DummyRegressor(strategy="median"),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=config.RANDOM_STATE,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            max_features=0.8,
            n_jobs=config.N_JOBS,
            random_state=config.RANDOM_STATE,
        ),
    }


def build_pipeline(model: object) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", model),
        ]
    )


def metrics(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - y_true.to_numpy()
    return {
        "MAE_psi": mean_absolute_error(y_true, prediction),
        "RMSE_psi": mean_squared_error(y_true, prediction) ** 0.5,
        "R2": r2_score(y_true, prediction),
        "MeanBias_psi": float(np.mean(error)),
        "P90AbsoluteError_psi": float(np.percentile(np.abs(error), 90)),
    }


def main() -> None:
    validate_config()
    output_dir = config.REGRESSION_OUTPUT_DIR
    model_dir = output_dir / "saved_models"
    prediction_dir = output_dir / "test_predictions"
    model_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    df = read_dataset(config.REGRESSION_INPUT_PATH)
    required = [config.REGRESSION_TARGET, config.GROUP_COLUMN, "testId"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    df = df[df[config.REGRESSION_TARGET].notna()].copy()
    train_df, test_df = grouped_train_test_split(df)

    result_rows: list[dict[str, object]] = []
    for feature_set_name in config.TRAIN_FEATURE_SET_NAMES:
        features = available_features(df, feature_set_name)
        feature_train_df = train_df
        feature_test_df = test_df
        if feature_set_name.startswith("Day7"):
            feature_train_df = train_df[
                train_df["AverageActualStrength7_psi"].notna()
            ].copy()
            feature_test_df = test_df[
                test_df["AverageActualStrength7_psi"].notna()
            ].copy()
        x_train = feature_train_df[features].apply(pd.to_numeric, errors="coerce")
        x_test = feature_test_df[features].apply(pd.to_numeric, errors="coerce")
        y_train = feature_train_df[config.REGRESSION_TARGET]
        y_test = feature_test_df[config.REGRESSION_TARGET]

        for model_name, model in model_candidates().items():
            pipeline = build_pipeline(model)
            pipeline.fit(x_train, y_train)
            prediction = pipeline.predict(x_test)
            row = {
                "feature_set": feature_set_name,
                "model": model_name,
                "train_rows": len(feature_train_df),
                "test_rows": len(feature_test_df),
                "feature_count": len(features),
                **metrics(y_test, prediction),
            }
            result_rows.append(row)

            artifact = {
                "pipeline": pipeline,
                "features": features,
                "feature_set": feature_set_name,
                "target": config.REGRESSION_TARGET,
            }
            joblib.dump(artifact, model_dir / f"{feature_set_name}__{model_name}.joblib")
            pd.DataFrame(
                {
                    "testId": feature_test_df["testId"].to_numpy(),
                    "projectId": feature_test_df[config.GROUP_COLUMN].to_numpy(),
                    "actual_strength_28_psi": y_test.to_numpy(),
                    "predicted_strength_28_psi": prediction,
                    "error_psi": prediction - y_test.to_numpy(),
                }
            ).to_csv(
                prediction_dir / f"{feature_set_name}__{model_name}.csv", index=False
            )

    results = pd.DataFrame(result_rows).sort_values(["MAE_psi", "RMSE_psi"])
    results.to_csv(output_dir / "regression_model_comparison.csv", index=False)
    summary = {
        "input_rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_project_count": int(train_df[config.GROUP_COLUMN].nunique()),
        "test_project_count": int(test_df[config.GROUP_COLUMN].nunique()),
        "best_model_by_mae": results.iloc[0].to_dict(),
    }
    with (output_dir / "regression_run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("Regression training completed.")
    print(results.to_string(index=False))
    print(f"Outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
