"""Controlled strength-model comparison using the existing mix preprocessing.

Purpose
-------
Compare whether adding actual field slump and air improves 28-day strength
prediction while holding every other important condition constant:

* Same prepared input created by 04_preprocess_mix_optimization_data.py
* Same rows for both feature sets
* Same projectId grouped train/test split
* Same models and hyperparameters
* Same target and metrics

No command-line arguments are used. Edit paths in mix_config.py, then run:
    python 05b_compare_batch_vs_field_adjusted_project_group.py
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

import mix_config as cfg


FIELD_FEATURES = [
    "EffectiveSlump_in",
    "EffectiveAir_percent",
]

OUTPUT_DIR = cfg.MODEL_OUTPUT_DIR / "batch_vs_field_adjusted_project_group"


def candidate_models() -> dict[str, object]:
    """Use the same model settings as 05_train_mix_surrogate_models.py."""
    return {
        "DummyMedian": DummyRegressor(strategy="median"),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=cfg.RANDOM_STATE,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            max_features=0.8,
            n_jobs=cfg.N_JOBS,
            random_state=cfg.RANDOM_STATE,
        ),
    }


def grouped_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    groups = df[cfg.GROUP_COLUMN].astype(str)
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE,
    )
    train_index, test_index = next(splitter.split(df, groups=groups))

    train_groups = set(groups.iloc[train_index])
    test_groups = set(groups.iloc[test_index])
    overlap = train_groups.intersection(test_groups)
    if overlap:
        raise RuntimeError(
            f"Project group leakage detected: {len(overlap)} overlapping projects."
        )
    return train_index, test_index


def calculate_metrics(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    residual = prediction - y_true.to_numpy()
    absolute_error = np.abs(residual)
    return {
        "MAE_psi": float(mean_absolute_error(y_true, prediction)),
        "MedianAE_psi": float(median_absolute_error(y_true, prediction)),
        "RMSE_psi": float(mean_squared_error(y_true, prediction) ** 0.5),
        "R2": float(r2_score(y_true, prediction)),
        "MeanBias_psi": float(np.mean(residual)),
        "P90AbsoluteError_psi": float(np.percentile(absolute_error, 90)),
    }


def main() -> None:
    df = pd.read_csv(cfg.PREPARED_DATA_PATH, low_memory=False)

    batch_features = [column for column in cfg.MODEL_FEATURES if column in df.columns]
    required_columns = [
        "testId",
        cfg.GROUP_COLUMN,
        cfg.STRENGTH_TARGET,
        *FIELD_FEATURES,
        *cfg.CORE_BATCH_FEATURES,
    ]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError("Prepared data is missing required columns: " + ", ".join(missing))

    # Both feature sets must use exactly the same rows. Slump and air are not
    # imputed for cohort selection; rows without either field value are removed
    # from BOTH comparisons so row composition cannot explain the R2 difference.
    comparison_df = df[
        df[cfg.STRENGTH_TARGET].notna()
        & df[cfg.GROUP_COLUMN].notna()
        & df[FIELD_FEATURES].notna().all(axis=1)
        & df[cfg.CORE_BATCH_FEATURES].notna().all(axis=1)
    ].copy()

    if len(comparison_df) < 200:
        raise ValueError(
            f"Only {len(comparison_df)} complete comparison rows remain. "
            "At least 200 are required for this diagnostic."
        )

    feature_sets = {
        "BatchOnly": batch_features,
        "BatchPlusActualSlumpAir": batch_features + FIELD_FEATURES,
    }

    train_index, test_index = grouped_split(comparison_df)
    train_df = comparison_df.iloc[train_index].copy()
    test_df = comparison_df.iloc[test_index].copy()
    y_train = train_df[cfg.STRENGTH_TARGET]
    y_test = test_df[cfg.STRENGTH_TARGET]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = OUTPUT_DIR / "saved_models"
    prediction_dir = OUTPUT_DIR / "test_predictions"
    model_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    result_rows: list[dict[str, object]] = []
    for feature_set_name, features in feature_sets.items():
        x_train = train_df[features].apply(pd.to_numeric, errors="coerce")
        x_test = test_df[features].apply(pd.to_numeric, errors="coerce")

        for model_name, estimator in candidate_models().items():
            pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("model", estimator),
                ]
            )
            pipeline.fit(x_train, y_train)
            prediction = pipeline.predict(x_test)
            metrics = calculate_metrics(y_test, prediction)
            result_rows.append(
                {
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "group_column": cfg.GROUP_COLUMN,
                    "train_rows": len(train_df),
                    "test_rows": len(test_df),
                    "train_projects": int(train_df[cfg.GROUP_COLUMN].nunique()),
                    "test_projects": int(test_df[cfg.GROUP_COLUMN].nunique()),
                    "feature_count": len(features),
                    **metrics,
                }
            )

            artifact = {
                "pipeline": pipeline,
                "features": features,
                "feature_set": feature_set_name,
                "target": cfg.STRENGTH_TARGET,
                "group_column": cfg.GROUP_COLUMN,
            }
            joblib.dump(
                artifact,
                model_dir / f"{feature_set_name}__{model_name}.joblib",
            )

            pd.DataFrame(
                {
                    "testId": test_df["testId"].to_numpy(),
                    "projectId": test_df[cfg.GROUP_COLUMN].to_numpy(),
                    "actual_strength_28_psi": y_test.to_numpy(),
                    "predicted_strength_28_psi": prediction,
                    "residual_psi": prediction - y_test.to_numpy(),
                    "absolute_error_psi": np.abs(prediction - y_test.to_numpy()),
                }
            ).to_csv(
                prediction_dir / f"{feature_set_name}__{model_name}.csv",
                index=False,
            )

    results = pd.DataFrame(result_rows).sort_values(
        ["feature_set", "MAE_psi", "RMSE_psi"]
    )
    results.to_csv(OUTPUT_DIR / "batch_vs_field_adjusted_results.csv", index=False)

    # Direct improvement table: same model, same rows, same project split.
    improvement_rows = []
    for model_name in results["model"].unique():
        batch = results[
            (results["model"] == model_name)
            & (results["feature_set"] == "BatchOnly")
        ].iloc[0]
        field = results[
            (results["model"] == model_name)
            & (results["feature_set"] == "BatchPlusActualSlumpAir")
        ].iloc[0]
        improvement_rows.append(
            {
                "model": model_name,
                "BatchOnly_R2": batch["R2"],
                "FieldAdjusted_R2": field["R2"],
                "R2_change": field["R2"] - batch["R2"],
                "BatchOnly_MAE_psi": batch["MAE_psi"],
                "FieldAdjusted_MAE_psi": field["MAE_psi"],
                "MAE_improvement_psi": batch["MAE_psi"] - field["MAE_psi"],
                "MAE_improvement_percent": (
                    100.0 * (batch["MAE_psi"] - field["MAE_psi"]) / batch["MAE_psi"]
                    if batch["MAE_psi"] else np.nan
                ),
            }
        )
    improvement = pd.DataFrame(improvement_rows).sort_values(
        "FieldAdjusted_MAE_psi"
    )
    improvement.to_csv(OUTPUT_DIR / "field_feature_improvement.csv", index=False)

    summary = {
        "prepared_input": str(cfg.PREPARED_DATA_PATH),
        "comparison_rows": len(comparison_df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "group_column": cfg.GROUP_COLUMN,
        "project_overlap": 0,
        "same_rows_used_for_both_feature_sets": True,
        "batch_features": batch_features,
        "field_features_added": FIELD_FEATURES,
        "note": (
            "Actual slump and air are valid for post-field predictive quality "
            "analysis, but are not available when designing a new mix."
        ),
    }
    with (OUTPUT_DIR / "comparison_run_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    print("Controlled Batch vs Field-Adjusted comparison completed.")
    print("\nModel results")
    print(results.to_string(index=False))
    print("\nField-feature improvement")
    print(improvement.to_string(index=False))
    print(f"\nOutputs: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
