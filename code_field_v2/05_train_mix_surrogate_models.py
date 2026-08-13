"""Train batch-to-strength, batch-to-slump, and batch-to-air surrogate models.

Edit mix_config.py, then run: python 05_train_mix_surrogate_models.py
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

import mix_config as cfg


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = df[cfg.GROUP_COLUMN].fillna(df["testId"].astype(str))
    splitter = GroupShuffleSplit(1, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE)
    a, b = next(splitter.split(df, groups=groups))
    return df.iloc[a].copy(), df.iloc[b].copy()


def candidates() -> dict[str, object]:
    return {
        "DummyMedian": DummyRegressor(strategy="median"),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=cfg.RANDOM_STATE
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=3, max_features=0.8,
            n_jobs=cfg.N_JOBS, random_state=cfg.RANDOM_STATE
        ),
    }


def main() -> None:
    df = pd.read_csv(cfg.PREPARED_DATA_PATH, low_memory=False)
    features = [c for c in cfg.MODEL_FEATURES if c in df.columns]
    missing_core = [c for c in cfg.CORE_BATCH_FEATURES if c not in features]
    if missing_core:
        raise ValueError("Prepared data is missing: " + ", ".join(missing_core))
    train_df, test_df = split(df)
    cfg.MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    targets = {
        "strength28": cfg.STRENGTH_TARGET,
        "slump": cfg.SLUMP_TARGET,
        "air": cfg.AIR_TARGET,
    }

    for target_key, target in targets.items():
        if target not in df.columns:
            continue
        target_train = train_df[train_df[target].notna()].copy()
        target_test = test_df[test_df[target].notna()].copy()
        if len(target_train) < 100 or len(target_test) < 30:
            print(f"Skipping {target}: insufficient non-null rows.")
            continue
        x_train = target_train[features].apply(pd.to_numeric, errors="coerce")
        x_test = target_test[features].apply(pd.to_numeric, errors="coerce")
        y_train, y_test = target_train[target], target_test[target]
        best = None
        for model_name, model in candidates().items():
            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("model", model),
            ])
            pipeline.fit(x_train, y_train)
            pred = pipeline.predict(x_test)
            absolute_error = np.abs(pred - y_test.to_numpy())
            row = {
                "target_key": target_key, "target": target, "model": model_name,
                "train_rows": len(target_train), "test_rows": len(target_test),
                "feature_count": len(features),
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": mean_squared_error(y_test, pred) ** 0.5,
                "R2": r2_score(y_test, pred),
                "P90AbsoluteError": float(np.percentile(absolute_error, 90)),
            }
            results.append(row)
            if model_name != "DummyMedian" and (best is None or row["MAE"] < best[0]):
                best = (row["MAE"], pipeline, row)
        _, best_pipeline, best_row = best
        artifact = {
            "pipeline": best_pipeline,
            "features": features,
            "target": target,
            "model_name": best_row["model"],
            "p90_absolute_error": best_row["P90AbsoluteError"],
            "test_r2": best_row["R2"],
            "test_mae": best_row["MAE"],
        }
        joblib.dump(artifact, cfg.MODEL_OUTPUT_DIR / f"best_{target_key}_model.joblib")

    result_df = pd.DataFrame(results).sort_values(["target_key", "MAE"])
    result_df.to_csv(cfg.MODEL_OUTPUT_DIR / "mix_surrogate_model_comparison.csv", index=False)
    with (cfg.MODEL_OUTPUT_DIR / "mix_model_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"features": features, "results": results}, f, indent=2, default=str)
    print("Mix surrogate training completed.")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
