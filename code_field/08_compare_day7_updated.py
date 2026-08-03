from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd

from field_core_experiment_common import (
    DAY7_FEATURES,
    FIELD_PLUS_REQUIRED_FEATURES,
    NEXT_OUTPUT_ROOT,
    TARGET,
    add_day7_features,
    build_context_categories,
    build_regression_models,
    cross_fitted_target_encode,
    load_comparison_splits,
    numeric_frame,
    numeric_series,
    regression_metrics,
    resolve_context_sources,
    resolve_group_column,
    save_json,
)


# Run with:
#     python 08_compare_day7_updated.py
OUTPUT_DIR = NEXT_OUTPUT_ROOT / "08_day7_updated"
MODEL_DIR = OUTPUT_DIR / "saved_models"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Model experiment 08 started: 7-day updated models")
    train_raw, test_raw = load_comparison_splits()
    train, day7_metadata = add_day7_features(train_raw)
    test, _ = add_day7_features(test_raw)

    # Fair comparison: every feature set below uses the same rows with a valid 7-day result.
    train_mask = numeric_series(train, "Day7AverageStrength_psi").gt(0)
    test_mask = numeric_series(test, "Day7AverageStrength_psi").gt(0)
    train = train.loc[train_mask].copy()
    test = test.loc[test_mask].copy()

    y_train = numeric_series(train, TARGET)
    y_test = numeric_series(test, TARGET)

    base_train = numeric_frame(train, FIELD_PLUS_REQUIRED_FEATURES)
    base_test = numeric_frame(test, FIELD_PLUS_REQUIRED_FEATURES)
    day7_train = numeric_frame(train, DAY7_FEATURES)
    day7_test = numeric_frame(test, DAY7_FEATURES)

    context_sources = resolve_context_sources(train)
    train_context = build_context_categories(train, context_sources)
    test_context = build_context_categories(test, context_sources)
    group_column = resolve_group_column(train)
    encoded_train, encoded_test, context_metadata = cross_fitted_target_encode(
        train_context,
        test_context,
        y_train,
        train[group_column],
    )

    feature_sets = {
        "FieldPlusRequired_Day7CommonRows": (base_train, base_test),
        "FieldPlusRequiredPlusContext_Day7CommonRows": (
            pd.concat([base_train, encoded_train], axis=1),
            pd.concat([base_test, encoded_test], axis=1),
        ),
        "FieldPlusRequiredPlus7Day": (
            pd.concat([base_train, day7_train], axis=1),
            pd.concat([base_test, day7_test], axis=1),
        ),
        "FullUpdated_ContextPlus7Day": (
            pd.concat([base_train, encoded_train, day7_train], axis=1),
            pd.concat([base_test, encoded_test, day7_test], axis=1),
        ),
    }

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for feature_set_name, (x_train, x_test) in feature_sets.items():
        for model_name, model in build_regression_models().items():
            print(f"Training {feature_set_name} / {model_name} ...")
            start = time.perf_counter()
            model.fit(x_train, y_train)
            predicted = model.predict(x_test)
            elapsed = time.perf_counter() - start

            metric_rows.append(
                {
                    "FeatureSet": feature_set_name,
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
                    "FeatureSet": feature_set_name,
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

            joblib.dump(
                {
                    "model": model,
                    "numeric_features": list(x_train.columns),
                    "target": TARGET,
                    "feature_set": feature_set_name,
                    "model_name": model_name,
                    "day7_source_columns": day7_metadata,
                    "context_sources": context_sources.__dict__,
                },
                MODEL_DIR / f"{feature_set_name}__{model_name}.joblib",
            )

    metrics = pd.DataFrame(metric_rows).sort_values(["MAE", "RMSE"])
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics.to_csv(OUTPUT_DIR / "day7_updated_model_metrics.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "day7_updated_predictions.csv", index=False)
    context_metadata.to_csv(OUTPUT_DIR / "context_encoding_metadata.csv", index=False)

    best_rows = (
        metrics.sort_values("MAE")
        .groupby("FeatureSet", as_index=False)
        .first()
        .sort_values("MAE")
    )
    best_rows.to_csv(OUTPUT_DIR / "best_model_by_feature_set.csv", index=False)

    save_json(
        {
            "train_rows_with_7_day": len(train),
            "test_rows_with_7_day": len(test),
            "day7_source_columns": day7_metadata,
            "context_sources": context_sources.__dict__,
            "group_column": group_column,
            "best_overall_feature_set": str(metrics.iloc[0]["FeatureSet"]),
            "best_overall_model": str(metrics.iloc[0]["Model"]),
            "best_overall_mae": float(metrics.iloc[0]["MAE"]),
        },
        OUTPUT_DIR / "day7_updated_summary.json",
    )

    print("Model experiment 08 completed.")
    print(f"Common train rows with 7-day strength: {len(train):,}")
    print(f"Common test rows with 7-day strength: {len(test):,}")
    print(metrics[["FeatureSet", "Model", "MAE", "RMSE", "R2"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
