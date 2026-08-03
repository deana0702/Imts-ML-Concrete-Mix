from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd

from field_core_experiment_common import (
    FIELD_PLUS_REQUIRED_FEATURES,
    NEXT_OUTPUT_ROOT,
    TARGET,
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
#     python 07_compare_day0_context.py
OUTPUT_DIR = NEXT_OUTPUT_ROOT / "07_day0_context"
MODEL_DIR = OUTPUT_DIR / "saved_models"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Model experiment 07 started: Day-0 supplier/plant/mix context")
    train, test = load_comparison_splits()
    y_train = numeric_series(train, TARGET)
    y_test = numeric_series(test, TARGET)

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

    base_train = numeric_frame(train, FIELD_PLUS_REQUIRED_FEATURES)
    base_test = numeric_frame(test, FIELD_PLUS_REQUIRED_FEATURES)
    context_train = pd.concat([base_train, encoded_train], axis=1)
    context_test = pd.concat([base_test, encoded_test], axis=1)

    feature_sets = {
        "FieldPlusRequired": (base_train, base_test),
        "FieldPlusRequiredPlusContext": (context_train, context_test),
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
                    "context_sources": context_sources.__dict__,
                    "context_encoding_note": (
                        "Training context target means were created out-of-fold by "
                        "project group. Test mappings came only from training data."
                    ),
                },
                MODEL_DIR / f"{feature_set_name}__{model_name}.joblib",
            )

    metrics = pd.DataFrame(metric_rows).sort_values(["MAE", "RMSE"])
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics.to_csv(OUTPUT_DIR / "day0_context_model_metrics.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "day0_context_predictions.csv", index=False)
    context_metadata.to_csv(OUTPUT_DIR / "context_encoding_metadata.csv", index=False)

    best_base = metrics.loc[metrics["FeatureSet"].eq("FieldPlusRequired")].iloc[0]
    best_context = metrics.loc[
        metrics["FeatureSet"].eq("FieldPlusRequiredPlusContext")
    ].iloc[0]
    improvement = float(best_base["MAE"] - best_context["MAE"])
    improvement_percent = float(improvement / best_base["MAE"] * 100.0)

    save_json(
        {
            "context_sources": context_sources.__dict__,
            "group_column": group_column,
            "best_base_model": str(best_base["Model"]),
            "best_base_mae": float(best_base["MAE"]),
            "best_context_model": str(best_context["Model"]),
            "best_context_mae": float(best_context["MAE"]),
            "context_mae_improvement": improvement,
            "context_mae_improvement_percent": improvement_percent,
        },
        OUTPUT_DIR / "day0_context_summary.json",
    )

    print("Model experiment 07 completed.")
    print(metrics[["FeatureSet", "Model", "MAE", "RMSE", "R2"]].to_string(index=False))
    print(
        "Context improvement over Field + Required: "
        f"{improvement:.1f} psi ({improvement_percent:.1f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
