from __future__ import annotations

from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from field_core_experiment_common import (
    FIELD_PLUS_REQUIRED_FEATURES,
    NEXT_OUTPUT_ROOT,
    REQUIRED_STRENGTH,
    TARGET,
    build_regression_models,
    load_comparison_splits,
    numeric_frame,
    numeric_series,
    regression_metrics,
    save_json,
)


# Run with:
#     python 06_compare_required_only.py
OUTPUT_DIR = NEXT_OUTPUT_ROOT / "06_required_only"
MODEL_DIR = OUTPUT_DIR / "saved_models"

FEATURE_SETS = {
    "RequiredOnly": [REQUIRED_STRENGTH],
    "FieldPlusRequired": FIELD_PLUS_REQUIRED_FEATURES,
}


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Model experiment 06 started: Required Only comparison")
    train, test = load_comparison_splits()
    y_train = numeric_series(train, TARGET)
    y_test = numeric_series(test, TARGET)

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for feature_set_name, features in FEATURE_SETS.items():
        x_train = numeric_frame(train, features)
        x_test = numeric_frame(test, features)

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
                    "FeatureCount": len(features),
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
                    "features": features,
                    "target": TARGET,
                    "feature_set": feature_set_name,
                    "model_name": model_name,
                },
                MODEL_DIR / f"{feature_set_name}__{model_name}.joblib",
            )

    metrics = pd.DataFrame(metric_rows).sort_values(["MAE", "RMSE"])
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics.to_csv(OUTPUT_DIR / "required_only_model_metrics.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "required_only_predictions.csv", index=False)

    best_required = metrics.loc[metrics["FeatureSet"].eq("RequiredOnly")].iloc[0]
    best_combined = metrics.loc[
        metrics["FeatureSet"].eq("FieldPlusRequired")
    ].iloc[0]
    improvement = float(best_required["MAE"] - best_combined["MAE"])
    improvement_percent = float(improvement / best_required["MAE"] * 100.0)

    summary = {
        "best_required_only_model": str(best_required["Model"]),
        "best_required_only_mae": float(best_required["MAE"]),
        "best_field_plus_required_model": str(best_combined["Model"]),
        "best_field_plus_required_mae": float(best_combined["MAE"]),
        "field_measurement_added_mae_improvement": improvement,
        "field_measurement_added_mae_improvement_percent": improvement_percent,
    }
    save_json(summary, OUTPUT_DIR / "required_only_summary.json")

    print("Model experiment 06 completed.")
    print(metrics[["FeatureSet", "Model", "MAE", "RMSE", "R2"]].to_string(index=False))
    print(
        "Best Required Only MAE: "
        f"{best_required['MAE']:.1f} psi; "
        "Best Field + Required MAE: "
        f"{best_combined['MAE']:.1f} psi"
    )
    print(
        "Additional value from field measurements: "
        f"{improvement:.1f} psi ({improvement_percent:.1f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
