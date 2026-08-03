from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd

from field_core_experiment_common import (
    DAY7_FEATURES,
    FIELD_PLUS_REQUIRED_FEATURES,
    NEXT_OUTPUT_ROOT,
    REQUIRED_STRENGTH,
    TARGET,
    add_day7_features,
    build_classification_models,
    build_context_categories,
    classification_metrics,
    cross_fitted_target_encode,
    fit_classifier,
    load_comparison_splits,
    numeric_frame,
    numeric_series,
    positive_probability,
    resolve_context_sources,
    resolve_group_column,
    save_json,
    threshold_metrics_table,
)


# Run with:
#     python 09_early_risk_classification.py
OUTPUT_DIR = NEXT_OUTPUT_ROOT / "09_early_risk"
MODEL_DIR = OUTPUT_DIR / "saved_models"


def prepare_failure_target(df: pd.DataFrame) -> pd.Series:
    actual = numeric_series(df, TARGET)
    required = numeric_series(df, REQUIRED_STRENGTH)
    if actual.isna().any() or required.isna().any():
        raise ValueError("Risk data contains missing actual or required strength.")
    return actual.lt(required).astype(int)


def run_stage(
    *,
    stage_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_sets: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[list[dict[str, object]], list[pd.DataFrame], list[pd.DataFrame]]:
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    threshold_frames: list[pd.DataFrame] = []

    for feature_set_name, (x_train, x_test) in feature_sets.items():
        for model_name, model in build_classification_models().items():
            print(f"Training {stage_name} / {feature_set_name} / {model_name} ...")
            start = time.perf_counter()
            fitted = fit_classifier(model_name, model, x_train, y_train)
            probability = positive_probability(fitted, x_test)
            elapsed = time.perf_counter() - start

            metric_rows.append(
                {
                    "Stage": stage_name,
                    "FeatureSet": feature_set_name,
                    "Model": model_name,
                    "TrainRows": len(train),
                    "TestRows": len(test),
                    "TrainFailures": int(y_train.sum()),
                    "TestFailures": int(y_test.sum()),
                    "FeatureCount": x_train.shape[1],
                    "TrainingSeconds": elapsed,
                    **classification_metrics(y_test.to_numpy(), probability, 0.50),
                }
            )

            prediction_frame = pd.DataFrame(
                {
                    "Stage": stage_name,
                    "FeatureSet": feature_set_name,
                    "Model": model_name,
                    "ActualFailureFlag28": y_test.to_numpy(),
                    "FailureProbability28": probability,
                    "PredictedFailureAt050": (probability >= 0.50).astype(int),
                }
            )
            for identifier in ["testId", "projectId", "projectNo", "OfficeName"]:
                if identifier in test.columns:
                    prediction_frame[identifier] = test[identifier].to_numpy()
            prediction_frames.append(prediction_frame)

            threshold_frame = threshold_metrics_table(y_test.to_numpy(), probability)
            threshold_frame.insert(0, "Model", model_name)
            threshold_frame.insert(0, "FeatureSet", feature_set_name)
            threshold_frame.insert(0, "Stage", stage_name)
            threshold_frames.append(threshold_frame)

            joblib.dump(
                {
                    "model": fitted,
                    "numeric_features": list(x_train.columns),
                    "target": "FailureFlag28",
                    "stage": stage_name,
                    "feature_set": feature_set_name,
                    "model_name": model_name,
                    "warning": (
                        "This is an early screening model, not an engineering "
                        "acceptance or rejection decision."
                    ),
                },
                MODEL_DIR / f"{stage_name}__{feature_set_name}__{model_name}.joblib",
            )

    return metric_rows, prediction_frames, threshold_frames


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Model experiment 09 started: Early Quality Risk classification")
    train_raw, test_raw = load_comparison_splits()
    y_train_day0 = prepare_failure_target(train_raw)
    y_test_day0 = prepare_failure_target(test_raw)

    base_train = numeric_frame(train_raw, FIELD_PLUS_REQUIRED_FEATURES)
    base_test = numeric_frame(test_raw, FIELD_PLUS_REQUIRED_FEATURES)

    context_sources = resolve_context_sources(train_raw)
    group_column = resolve_group_column(train_raw)
    train_context = build_context_categories(train_raw, context_sources)
    test_context = build_context_categories(test_raw, context_sources)
    encoded_train, encoded_test, day0_context_metadata = cross_fitted_target_encode(
        train_context,
        test_context,
        y_train_day0,
        train_raw[group_column],
    )

    all_metric_rows: list[dict[str, object]] = []
    all_prediction_frames: list[pd.DataFrame] = []
    all_threshold_frames: list[pd.DataFrame] = []

    day0_sets = {
        "FieldPlusRequired": (base_train, base_test),
        "FieldPlusRequiredPlusContext": (
            pd.concat([base_train, encoded_train], axis=1),
            pd.concat([base_test, encoded_test], axis=1),
        ),
    }
    metrics, predictions, thresholds = run_stage(
        stage_name="Day0",
        train=train_raw,
        test=test_raw,
        feature_sets=day0_sets,
        y_train=y_train_day0,
        y_test=y_test_day0,
    )
    all_metric_rows.extend(metrics)
    all_prediction_frames.extend(predictions)
    all_threshold_frames.extend(thresholds)

    train_day7, day7_metadata = add_day7_features(train_raw)
    test_day7, _ = add_day7_features(test_raw)
    train_mask = numeric_series(train_day7, "Day7AverageStrength_psi").gt(0)
    test_mask = numeric_series(test_day7, "Day7AverageStrength_psi").gt(0)
    train_day7 = train_day7.loc[train_mask].copy()
    test_day7 = test_day7.loc[test_mask].copy()
    y_train_day7 = prepare_failure_target(train_day7)
    y_test_day7 = prepare_failure_target(test_day7)

    day7_base_train = numeric_frame(train_day7, FIELD_PLUS_REQUIRED_FEATURES)
    day7_base_test = numeric_frame(test_day7, FIELD_PLUS_REQUIRED_FEATURES)
    day7_numeric_train = numeric_frame(train_day7, DAY7_FEATURES)
    day7_numeric_test = numeric_frame(test_day7, DAY7_FEATURES)

    day7_train_context = build_context_categories(train_day7, context_sources)
    day7_test_context = build_context_categories(test_day7, context_sources)
    day7_encoded_train, day7_encoded_test, day7_context_metadata = (
        cross_fitted_target_encode(
            day7_train_context,
            day7_test_context,
            y_train_day7,
            train_day7[group_column],
        )
    )

    day7_sets = {
        "FieldPlusRequiredPlus7Day": (
            pd.concat([day7_base_train, day7_numeric_train], axis=1),
            pd.concat([day7_base_test, day7_numeric_test], axis=1),
        ),
        "FullUpdated_ContextPlus7Day": (
            pd.concat(
                [day7_base_train, day7_encoded_train, day7_numeric_train],
                axis=1,
            ),
            pd.concat(
                [day7_base_test, day7_encoded_test, day7_numeric_test],
                axis=1,
            ),
        ),
    }
    metrics, predictions, thresholds = run_stage(
        stage_name="Day7",
        train=train_day7,
        test=test_day7,
        feature_sets=day7_sets,
        y_train=y_train_day7,
        y_test=y_test_day7,
    )
    all_metric_rows.extend(metrics)
    all_prediction_frames.extend(predictions)
    all_threshold_frames.extend(thresholds)

    metrics_df = pd.DataFrame(all_metric_rows).sort_values(
        ["Stage", "AveragePrecision_PR_AUC", "Recall"],
        ascending=[True, False, False],
    )
    predictions_df = pd.concat(all_prediction_frames, ignore_index=True)
    thresholds_df = pd.concat(all_threshold_frames, ignore_index=True)

    metrics_df.to_csv(OUTPUT_DIR / "risk_classifier_metrics_at_050.csv", index=False)
    predictions_df.to_csv(OUTPUT_DIR / "risk_classifier_predictions.csv", index=False)
    thresholds_df.to_csv(OUTPUT_DIR / "risk_threshold_analysis.csv", index=False)
    day0_context_metadata.to_csv(
        OUTPUT_DIR / "day0_context_encoding_metadata.csv",
        index=False,
    )
    day7_context_metadata.to_csv(
        OUTPUT_DIR / "day7_context_encoding_metadata.csv",
        index=False,
    )

    best_day0 = metrics_df.loc[metrics_df["Stage"].eq("Day0")].iloc[0]
    best_day7 = metrics_df.loc[metrics_df["Stage"].eq("Day7")].iloc[0]
    save_json(
        {
            "day0_train_rows": len(train_raw),
            "day0_test_rows": len(test_raw),
            "day0_train_failures": int(y_train_day0.sum()),
            "day0_test_failures": int(y_test_day0.sum()),
            "day7_train_rows": len(train_day7),
            "day7_test_rows": len(test_day7),
            "day7_train_failures": int(y_train_day7.sum()),
            "day7_test_failures": int(y_test_day7.sum()),
            "day7_source_columns": day7_metadata,
            "context_sources": context_sources.__dict__,
            "best_day0_by_pr_auc": {
                "feature_set": str(best_day0["FeatureSet"]),
                "model": str(best_day0["Model"]),
                "pr_auc": float(best_day0["AveragePrecision_PR_AUC"]),
                "recall_at_050": float(best_day0["Recall"]),
                "precision_at_050": float(best_day0["Precision"]),
            },
            "best_day7_by_pr_auc": {
                "feature_set": str(best_day7["FeatureSet"]),
                "model": str(best_day7["Model"]),
                "pr_auc": float(best_day7["AveragePrecision_PR_AUC"]),
                "recall_at_050": float(best_day7["Recall"]),
                "precision_at_050": float(best_day7["Precision"]),
            },
        },
        OUTPUT_DIR / "risk_classification_summary.json",
    )

    print("Model experiment 09 completed.")
    print(
        metrics_df[
            [
                "Stage",
                "FeatureSet",
                "Model",
                "AveragePrecision_PR_AUC",
                "ROC_AUC",
                "Recall",
                "Precision",
                "FalseNegative",
            ]
        ].to_string(index=False)
    )
    print(
        "Review risk_threshold_analysis.csv before selecting a screening threshold. "
        "Do not choose a final threshold from the test data alone."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
