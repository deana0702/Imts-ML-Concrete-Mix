from __future__ import annotations

import time

import numpy as np
import pandas as pd

from field_core_experiment_common import (
    TARGET,
    classification_metrics,
    fit_classifier,
    numeric_series,
    positive_probability,
    regression_metrics,
)
from field_core_validation_common import (
    EXTERNAL_PREPARED_FILE,
    VALIDATION_OUTPUT_ROOT,
    has_valid_day7,
    load_clean_training_data,
    metric_by_year_classification,
    metric_by_year_regression,
    prepare_context_encoded_features,
    prepare_failure_target,
    read_csv,
    resolve_cast_date_column,
    selected_classification_model,
    selected_regression_model,
)


# Run with:
#     python code_field/14_validate_external_2022_2026.py
#
# Prerequisite:
#     python code_field/13_prepare_external_2022_2026.py
OUTPUT_DIR = VALIDATION_OUTPUT_ROOT / "14_external_2022_2026"
TRAIN_END_YEAR = 2021
EXTERNAL_START_YEAR = 2022
EXTERNAL_END_YEAR = 2026
REGRESSION_MODELS = ["RandomForest", "HistGradientBoosting"]
CLASSIFICATION_MODELS = ["LogisticRegression", "HistGradientBoosting"]
EVALUATION_THRESHOLD = 0.50


def filter_training_period(train: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    cast_column = resolve_cast_date_column(train)
    dates = pd.to_datetime(train[cast_column], errors="coerce")
    mask = dates.dt.year.le(TRAIN_END_YEAR)
    result = train.loc[mask].copy()
    result[cast_column] = dates.loc[mask]
    if result.empty:
        raise ValueError(f"No training rows were found through {TRAIN_END_YEAR}.")
    return result, cast_column


def validate_external_period(external: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    cast_column = resolve_cast_date_column(external)
    dates = pd.to_datetime(external[cast_column], errors="coerce")
    mask = dates.dt.year.between(EXTERNAL_START_YEAR, EXTERNAL_END_YEAR)
    result = external.loc[mask].copy()
    result[cast_column] = dates.loc[mask]
    result["ExternalValidationYear"] = result[cast_column].dt.year.astype(int)
    if result.empty:
        raise ValueError(
            f"No external rows were found from {EXTERNAL_START_YEAR} through "
            f"{EXTERNAL_END_YEAR}."
        )
    return result, cast_column


def run_regression_stage(
    *,
    stage: str,
    train: pd.DataFrame,
    external: pd.DataFrame,
    include_day7: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if include_day7:
        train = train.loc[has_valid_day7(train)].copy()
        external = external.loc[has_valid_day7(external)].copy()

    y_train = numeric_series(train, TARGET)
    y_external = numeric_series(external, TARGET)
    x_train, x_external, context_metadata = prepare_context_encoded_features(
        train,
        external,
        y_train,
        include_day7=include_day7,
    )

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    yearly_frames: list[pd.DataFrame] = []

    for model_name in REGRESSION_MODELS:
        print(f"  External {stage} regression / {model_name}")
        model = selected_regression_model(model_name)
        started = time.perf_counter()
        model.fit(x_train, y_train)
        predicted = model.predict(x_external)
        elapsed = time.perf_counter() - started

        metric_rows.append(
            {
                "Stage": stage,
                "Task": "Regression",
                "Model": model_name,
                "TrainRows": int(len(train)),
                "ExternalRows": int(len(external)),
                "FeatureCount": int(x_train.shape[1]),
                "TrainingSeconds": elapsed,
                **regression_metrics(y_external.to_numpy(), predicted),
            }
        )

        frame = pd.DataFrame(
            {
                "Stage": stage,
                "Task": "Regression",
                "Model": model_name,
                "ActualStrength28_psi": y_external.to_numpy(),
                "PredictedStrength28_psi": predicted,
                "ResidualPsi": predicted - y_external.to_numpy(),
                "AbsoluteErrorPsi": np.abs(predicted - y_external.to_numpy()),
                "ExternalValidationYear": external["ExternalValidationYear"].to_numpy(),
            },
            index=external.index,
        )
        for identifier in [
            "testId",
            "projectId",
            "projectNo",
            "OfficeName",
            "SupplierId",
            "SupplierName",
            "PlantNumber",
            "MixNumber",
            "castDate",
        ]:
            if identifier in external.columns:
                frame[identifier] = external[identifier]
        frame = frame.reset_index(drop=True)
        prediction_frames.append(frame)

        yearly = metric_by_year_regression(
            frame,
            actual_column="ActualStrength28_psi",
            predicted_column="PredictedStrength28_psi",
            year_column="ExternalValidationYear",
        )
        yearly.insert(0, "Model", model_name)
        yearly.insert(0, "Stage", stage)
        yearly_frames.append(yearly)

    context_metadata = context_metadata.copy()
    context_metadata.insert(0, "Stage", stage)
    return (
        pd.DataFrame(metric_rows),
        pd.concat(prediction_frames, ignore_index=True),
        pd.concat(yearly_frames, ignore_index=True),
        context_metadata,
    )


def run_classification_stage(
    *,
    stage: str,
    train: pd.DataFrame,
    external: pd.DataFrame,
    include_day7: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if include_day7:
        train = train.loc[has_valid_day7(train)].copy()
        external = external.loc[has_valid_day7(external)].copy()

    y_train = prepare_failure_target(train)
    y_external = prepare_failure_target(external)
    if y_train.nunique() < 2:
        raise ValueError(f"{stage} training rows need both classes.")

    x_train, x_external, context_metadata = prepare_context_encoded_features(
        train,
        external,
        y_train,
        include_day7=include_day7,
    )

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    yearly_frames: list[pd.DataFrame] = []

    for model_name in CLASSIFICATION_MODELS:
        print(f"  External {stage} risk / {model_name}")
        model = selected_classification_model(model_name)
        started = time.perf_counter()
        fitted = fit_classifier(model_name, model, x_train, y_train)
        probability = positive_probability(fitted, x_external)
        elapsed = time.perf_counter() - started

        metric_rows.append(
            {
                "Stage": stage,
                "Task": "Classification",
                "Model": model_name,
                "TrainRows": int(len(train)),
                "ExternalRows": int(len(external)),
                "TrainFailures": int(y_train.sum()),
                "ExternalFailures": int(y_external.sum()),
                "FeatureCount": int(x_train.shape[1]),
                "TrainingSeconds": elapsed,
                **classification_metrics(
                    y_external.to_numpy(),
                    probability,
                    EVALUATION_THRESHOLD,
                ),
            }
        )

        frame = pd.DataFrame(
            {
                "Stage": stage,
                "Task": "Classification",
                "Model": model_name,
                "ActualFailureFlag28": y_external.to_numpy(),
                "FailureProbability28": probability,
                "PredictedFailureAt050": (
                    probability >= EVALUATION_THRESHOLD
                ).astype(int),
                "ExternalValidationYear": external["ExternalValidationYear"].to_numpy(),
            },
            index=external.index,
        )
        for identifier in [
            "testId",
            "projectId",
            "projectNo",
            "OfficeName",
            "SupplierId",
            "SupplierName",
            "PlantNumber",
            "MixNumber",
            "castDate",
        ]:
            if identifier in external.columns:
                frame[identifier] = external[identifier]
        frame = frame.reset_index(drop=True)
        prediction_frames.append(frame)

        yearly = metric_by_year_classification(
            frame,
            actual_column="ActualFailureFlag28",
            probability_column="FailureProbability28",
            year_column="ExternalValidationYear",
            threshold=EVALUATION_THRESHOLD,
        )
        yearly.insert(0, "Model", model_name)
        yearly.insert(0, "Stage", stage)
        yearly_frames.append(yearly)

    context_metadata = context_metadata.copy()
    context_metadata.insert(0, "Stage", stage)
    return (
        pd.DataFrame(metric_rows),
        pd.concat(prediction_frames, ignore_index=True),
        pd.concat(yearly_frames, ignore_index=True),
        context_metadata,
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Validation 14 started: true time-based external validation, 2022-2026")
    train_raw, train_path = load_clean_training_data()
    external_raw = read_csv(EXTERNAL_PREPARED_FILE)
    train, train_cast_column = filter_training_period(train_raw)
    external, external_cast_column = validate_external_period(external_raw)

    overlap = set(train["testId"]).intersection(set(external["testId"]))
    if overlap:
        raise RuntimeError(
            f"Training and external data contain {len(overlap)} overlapping test IDs."
        )

    print(f"Training input: {train_path}")
    print(
        f"Training rows through {TRAIN_END_YEAR}: {len(train):,}; "
        f"date range {train[train_cast_column].min()} to {train[train_cast_column].max()}"
    )
    print(f"External input: {EXTERNAL_PREPARED_FILE}")
    print(
        f"External rows {EXTERNAL_START_YEAR}-{EXTERNAL_END_YEAR}: {len(external):,}; "
        f"date range {external[external_cast_column].min()} to "
        f"{external[external_cast_column].max()}"
    )

    regression_metrics_frames: list[pd.DataFrame] = []
    regression_prediction_frames: list[pd.DataFrame] = []
    regression_yearly_frames: list[pd.DataFrame] = []
    classification_metrics_frames: list[pd.DataFrame] = []
    classification_prediction_frames: list[pd.DataFrame] = []
    classification_yearly_frames: list[pd.DataFrame] = []
    metadata_frames: list[pd.DataFrame] = []

    for stage, include_day7 in [("Day0_Context", False), ("Day7_FullUpdated", True)]:
        metrics, predictions, yearly, metadata = run_regression_stage(
            stage=stage,
            train=train,
            external=external,
            include_day7=include_day7,
        )
        regression_metrics_frames.append(metrics)
        regression_prediction_frames.append(predictions)
        regression_yearly_frames.append(yearly)
        metadata_frames.append(metadata.assign(Task="Regression"))

        metrics, predictions, yearly, metadata = run_classification_stage(
            stage=stage,
            train=train,
            external=external,
            include_day7=include_day7,
        )
        classification_metrics_frames.append(metrics)
        classification_prediction_frames.append(predictions)
        classification_yearly_frames.append(yearly)
        metadata_frames.append(metadata.assign(Task="Classification"))

    regression_metrics_df = pd.concat(regression_metrics_frames, ignore_index=True)
    regression_metrics_df = regression_metrics_df.sort_values(["Stage", "MAE"])
    classification_metrics_df = pd.concat(
        classification_metrics_frames, ignore_index=True
    ).sort_values(
        ["Stage", "AveragePrecision_PR_AUC"],
        ascending=[True, False],
    )
    regression_predictions_df = pd.concat(
        regression_prediction_frames, ignore_index=True
    )
    classification_predictions_df = pd.concat(
        classification_prediction_frames, ignore_index=True
    )
    regression_yearly_df = pd.concat(regression_yearly_frames, ignore_index=True)
    classification_yearly_df = pd.concat(
        classification_yearly_frames, ignore_index=True
    )
    metadata_df = pd.concat(metadata_frames, ignore_index=True)

    regression_metrics_df.to_csv(
        OUTPUT_DIR / "external_regression_metrics.csv", index=False
    )
    classification_metrics_df.to_csv(
        OUTPUT_DIR / "external_risk_metrics_at_050.csv", index=False
    )
    regression_predictions_df.to_csv(
        OUTPUT_DIR / "external_regression_predictions.csv", index=False
    )
    classification_predictions_df.to_csv(
        OUTPUT_DIR / "external_risk_predictions.csv", index=False
    )
    regression_yearly_df.to_csv(
        OUTPUT_DIR / "external_regression_metrics_by_year.csv", index=False
    )
    classification_yearly_df.to_csv(
        OUTPUT_DIR / "external_risk_metrics_by_year.csv", index=False
    )
    metadata_df.to_csv(
        OUTPUT_DIR / "external_context_encoding_metadata.csv", index=False
    )

    print("Validation 14 completed.")
    print("\nExternal regression:")
    print(
        regression_metrics_df[
            ["Stage", "Model", "ExternalRows", "MAE", "RMSE", "R2"]
        ].to_string(index=False)
    )
    print("\nExternal risk classification at threshold 0.50:")
    print(
        classification_metrics_df[
            [
                "Stage",
                "Model",
                "ExternalRows",
                "ExternalFailures",
                "AveragePrecision_PR_AUC",
                "ROC_AUC",
                "Recall",
                "Precision",
                "FalseNegative",
            ]
        ].to_string(index=False)
    )
    print(
        "Threshold 0.50 is reported for comparison only. "
        "Select an operational threshold using training/validation data, not this external set."
    )
    print(f"Output: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
