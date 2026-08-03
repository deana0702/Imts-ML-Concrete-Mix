from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from field_core_experiment_common import TARGET, numeric_series, regression_metrics
from field_core_validation_common import (
    VALIDATION_OUTPUT_ROOT,
    has_valid_day7,
    load_clean_training_data,
    prepare_context_encoded_features,
    project_groups,
    selected_regression_model,
    summarize_regression_folds,
)


# Run with:
#     python code_field/10_grouped_cv_regression.py
OUTPUT_DIR = VALIDATION_OUTPUT_ROOT / "10_grouped_cv_regression"
REQUESTED_FOLDS = 5
MODELS = ["RandomForest", "HistGradientBoosting"]


def evaluate_stage(
    *,
    stage: str,
    data: pd.DataFrame,
    fold_train: pd.DataFrame,
    fold_validation: pd.DataFrame,
    fold_number: int,
    include_day7: bool,
) -> tuple[list[dict[str, object]], list[pd.DataFrame], pd.DataFrame]:
    if include_day7:
        fold_train = fold_train.loc[has_valid_day7(fold_train)].copy()
        fold_validation = fold_validation.loc[has_valid_day7(fold_validation)].copy()

    if fold_train.empty or fold_validation.empty:
        raise ValueError(f"{stage} fold {fold_number} has no usable rows.")

    y_train = numeric_series(fold_train, TARGET)
    y_validation = numeric_series(fold_validation, TARGET)

    x_train, x_validation, context_metadata = prepare_context_encoded_features(
        fold_train,
        fold_validation,
        y_train,
        include_day7=include_day7,
    )

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for model_name in MODELS:
        print(f"  Fold {fold_number}: {stage} / {model_name}")
        model = selected_regression_model(model_name)
        started = time.perf_counter()
        model.fit(x_train, y_train)
        predicted = model.predict(x_validation)
        elapsed = time.perf_counter() - started

        metric_rows.append(
            {
                "Stage": stage,
                "Fold": fold_number,
                "Model": model_name,
                "TrainRows": int(len(fold_train)),
                "ValidationRows": int(len(fold_validation)),
                "FeatureCount": int(x_train.shape[1]),
                "TrainingSeconds": elapsed,
                **regression_metrics(y_validation.to_numpy(), predicted),
            }
        )

        predictions = pd.DataFrame(
            {
                "Stage": stage,
                "Fold": fold_number,
                "Model": model_name,
                "ActualStrength28_psi": y_validation.to_numpy(),
                "PredictedStrength28_psi": predicted,
                "ResidualPsi": predicted - y_validation.to_numpy(),
                "AbsoluteErrorPsi": np.abs(predicted - y_validation.to_numpy()),
            },
            index=fold_validation.index,
        )
        for identifier in [
            "testId",
            "projectId",
            "projectNo",
            "SupplierId",
            "SupplierName",
            "PlantNumber",
            "MixNumber",
            "castDate",
        ]:
            if identifier in fold_validation.columns:
                predictions[identifier] = fold_validation[identifier]
        prediction_frames.append(predictions.reset_index(drop=True))

    context_metadata = context_metadata.copy()
    context_metadata.insert(0, "Fold", fold_number)
    context_metadata.insert(0, "Stage", stage)
    return metric_rows, prediction_frames, context_metadata


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Validation 10 started: project-grouped regression cross-validation")
    data, input_path = load_clean_training_data()
    groups, group_column = project_groups(data)
    fold_count = min(REQUESTED_FOLDS, int(groups.nunique()))
    if fold_count < 2:
        raise ValueError("At least two project groups are required.")

    print(f"Input: {input_path}")
    print(f"Rows: {len(data):,}")
    print(f"Grouping column: {group_column}")
    print(f"Project groups: {groups.nunique():,}")
    print(f"Folds: {fold_count}")

    splitter = GroupKFold(n_splits=fold_count)
    all_metrics: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []
    all_metadata: list[pd.DataFrame] = []

    for fold_number, (train_positions, validation_positions) in enumerate(
        splitter.split(data, y=numeric_series(data, TARGET), groups=groups),
        start=1,
    ):
        fold_train = data.iloc[train_positions].copy()
        fold_validation = data.iloc[validation_positions].copy()

        train_groups = set(groups.iloc[train_positions])
        validation_groups = set(groups.iloc[validation_positions])
        overlap = train_groups.intersection(validation_groups)
        if overlap:
            raise RuntimeError(f"Project group leakage found in fold {fold_number}.")

        for stage, include_day7 in [("Day0_Context", False), ("Day7_FullUpdated", True)]:
            metrics, predictions, metadata = evaluate_stage(
                stage=stage,
                data=data,
                fold_train=fold_train,
                fold_validation=fold_validation,
                fold_number=fold_number,
                include_day7=include_day7,
            )
            all_metrics.extend(metrics)
            all_predictions.extend(predictions)
            all_metadata.append(metadata)

    fold_metrics = pd.DataFrame(all_metrics).sort_values(["Stage", "Fold", "MAE"])
    cv_summary = summarize_regression_folds(fold_metrics)
    predictions = pd.concat(all_predictions, ignore_index=True)
    metadata = pd.concat(all_metadata, ignore_index=True)

    fold_metrics.to_csv(OUTPUT_DIR / "grouped_cv_regression_fold_metrics.csv", index=False)
    cv_summary.to_csv(OUTPUT_DIR / "grouped_cv_regression_summary.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "grouped_cv_regression_predictions.csv", index=False)
    metadata.to_csv(OUTPUT_DIR / "grouped_cv_context_metadata.csv", index=False)

    print("Validation 10 completed.")
    print(
        cv_summary[
            [
                "Stage",
                "Model",
                "MeanCV_MAE",
                "StdCV_MAE",
                "MeanCV_RMSE",
                "MeanCV_R2",
            ]
        ].to_string(index=False)
    )
    print(f"Output: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
