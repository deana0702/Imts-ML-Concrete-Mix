from __future__ import annotations

import time

import pandas as pd
from sklearn.model_selection import GroupKFold

from field_core_experiment_common import (
    classification_metrics,
    fit_classifier,
    positive_probability,
)
from field_core_validation_common import (
    VALIDATION_OUTPUT_ROOT,
    has_valid_day7,
    load_clean_training_data,
    prepare_context_encoded_features,
    prepare_failure_target,
    project_groups,
    selected_classification_model,
    summarize_classification_folds,
)


# Run with:
#     python code_field/11_grouped_cv_risk_classification.py
OUTPUT_DIR = VALIDATION_OUTPUT_ROOT / "11_grouped_cv_risk"
REQUESTED_FOLDS = 5
MODELS = ["LogisticRegression", "HistGradientBoosting"]
EVALUATION_THRESHOLD = 0.50


def evaluate_stage(
    *,
    stage: str,
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

    y_train = prepare_failure_target(fold_train)
    y_validation = prepare_failure_target(fold_validation)
    if y_train.nunique() < 2:
        raise ValueError(
            f"{stage} fold {fold_number} training rows do not contain both classes."
        )

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
        model = selected_classification_model(model_name)
        started = time.perf_counter()
        fitted = fit_classifier(model_name, model, x_train, y_train)
        probability = positive_probability(fitted, x_validation)
        elapsed = time.perf_counter() - started

        metric_rows.append(
            {
                "Stage": stage,
                "Fold": fold_number,
                "Model": model_name,
                "TrainRows": int(len(fold_train)),
                "ValidationRows": int(len(fold_validation)),
                "TrainFailures": int(y_train.sum()),
                "ValidationFailures": int(y_validation.sum()),
                "FeatureCount": int(x_train.shape[1]),
                "TrainingSeconds": elapsed,
                **classification_metrics(
                    y_validation.to_numpy(),
                    probability,
                    EVALUATION_THRESHOLD,
                ),
            }
        )

        predictions = pd.DataFrame(
            {
                "Stage": stage,
                "Fold": fold_number,
                "Model": model_name,
                "ActualFailureFlag28": y_validation.to_numpy(),
                "FailureProbability28": probability,
                "PredictedFailureAt050": (
                    probability >= EVALUATION_THRESHOLD
                ).astype(int),
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

    print("Validation 11 started: project-grouped risk cross-validation")
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
    y_for_split = prepare_failure_target(data)

    all_metrics: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []
    all_metadata: list[pd.DataFrame] = []

    for fold_number, (train_positions, validation_positions) in enumerate(
        splitter.split(data, y=y_for_split, groups=groups),
        start=1,
    ):
        fold_train = data.iloc[train_positions].copy()
        fold_validation = data.iloc[validation_positions].copy()

        overlap = set(groups.iloc[train_positions]).intersection(
            set(groups.iloc[validation_positions])
        )
        if overlap:
            raise RuntimeError(f"Project group leakage found in fold {fold_number}.")

        for stage, include_day7 in [("Day0_Context", False), ("Day7_FullUpdated", True)]:
            metrics, predictions, metadata = evaluate_stage(
                stage=stage,
                fold_train=fold_train,
                fold_validation=fold_validation,
                fold_number=fold_number,
                include_day7=include_day7,
            )
            all_metrics.extend(metrics)
            all_predictions.extend(predictions)
            all_metadata.append(metadata)

    fold_metrics = pd.DataFrame(all_metrics).sort_values(
        ["Stage", "Fold", "AveragePrecision_PR_AUC"],
        ascending=[True, True, False],
    )
    cv_summary = summarize_classification_folds(fold_metrics)
    predictions = pd.concat(all_predictions, ignore_index=True)
    metadata = pd.concat(all_metadata, ignore_index=True)

    fold_metrics.to_csv(OUTPUT_DIR / "grouped_cv_risk_fold_metrics.csv", index=False)
    cv_summary.to_csv(OUTPUT_DIR / "grouped_cv_risk_summary.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "grouped_cv_risk_predictions.csv", index=False)
    metadata.to_csv(OUTPUT_DIR / "grouped_cv_risk_context_metadata.csv", index=False)

    print("Validation 11 completed.")
    print(
        cv_summary[
            [
                "Stage",
                "Model",
                "MeanCV_AveragePrecision_PR_AUC",
                "StdCV_AveragePrecision_PR_AUC",
                "MeanCV_ROC_AUC",
                "MeanCV_Recall",
                "MeanCV_Precision",
                "MeanCV_FalseNegative",
            ]
        ].to_string(index=False)
    )
    print(
        "Threshold is fixed at 0.50 for validation only. "
        "Do not treat it as the final operational threshold."
    )
    print(f"Output: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
