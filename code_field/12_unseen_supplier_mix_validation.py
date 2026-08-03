from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from field_core_experiment_common import (
    TARGET,
    classification_metrics,
    fit_classifier,
    numeric_series,
    positive_probability,
    regression_metrics,
)
from field_core_validation_common import (
    VALIDATION_OUTPUT_ROOT,
    category_keys,
    has_valid_day7,
    load_clean_training_data,
    prepare_context_encoded_features,
    prepare_failure_target,
    scenario_group_values,
    selected_classification_model,
    selected_regression_model,
    valid_context_mask,
)


# Run with:
#     python code_field/12_unseen_supplier_mix_validation.py
OUTPUT_DIR = VALIDATION_OUTPUT_ROOT / "12_unseen_supplier_mix"
TEST_SIZE = 0.20
RANDOM_STATE = 42
SCENARIOS = ["UnseenSupplier", "UnseenSupplierPlantMix"]
REGRESSION_MODELS = ["RandomForest", "HistGradientBoosting"]
CLASSIFICATION_MODELS = ["LogisticRegression", "HistGradientBoosting"]
EVALUATION_THRESHOLD = 0.50


def evaluate_regression(
    *,
    scenario: str,
    stage: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    include_day7: bool,
) -> tuple[list[dict[str, object]], list[pd.DataFrame], pd.DataFrame]:
    if include_day7:
        train = train.loc[has_valid_day7(train)].copy()
        test = test.loc[has_valid_day7(test)].copy()

    y_train = numeric_series(train, TARGET)
    y_test = numeric_series(test, TARGET)
    x_train, x_test, metadata = prepare_context_encoded_features(
        train,
        test,
        y_train,
        include_day7=include_day7,
    )

    metrics: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for model_name in REGRESSION_MODELS:
        print(f"  {scenario}: {stage} regression / {model_name}")
        model = selected_regression_model(model_name)
        started = time.perf_counter()
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        elapsed = time.perf_counter() - started

        metrics.append(
            {
                "Scenario": scenario,
                "Task": "Regression",
                "Stage": stage,
                "Model": model_name,
                "TrainRows": int(len(train)),
                "TestRows": int(len(test)),
                "FeatureCount": int(x_train.shape[1]),
                "TrainingSeconds": elapsed,
                **regression_metrics(y_test.to_numpy(), predicted),
            }
        )

        frame = pd.DataFrame(
            {
                "Scenario": scenario,
                "Task": "Regression",
                "Stage": stage,
                "Model": model_name,
                "ActualStrength28_psi": y_test.to_numpy(),
                "PredictedStrength28_psi": predicted,
                "ResidualPsi": predicted - y_test.to_numpy(),
                "AbsoluteErrorPsi": np.abs(predicted - y_test.to_numpy()),
            },
            index=test.index,
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
            if identifier in test.columns:
                frame[identifier] = test[identifier]
        predictions.append(frame.reset_index(drop=True))

    metadata = metadata.copy()
    metadata.insert(0, "Stage", stage)
    metadata.insert(0, "Scenario", scenario)
    return metrics, predictions, metadata


def evaluate_classification(
    *,
    scenario: str,
    stage: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    include_day7: bool,
) -> tuple[list[dict[str, object]], list[pd.DataFrame], pd.DataFrame]:
    if include_day7:
        train = train.loc[has_valid_day7(train)].copy()
        test = test.loc[has_valid_day7(test)].copy()

    y_train = prepare_failure_target(train)
    y_test = prepare_failure_target(test)
    if y_train.nunique() < 2:
        raise ValueError(f"{scenario}/{stage} training rows need both classes.")

    x_train, x_test, metadata = prepare_context_encoded_features(
        train,
        test,
        y_train,
        include_day7=include_day7,
    )

    metrics: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for model_name in CLASSIFICATION_MODELS:
        print(f"  {scenario}: {stage} risk / {model_name}")
        model = selected_classification_model(model_name)
        started = time.perf_counter()
        fitted = fit_classifier(model_name, model, x_train, y_train)
        probability = positive_probability(fitted, x_test)
        elapsed = time.perf_counter() - started

        metrics.append(
            {
                "Scenario": scenario,
                "Task": "Classification",
                "Stage": stage,
                "Model": model_name,
                "TrainRows": int(len(train)),
                "TestRows": int(len(test)),
                "TrainFailures": int(y_train.sum()),
                "TestFailures": int(y_test.sum()),
                "FeatureCount": int(x_train.shape[1]),
                "TrainingSeconds": elapsed,
                **classification_metrics(
                    y_test.to_numpy(),
                    probability,
                    EVALUATION_THRESHOLD,
                ),
            }
        )

        frame = pd.DataFrame(
            {
                "Scenario": scenario,
                "Task": "Classification",
                "Stage": stage,
                "Model": model_name,
                "ActualFailureFlag28": y_test.to_numpy(),
                "FailureProbability28": probability,
                "PredictedFailureAt050": (
                    probability >= EVALUATION_THRESHOLD
                ).astype(int),
            },
            index=test.index,
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
            if identifier in test.columns:
                frame[identifier] = test[identifier]
        predictions.append(frame.reset_index(drop=True))

    metadata = metadata.copy()
    metadata.insert(0, "Stage", stage)
    metadata.insert(0, "Scenario", scenario)
    return metrics, predictions, metadata


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Validation 12 started: unseen supplier and supplier/plant/mix holdout")
    data, input_path = load_clean_training_data()
    categories = category_keys(data)
    print(f"Input: {input_path}")
    print(f"Rows before context filtering: {len(data):,}")

    all_regression_metrics: list[dict[str, object]] = []
    all_classification_metrics: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []
    all_metadata: list[pd.DataFrame] = []
    split_rows: list[pd.DataFrame] = []
    scenario_summaries: list[dict[str, object]] = []

    for scenario in SCENARIOS:
        mask = valid_context_mask(categories, scenario)
        scenario_data = data.loc[mask].copy()
        scenario_categories = categories.loc[mask].copy()
        groups = scenario_group_values(scenario_categories, scenario)

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
        train_positions, test_positions = next(
            splitter.split(scenario_data, y=numeric_series(scenario_data, TARGET), groups=groups)
        )
        train = scenario_data.iloc[train_positions].copy()
        test = scenario_data.iloc[test_positions].copy()
        train_groups = set(groups.iloc[train_positions])
        test_groups = set(groups.iloc[test_positions])
        overlap = train_groups.intersection(test_groups)
        if overlap:
            raise RuntimeError(f"Group leakage found for {scenario}.")

        print(
            f"{scenario}: rows={len(scenario_data):,}, train={len(train):,}, "
            f"test={len(test):,}, train groups={len(train_groups):,}, "
            f"test groups={len(test_groups):,}"
        )

        assignments = pd.DataFrame(
            {
                "Scenario": scenario,
                "testId": scenario_data["testId"].to_numpy(),
                "HoldoutGroup": groups.to_numpy(),
                "DatasetSplit": "train",
            },
            index=scenario_data.index,
        )
        assignments.iloc[test_positions, assignments.columns.get_loc("DatasetSplit")] = "test"
        split_rows.append(assignments.reset_index(drop=True))

        scenario_summaries.append(
            {
                "Scenario": scenario,
                "RowsBeforeContextFilter": int(len(data)),
                "RowsAfterContextFilter": int(len(scenario_data)),
                "RowsExcludedForMissingContext": int((~mask).sum()),
                "TrainRows": int(len(train)),
                "TestRows": int(len(test)),
                "TrainGroups": int(len(train_groups)),
                "TestGroups": int(len(test_groups)),
                "OverlappingGroups": int(len(overlap)),
            }
        )

        for stage, include_day7 in [("Day0_Context", False), ("Day7_FullUpdated", True)]:
            metrics, predictions, metadata = evaluate_regression(
                scenario=scenario,
                stage=stage,
                train=train,
                test=test,
                include_day7=include_day7,
            )
            all_regression_metrics.extend(metrics)
            all_predictions.extend(predictions)
            all_metadata.append(metadata)

            metrics, predictions, metadata = evaluate_classification(
                scenario=scenario,
                stage=stage,
                train=train,
                test=test,
                include_day7=include_day7,
            )
            all_classification_metrics.extend(metrics)
            all_predictions.extend(predictions)
            all_metadata.append(metadata)

    regression_metrics_df = pd.DataFrame(all_regression_metrics).sort_values(
        ["Scenario", "Stage", "MAE"]
    )
    classification_metrics_df = pd.DataFrame(all_classification_metrics).sort_values(
        ["Scenario", "Stage", "AveragePrecision_PR_AUC"],
        ascending=[True, True, False],
    )
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metadata_df = pd.concat(all_metadata, ignore_index=True)
    assignments_df = pd.concat(split_rows, ignore_index=True)
    summary_df = pd.DataFrame(scenario_summaries)

    regression_metrics_df.to_csv(
        OUTPUT_DIR / "unseen_context_regression_metrics.csv", index=False
    )
    classification_metrics_df.to_csv(
        OUTPUT_DIR / "unseen_context_risk_metrics.csv", index=False
    )
    predictions_df.to_csv(OUTPUT_DIR / "unseen_context_predictions.csv", index=False)
    metadata_df.to_csv(OUTPUT_DIR / "unseen_context_encoding_metadata.csv", index=False)
    assignments_df.to_csv(OUTPUT_DIR / "unseen_context_split_assignments.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "unseen_context_split_summary.csv", index=False)

    print("Validation 12 completed.")
    print("\nRegression:")
    print(
        regression_metrics_df[
            ["Scenario", "Stage", "Model", "MAE", "RMSE", "R2"]
        ].to_string(index=False)
    )
    print("\nRisk classification at threshold 0.50:")
    print(
        classification_metrics_df[
            [
                "Scenario",
                "Stage",
                "Model",
                "AveragePrecision_PR_AUC",
                "ROC_AUC",
                "Recall",
                "Precision",
                "FalseNegative",
            ]
        ].to_string(index=False)
    )
    print(f"Output: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
