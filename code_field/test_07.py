from __future__ import annotations

import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from field_core_common import (
    FIELD_FEATURES,
    FIELD_FEATURES_7DAYS,
    TARGET,
    TEST_SIZE,
    RANDOM_STATE,
    GROUP_COLUMN,
    RQUESTED_FOLDS,
    read_csv,
    split_data,
    build_models,
    calculate_metrics,
    numeric_frame,
    normalize_category,
    build_context_categories,
    cross_fitted_target_encode,
    save_json
)

# Run with:
#     python code_field/test.py
ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = (
    ROOT_DIR
    / "data"
    / "field_core_outputs"
    / "field_core_clean"
    / "field_core_clean_with_required.csv"
)
OUTPUT_DIR = ROOT_DIR / "data" / "field_core_outputs" / "test_field_features_only"
MODEL_DIR = OUTPUT_DIR / "saved_models"

def field_features_only_test(train_data: pd.DataFrame, test_data: pd.DataFrame) -> int:
    print(f"train_data.shape: {train_data.shape}, test_data.shape: {test_data.shape}")

    X_train = train_data[FIELD_FEATURES]
    X_test = test_data[FIELD_FEATURES]

    # Coerce target to numeric and ensure no NaNs remain after coercion
    y_train = pd.to_numeric(train_data[TARGET], errors="coerce")
    y_test = pd.to_numeric(test_data[TARGET], errors="coerce")

    if y_train.isna().any() or y_test.isna().any():
        raise ValueError("The clean split contains missing target values.")

    print(f"X_train.shape: {X_train.shape}, X_test.shape: {X_test.shape}")
    metric_rows: list[dict[str, object]] = []

    for model_name, model in build_models().items():
        print(f"Traing {model_name} ....")
        start = time.perf_counter()
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        elapsed = time.perf_counter()-start

        metrics = calculate_metrics(y_test.to_numpy(), predicted)
        metric_rows.append(
            {
                "Model": model_name,
                "TrainRows": len(X_train),
                "TestRows": len(X_test),
                "FeatureCount": len(FIELD_FEATURES),
                "TrainingSeconds": elapsed,
                **metrics,
            }      
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        ["MAE", "RMSE"],
        ascending=[True, True],
    )
    print(metrics_df[["Model", "MAE", "RMSE", "R2"]].to_string(index=False))
    return 0

def cross_validation_test(df: pd.DataFrame) -> int:
    # Cross validation using grouped folds by projectId

    # Determine how many splits we can safely use based on unique groups
    unique_group_count = df[GROUP_COLUMN].nunique()
    n_splits = min(RQUESTED_FOLDS, unique_group_count)

    # Create the GroupKFold splitter with a valid integer number of splits
    cv_splitter = GroupKFold(n_splits=n_splits)

    # Prepare numeric features and target
    x = numeric_frame(df, FIELD_FEATURES)
    y = pd.to_numeric(df[TARGET], errors="coerce")
    if y.isna().any():
        raise ValueError("Target column contains non-numeric or missing values after coercion.")

    groups = df[GROUP_COLUMN]

    fold_rows: list[dict[str, object]] = []
    metric_columns = [
        "MAE",
        "MedianAE",
        "RMSE",
        "R2",
        "MeanBias",
        "Within300PsiPercent",
        "Within500PsiPercent",
    ]

    for model_name, model in build_models().items():
        print(f"Cross-validating {model_name} ...")

        # Use the GroupKFold splitter correctly with `groups=` argument
        for fold_number, (train_idx, test_idx) in enumerate(
            cv_splitter.split(x, y, groups=groups), start=1
        ):
            start = time.perf_counter()
            model.fit(x.iloc[train_idx], y.iloc[train_idx])
            predicted = model.predict(x.iloc[test_idx])
            elapsed = time.perf_counter() - start

            fold_metrics = calculate_metrics(y.iloc[test_idx].to_numpy(), predicted)

            fold_rows.append(
                {
                    "FeatureSet": "FIELD_FEATURES",  # constant name for this feature set
                    "Model": model_name,
                    "Fold": fold_number,
                    "TrainRows": len(train_idx),
                    "TestRows": len(test_idx),
                    "TrainGroups": groups.iloc[train_idx].nunique(),
                    "TestGroups": groups.iloc[test_idx].nunique(),
                    "TrainingSeconds": elapsed,
                    **fold_metrics,
                }
            )

    folds = pd.DataFrame(fold_rows)

    # Aggregate metrics across folds
    aggregations: dict[str, list[str]] = {
        metric: ["mean", "std", "min", "max"] for metric in metric_columns
    }
    summary = folds.groupby(["FeatureSet", "Model"], as_index=False).agg(aggregations)

    # Flatten the MultiIndex columns created by aggregation
    summary.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    print(f"Grouping column: {GROUP_COLUMN}")
    print(f"Folds: {n_splits}")
    print(
        summary[
            [
                "Model",
                "MAE_mean",
                "MAE_std",
                "RMSE_mean",
                "R2_mean",
            ]
        ].to_string(index=False)
    )
    return 0

def field_features_7_days_test(train_data: pd.DataFrame, test_data: pd.DataFrame) -> int:

    X7_train = train_data[FIELD_FEATURES_7DAYS]
    X7_test = test_data[FIELD_FEATURES_7DAYS]

    y7_train = train_data[TARGET]
    y7_test = test_data[TARGET]

    print(X7_train.shape, X7_test.shape)

    metric_rows_7: list[dict[str, object]] = []

    for model_name, model in build_models().items():
        print(f"Traing {model_name} ....")
        start = time.perf_counter()
        model.fit(X7_train, y7_train)
        predicted = model.predict(X7_test)
        elapsed = time.perf_counter()-start

        metrics = calculate_metrics(y7_test.to_numpy(), predicted)
        metric_rows_7.append(
            {
                "Model": model_name,
                "TrainRows": len(X7_train),
                "TestRows": len(X7_test),
                "FeatureCount": len(FIELD_FEATURES_7DAYS),
                "TrainingSeconds": elapsed,
                **metrics,
            }      
        )

    metrics_df_7 = pd.DataFrame(metric_rows_7).sort_values(
        ["MAE", "RMSE"],
        ascending=[True, True],
    )
    print(metrics_df_7[["Model", "MAE", "RMSE", "R2"]].to_string(index=False))
    return 0

def context_feature_encoding_test(train_data: pd.DataFrame, test_data: pd.DataFrame) -> int:
    context_sources = {
        "supplier": "supplierId",
        "plant":"plantNumber",
        "mix": "mixNumber",
        "testSubtype": "testSubTypeId",
    }
    X7_train = train_data[FIELD_FEATURES_7DAYS]
    X7_test = test_data[FIELD_FEATURES_7DAYS]
    
    y7_train = train_data[TARGET]
    y7_test = test_data[TARGET]

    train_context = build_context_categories(train_data, context_sources)
    test_context = build_context_categories(test_data, context_sources)
    
    print(train_context.shape, test_context.shape)

    encoded_train, encoded_test, context_metadata = cross_fitted_target_encode(
            train_context,
            test_context,
            y7_train
            
        )
    X7_context_train = pd.concat([X7_train, encoded_train], axis=1)
    X7_context_test = pd.concat([X7_test, encoded_test], axis=1)
    print(f"X7_context_train.shape: {X7_context_train.shape}, X7_context_test.shape: {X7_context_test.shape}")
    feature_sets = {
        "FieldPlusRequired7Days": (X7_train, X7_test),
        "FieldPlusRequired7DaysPlusContext": (X7_context_train, X7_context_test),
    }
    return 0


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = read_csv(INPUT_FILE)
    print(f"df.shape: {df.shape}")
    print(f"{'-' * 10} Field features {'-' * 10}")
    train_data, test_data = split_data(df, GROUP_COLUMN)
    field_features_only_test(train_data, test_data)
    cross_validation_test(df)
    print(f"{'-' * 10} Field features + 7 days {'-' * 10}")
    field_features_7_days_test(train_data, test_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())