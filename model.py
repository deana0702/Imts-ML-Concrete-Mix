from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_FILE = Path(
    "data/prepared_28_day_standard_cure/"
    "03_standard_cured_test_level_28_working_data_drop_rows_drop_columns_1.csv"
)

FEATURE_COLUMNS = [
    "CalcCementContent_lbs_yd3",
    "FlyAshContent_lbs_yd3",
    "SandSSD_lbs_yd3",
    "AggregateSSD_lbs_yd3",
    "calcWCRatio",
    "SandMoisture_percent",
    "AggregateMoisture_percent",
]

TARGET_COLUMN = "AverageActualStrength28_psi"

GROUP_COLUMNS = [
    "SupplierName",
    "plantNumber",
    "mixNumber",
]


def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    rmse = mean_squared_error(
        y_true,
        y_pred,
    ) ** 0.5

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse,
        "R2": r2_score(y_true, y_pred),
    }


def prepare_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()

    # The label must exist and be positive.
    result = result[
        result[TARGET_COLUMN].notna()
        & result[TARGET_COLUMN].gt(0)
    ].copy()

    # Keep approximately 28-day breaks.
    if "AverageActualAgeDays" in result.columns:
        result = result[
            result["AverageActualAgeDays"].between(25, 35)
        ].copy()

    # These values must be present and positive for a regular
    # concrete batch model.
    positive_columns = [
        "CalcCementContent_lbs_yd3",
        "SandSSD_lbs_yd3",
        "AggregateSSD_lbs_yd3",
        "calcWCRatio",
    ]

    for column in positive_columns:
        result = result[
            result[column].notna()
            & result[column].gt(0)
        ].copy()

    # Fly ash = 0 can be valid.
    result = result[
        result["FlyAshContent_lbs_yd3"].notna()
    ].copy()

    # Moisture = 0 can also be valid.
    result = result[
        result["SandMoisture_percent"].notna()
        & result["AggregateMoisture_percent"].notna()
    ].copy()

    # Initial broad quality checks.
    # Review excluded rows separately before making these final rules.
    result = result[
        result["calcWCRatio"].between(0.20, 0.90)
    ].copy()

    return result


def create_groups(dataframe: pd.DataFrame) -> pd.Series:
    available_group_columns = [
        column
        for column in GROUP_COLUMNS
        if column in dataframe.columns
    ]

    if not available_group_columns:
        raise ValueError(
            "At least one group column is required for grouped splitting."
        )

    group_values = (
        dataframe[available_group_columns]
        .fillna("UNKNOWN")
        .astype(str)
        .apply(lambda row: "|".join(row), axis=1)
    )

    return group_values


def main() -> None:
    dataframe = pd.read_csv(DATA_FILE)
    model_data = prepare_data(dataframe)

    print(f"Original rows: {len(dataframe):,}")
    print(f"Model rows:    {len(model_data):,}")

    X = model_data[FEATURE_COLUMNS]
    y = model_data[TARGET_COLUMN]
    groups = create_groups(model_data)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_indices, test_indices = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows:  {len(X_test):,}")

    models = {
        "Dummy Mean": Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "model",
                    DummyRegressor(strategy="mean"),
                ),
            ]
        ),
        "Linear Regression": Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    LinearRegression(),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        min_samples_leaf=5,
                        max_features=0.8,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    results: list[dict[str, float | str]] = []
    fitted_models: dict[str, Pipeline] = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metrics = calculate_metrics(y_test, predictions)
        metrics["Model"] = model_name

        results.append(metrics)
        fitted_models[model_name] = model

    results_dataframe = pd.DataFrame(results)[
        ["Model", "MAE", "RMSE", "R2"]
    ].sort_values("MAE")

    print("\nModel comparison")
    print(results_dataframe.to_string(index=False))

    best_model_name = results_dataframe.iloc[0]["Model"]
    best_model = fitted_models[str(best_model_name)]

    # Permutation importance works with the complete pipeline.
    importance = permutation_importance(
        best_model,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    importance_dataframe = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Importance": importance.importances_mean,
        }
    ).sort_values(
        "Importance",
        ascending=False,
    )

    print(f"\nBest model: {best_model_name}")
    print("\nPermutation feature importance")
    print(importance_dataframe.to_string(index=False))

    predictions = best_model.predict(X_test)

    prediction_results = X_test.copy()
    prediction_results["ActualStrength28_psi"] = y_test
    prediction_results["PredictedStrength28_psi"] = predictions
    prediction_results["AbsoluteError_psi"] = np.abs(
        prediction_results["ActualStrength28_psi"]
        - prediction_results["PredictedStrength28_psi"]
    )

    prediction_results.to_csv(
        "strength_prediction_test_results.csv",
        index=False,
    )

    results_dataframe.to_csv(
        "strength_model_comparison.csv",
        index=False,
    )

    importance_dataframe.to_csv(
        "strength_feature_importance.csv",
        index=False,
    )


if __name__ == "__main__":
    main()