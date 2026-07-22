from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASEPATH = Path("data/prepared_28_day_standard_cure")
DATA_FILE = BASEPATH / (
    "03_standard_cured_test_level_28_working_data_"
    "drop_rows_drop_columns_1.csv"
)
OUTPUT_PATH = BASEPATH / "results_baseline_models"

BATCH_FEATURE_COLUMNS = [
    "CalcCementContent_lbs_yd3",
    "FlyAshContent_lbs_yd3",
    "SandSSD_lbs_yd3",
    "AggregateSSD_lbs_yd3",
    "calcWCRatio",
    "SandMoisture_percent",
    "AggregateMoisture_percent",
]

# Note: The current code includes waterAdded in addition to
# actual slump and actual air. Remove it from this list if you want
# the field-adjusted comparison to contain only slump and air.
FIELD_FEATURE_COLUMNS = BATCH_FEATURE_COLUMNS + [
    "uwSlump_actual",
    "uwAir_actual",
    "waterAdded",
]

FEATURE_SETS = {
    "BatchOnly": BATCH_FEATURE_COLUMNS,
    "FieldAdjusted": FIELD_FEATURE_COLUMNS,
}

TARGET_COLUMN = "AverageActualStrength28_psi"

GROUP_COLUMNS = [
    "SupplierName",
    "plantNumber",
    "mixNumber",
]

TRACE_COLUMNS = [
    "testId",
    "labNo",
    "SupplierName",
    "plantNumber",
    "mixNumber",
]

# Strength bands are based on the actual 28-day strength.
STRENGTH_BINS = [-np.inf, 3000, 4000, 5000, 6000, np.inf]
STRENGTH_LABELS = [
    "< 3000 psi",
    "3000-3999 psi",
    "4000-4999 psi",
    "5000-5999 psi",
    ">= 6000 psi",
]

RANDOM_STATE = 42
TEST_SIZE = 0.20
LARGEST_ERRORS_PER_MODEL = 5


def validate_required_columns(dataframe: pd.DataFrame) -> None:
    required_columns = set(
        FIELD_FEATURE_COLUMNS
        + [TARGET_COLUMN]
        + GROUP_COLUMNS
    )
    missing_columns = sorted(required_columns - set(dataframe.columns))

    if missing_columns:
        raise ValueError(
            "The input file is missing required columns: "
            + ", ".join(missing_columns)
        )


def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    residuals = np.asarray(y_pred) - np.asarray(y_true)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse),
        "R2": float(r2_score(y_true, y_pred)),
        # Positive: overprediction. Negative: underprediction.
        "MeanBias": float(np.mean(residuals)),
    }


def prepare_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()

    # The target must exist and be positive.
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

    # Initial broad quality check.
    # Review excluded rows separately before making this a final rule.
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

    return (
        dataframe[available_group_columns]
        .fillna("UNKNOWN")
        .astype(str)
        .apply(lambda row: "|".join(row), axis=1)
    )


def create_models() -> dict[str, Pipeline]:
    """Create fresh baseline model pipelines for one feature set."""
    return {
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
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    ),
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
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    ),
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        min_samples_leaf=5,
                        max_features=0.8,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Extra Trees": Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    ),
                ),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=500,
                        min_samples_leaf=5,
                        max_features=1.0,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Hist Gradient Boosting": Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    ),
                ),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        loss="squared_error",
                        learning_rate=0.05,
                        max_iter=300,
                        max_leaf_nodes=15,
                        min_samples_leaf=20,
                        l2_regularization=1.0,
                        early_stopping=False,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    ),
                ),
                (
                    "model",
                    XGBRegressor(
                        objective="reg:squarederror",
                        n_estimators=500,
                        learning_rate=0.03,
                        max_depth=4,
                        min_child_weight=10,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0.0,
                        reg_lambda=1.0,
                        random_state=42,
                        n_jobs=-1,
                        tree_method="hist",
                    ),
                ),
            ]
        ),
    }


def build_prediction_dataframe(
    model_data: pd.DataFrame,
    test_indices: np.ndarray,
    feature_set_name: str,
    model_name: str,
    y_test: pd.Series,
    predictions: np.ndarray,
) -> pd.DataFrame:
    available_trace_columns = [
        column
        for column in TRACE_COLUMNS
        if column in model_data.columns
    ]
    output_columns = list(
        dict.fromkeys(available_trace_columns + FIELD_FEATURE_COLUMNS)
    )

    result = model_data.iloc[test_indices][output_columns].copy()
    result.insert(0, "SourceRowIndex", model_data.index[test_indices])
    result.insert(1, "FeatureSet", feature_set_name)
    result.insert(2, "Model", model_name)

    result["ActualStrength28_psi"] = y_test.to_numpy()
    result["PredictedStrength28_psi"] = predictions
    result["Residual_psi"] = (
        result["PredictedStrength28_psi"]
        - result["ActualStrength28_psi"]
    )
    result["AbsoluteError_psi"] = result["Residual_psi"].abs()
    result["StrengthBand"] = pd.cut(
        result["ActualStrength28_psi"],
        bins=STRENGTH_BINS,
        labels=STRENGTH_LABELS,
        right=False,
    )

    return result


def calculate_strength_band_metrics(
    prediction_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = (
        prediction_dataframe.groupby(
            ["FeatureSet", "Model", "StrengthBand"],
            observed=True,
            sort=False,
        )
        .agg(
            RowCount=("ActualStrength28_psi", "size"),
            MAE=("AbsoluteError_psi", "mean"),
            MeanBias=("Residual_psi", "mean"),
        )
        .reset_index()
    )

    result["StrengthBand"] = result["StrengthBand"].astype(str)
    return result


def get_largest_errors(
    prediction_dataframe: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    return (
        prediction_dataframe.sort_values(
            ["FeatureSet", "Model", "AbsoluteError_psi"],
            ascending=[True, True, False],
        )
        .groupby(
            ["FeatureSet", "Model"],
            sort=False,
            group_keys=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )


def get_model_parameters(
    fitted_models: dict[tuple[str, str], Pipeline],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for (feature_set_name, model_name), pipeline in fitted_models.items():
        estimator = pipeline.named_steps["model"]
        parameters = estimator.get_params(deep=False)

        rows.append(
            {
                "FeatureSet": feature_set_name,
                "Model": model_name,
                "Parameters": repr(parameters),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    dataframe = pd.read_csv(DATA_FILE)
    validate_required_columns(dataframe)
    model_data = prepare_data(dataframe)

    if model_data.empty:
        raise ValueError("No rows remain after prepare_data().")

    print(f"Original rows: {len(dataframe):,}")
    print(f"Model rows:    {len(model_data):,}")

    y = model_data[TARGET_COLUMN]
    groups = create_groups(model_data)

    # Split exactly once. Every feature set and every model uses these
    # same row positions.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_indices, test_indices = next(
        splitter.split(model_data, y, groups=groups)
    )

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    train_groups = set(groups.iloc[train_indices])
    test_groups = set(groups.iloc[test_indices])
    overlapping_groups = train_groups.intersection(test_groups)

    if overlapping_groups:
        raise RuntimeError(
            "Grouped split failed: train and test contain overlapping groups."
        )

    print(f"Training rows:   {len(train_indices):,}")
    print(f"Testing rows:    {len(test_indices):,}")
    print(f"Training groups: {len(train_groups):,}")
    print(f"Testing groups:  {len(test_groups):,}")
    print("Group overlap:   0")

    metric_rows: list[dict[str, float | str | int]] = []
    prediction_frames: list[pd.DataFrame] = []
    fitted_models: dict[tuple[str, str], Pipeline] = {}

    for feature_set_name, feature_columns in FEATURE_SETS.items():
        X = model_data[feature_columns]
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]

        for model_name, model in create_models().items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            metrics = calculate_metrics(y_test, predictions)
            metric_rows.append(
                {
                    "FeatureSet": feature_set_name,
                    "Model": model_name,
                    "FeatureCount": len(feature_columns),
                    "TrainRows": len(train_indices),
                    "TestRows": len(test_indices),
                    **metrics,
                }
            )

            prediction_frames.append(
                build_prediction_dataframe(
                    model_data=model_data,
                    test_indices=test_indices,
                    feature_set_name=feature_set_name,
                    model_name=model_name,
                    y_test=y_test,
                    predictions=predictions,
                )
            )
            fitted_models[(feature_set_name, model_name)] = model

    results_dataframe = (
        pd.DataFrame(metric_rows)
        [
            [
                "FeatureSet",
                "Model",
                "FeatureCount",
                "TrainRows",
                "TestRows",
                "MAE",
                "RMSE",
                "R2",
                "MeanBias",
            ]
        ]
        .sort_values(["MAE", "RMSE"])
        .reset_index(drop=True)
    )

    all_predictions_dataframe = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    strength_band_dataframe = calculate_strength_band_metrics(
        all_predictions_dataframe
    )

    largest_errors_dataframe = get_largest_errors(
        all_predictions_dataframe,
        top_n=LARGEST_ERRORS_PER_MODEL,
    )

    parameter_dataframe = get_model_parameters(fitted_models)

    print("\nModel comparison")
    print(
        results_dataframe.to_string(
            index=False,
            formatters={
                "MAE": "{:,.2f}".format,
                "RMSE": "{:,.2f}".format,
                "R2": "{:.4f}".format,
                "MeanBias": "{:,.2f}".format,
            },
        )
    )

    print("\nStrength-band comparison")
    print(
        strength_band_dataframe.to_string(
            index=False,
            formatters={
                "MAE": "{:,.2f}".format,
                "MeanBias": "{:,.2f}".format,
            },
        )
    )

    largest_error_display_columns = [
        column
        for column in [
            "FeatureSet",
            "Model",
            "SourceRowIndex",
            "testId",
            "labNo",
            "SupplierName",
            "plantNumber",
            "mixNumber",
            "ActualStrength28_psi",
            "PredictedStrength28_psi",
            "Residual_psi",
            "AbsoluteError_psi",
            "StrengthBand",
        ]
        if column in largest_errors_dataframe.columns
    ]

    print(
        f"\nLargest {LARGEST_ERRORS_PER_MODEL} prediction errors "
        "for each model and feature set"
    )
    print(
        largest_errors_dataframe[largest_error_display_columns].to_string(
            index=False,
            formatters={
                "ActualStrength28_psi": "{:,.2f}".format,
                "PredictedStrength28_psi": "{:,.2f}".format,
                "Residual_psi": "{:,.2f}".format,
                "AbsoluteError_psi": "{:,.2f}".format,
            },
        )
    )

    # Select the overall best feature-set/model combination by MAE.
    best_row = results_dataframe.iloc[0]
    best_feature_set_name = str(best_row["FeatureSet"])
    best_model_name = str(best_row["Model"])
    best_model = fitted_models[(best_feature_set_name, best_model_name)]
    best_feature_columns = FEATURE_SETS[best_feature_set_name]
    best_X_test = model_data[best_feature_columns].iloc[test_indices]

    # Permutation importance is applied to the complete pipeline, so the
    # output corresponds to the original input columns rather than the
    # imputer's generated indicator columns.
    importance = permutation_importance(
        best_model,
        best_X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_dataframe = (
        pd.DataFrame(
            {
                "FeatureSet": best_feature_set_name,
                "Model": best_model_name,
                "Feature": best_feature_columns,
                "Importance_MAE_Increase": importance.importances_mean,
                "ImportanceStd": importance.importances_std,
            }
        )
        .sort_values("Importance_MAE_Increase", ascending=False)
        .reset_index(drop=True)
    )

    print(
        f"\nBest model: {best_feature_set_name} / {best_model_name}"
    )
    print("\nPermutation feature importance")
    print(
        importance_dataframe.to_string(
            index=False,
            formatters={
                "Importance_MAE_Increase": "{:,.2f}".format,
                "ImportanceStd": "{:,.2f}".format,
            },
        )
    )

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    results_dataframe.to_csv(
        OUTPUT_PATH / "strength_model_comparison.csv",
        index=False,
    )
    strength_band_dataframe.to_csv(
        OUTPUT_PATH / "strength_band_model_comparison.csv",
        index=False,
    )
    all_predictions_dataframe.to_csv(
        OUTPUT_PATH / "strength_all_model_test_predictions.csv",
        index=False,
    )
    largest_errors_dataframe.to_csv(
        OUTPUT_PATH / "strength_largest_prediction_errors.csv",
        index=False,
    )
    importance_dataframe.to_csv(
        OUTPUT_PATH / "strength_best_model_feature_importance.csv",
        index=False,
    )
    parameter_dataframe.to_csv(
        OUTPUT_PATH / "strength_model_parameters.csv",
        index=False,
    )

    print(f"\nResults saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()