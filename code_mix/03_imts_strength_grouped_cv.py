from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


BASEPATH = Path("data/prepared_28_day_standard_cure")
# DATA_FILE = BASEPATH / (
#     "03_standard_cured_test_level_28_working_data_"
#     "drop_rows_drop_columns_1.csv"
# )
ROOT_DIR = Path(__file__).parent.parent
DATA_FILE = ROOT_DIR / (
    "data/concrete_us_data_v2.csv"
)
OUTPUT_PATH = BASEPATH / "results_grouped_cross_validation"

BATCH_FEATURE_COLUMNS = [
    "CalcCementContent_lbs_yd3",
    "FlyAshContent_lbs_yd3",
    "SandSSD_lbs_yd3",
    "AggregateSSD_lbs_yd3",
    "calcWCRatio",
    "SandMoisture_percent",
    "AggregateMoisture_percent",
]

# This cross-validation compares the three strongest models using the
# field-adjusted feature set that produced the best holdout results.
FIELD_FEATURE_COLUMNS = BATCH_FEATURE_COLUMNS + [
    "uwSlump_actual",
    "uwAir_actual",
    "WaterAdded_gal_per_yd3",
]

FEATURE_SET_NAME = "FieldAdjusted"
FEATURE_COLUMNS = FIELD_FEATURE_COLUMNS
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
REQUESTED_CV_SPLITS = 5
LARGEST_ERRORS_PER_MODEL = 10


def validate_required_columns(dataframe: pd.DataFrame) -> None:
    required_columns = set(
        FEATURE_COLUMNS + [TARGET_COLUMN] + GROUP_COLUMNS
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
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    residuals = y_pred_array - y_true_array

    return {
        "MAE": float(mean_absolute_error(y_true_array, y_pred_array)),
        "MedianAE": float(
            median_absolute_error(y_true_array, y_pred_array)
        ),
        "RMSE": float(
            mean_squared_error(y_true_array, y_pred_array) ** 0.5
        ),
        "R2": float(r2_score(y_true_array, y_pred_array)),
        # Positive means overprediction; negative means underprediction.
        "MeanBias": float(np.mean(residuals)),
    }


def prepare_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()

    result = result[
        result[TARGET_COLUMN].notna()
        & result[TARGET_COLUMN].gt(0)
    ].copy()

    if "AverageActualAgeDays" in result.columns:
        result = result[
            result["AverageActualAgeDays"].between(25, 35)
        ].copy()

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

    # Zero fly ash is valid, but a missing value is not used here.
    result = result[
        result["FlyAshContent_lbs_yd3"].notna()
    ].copy()

    # Zero moisture can be valid.
    result = result[
        result["SandMoisture_percent"].notna()
        & result["AggregateMoisture_percent"].notna()
    ].copy()

    result = result[
        result["calcWCRatio"].between(0.20, 0.90)
    ].copy()

    return result


def create_groups(dataframe: pd.DataFrame) -> pd.Series:
    return (
        dataframe[GROUP_COLUMNS]
        .fillna("UNKNOWN")
        .astype(str)
        .apply(lambda row: "|".join(row), axis=1)
    )


def create_candidate_models() -> dict[str, Pipeline]:
    """Return fresh baseline pipelines for the three candidate models."""
    return {
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
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        tree_method="hist",
                    ),
                ),
            ]
        ),
    }


def validate_group_separation(
    train_groups: pd.Series,
    validation_groups: pd.Series,
    context: str,
) -> None:
    overlap = set(train_groups).intersection(set(validation_groups))
    if overlap:
        raise RuntimeError(
            f"Group leakage detected in {context}: "
            f"{len(overlap)} overlapping groups."
        )


def build_prediction_frame(
    source_data: pd.DataFrame,
    row_positions: np.ndarray,
    model_name: str,
    dataset_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    fold_number: int | None = None,
) -> pd.DataFrame:
    available_trace_columns = [
        column
        for column in TRACE_COLUMNS
        if column in source_data.columns
    ]
    output_columns = list(
        dict.fromkeys(available_trace_columns + FEATURE_COLUMNS)
    )

    result = source_data.iloc[row_positions][output_columns].copy()
    result.insert(0, "SourceRowIndex", source_data.index[row_positions])
    result.insert(1, "FeatureSet", FEATURE_SET_NAME)
    result.insert(2, "Model", model_name)
    result.insert(3, "Dataset", dataset_name)

    if fold_number is not None:
        result.insert(4, "Fold", fold_number)

    result["ActualStrength28_psi"] = y_true.to_numpy()
    result["PredictedStrength28_psi"] = y_pred
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


def run_grouped_cross_validation(
    model_data: pd.DataFrame,
    outer_train_positions: np.ndarray,
    groups: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train_outer = model_data[FEATURE_COLUMNS].iloc[
        outer_train_positions
    ].reset_index(drop=True)
    y_train_outer = model_data[TARGET_COLUMN].iloc[
        outer_train_positions
    ].reset_index(drop=True)
    groups_train_outer = groups.iloc[
        outer_train_positions
    ].reset_index(drop=True)

    unique_train_group_count = groups_train_outer.nunique()
    n_splits = min(REQUESTED_CV_SPLITS, unique_train_group_count)

    if n_splits < 2:
        raise ValueError(
            "At least two unique training groups are required for "
            "GroupKFold cross-validation."
        )

    print(f"Grouped CV folds: {n_splits}")

    cross_validator = GroupKFold(n_splits=n_splits)
    models = create_candidate_models()

    fold_metric_rows: list[dict[str, float | int | str]] = []
    oof_prediction_frames: list[pd.DataFrame] = []

    for fold_number, (cv_train_positions, cv_valid_positions) in enumerate(
        cross_validator.split(
            X_train_outer,
            y_train_outer,
            groups=groups_train_outer,
        ),
        start=1,
    ):
        cv_train_groups = groups_train_outer.iloc[cv_train_positions]
        cv_valid_groups = groups_train_outer.iloc[cv_valid_positions]

        validate_group_separation(
            cv_train_groups,
            cv_valid_groups,
            context=f"cross-validation fold {fold_number}",
        )

        print(
            f"Fold {fold_number}: "
            f"train rows={len(cv_train_positions):,}, "
            f"validation rows={len(cv_valid_positions):,}, "
            f"train groups={cv_train_groups.nunique():,}, "
            f"validation groups={cv_valid_groups.nunique():,}"
        )

        X_cv_train = X_train_outer.iloc[cv_train_positions]
        X_cv_valid = X_train_outer.iloc[cv_valid_positions]
        y_cv_train = y_train_outer.iloc[cv_train_positions]
        y_cv_valid = y_train_outer.iloc[cv_valid_positions]

        # Convert fold-relative validation positions back to model_data
        # positions for traceable prediction output.
        original_valid_positions = outer_train_positions[
            cv_valid_positions
        ]

        for model_name, model_template in models.items():
            model = clone(model_template)
            model.fit(X_cv_train, y_cv_train)
            predictions = model.predict(X_cv_valid)
            metrics = calculate_metrics(y_cv_valid, predictions)

            fold_metric_rows.append(
                {
                    "FeatureSet": FEATURE_SET_NAME,
                    "Model": model_name,
                    "Fold": fold_number,
                    "TrainRows": len(cv_train_positions),
                    "ValidationRows": len(cv_valid_positions),
                    "TrainGroups": cv_train_groups.nunique(),
                    "ValidationGroups": cv_valid_groups.nunique(),
                    **metrics,
                }
            )

            oof_prediction_frames.append(
                build_prediction_frame(
                    source_data=model_data,
                    row_positions=original_valid_positions,
                    model_name=model_name,
                    dataset_name="CrossValidationOOF",
                    fold_number=fold_number,
                    y_true=y_cv_valid,
                    y_pred=predictions,
                )
            )

    fold_results = pd.DataFrame(fold_metric_rows).sort_values(
        ["Model", "Fold"]
    )
    oof_predictions = pd.concat(
        oof_prediction_frames,
        ignore_index=True,
    )

    summary = (
        fold_results.groupby(["FeatureSet", "Model"], as_index=False)
        .agg(
            FoldCount=("Fold", "count"),
            MeanCV_MAE=("MAE", "mean"),
            StdCV_MAE=("MAE", "std"),
            BestFold_MAE=("MAE", "min"),
            WorstFold_MAE=("MAE", "max"),
            MeanCV_MedianAE=("MedianAE", "mean"),
            StdCV_MedianAE=("MedianAE", "std"),
            MeanCV_RMSE=("RMSE", "mean"),
            StdCV_RMSE=("RMSE", "std"),
            MeanCV_R2=("R2", "mean"),
            StdCV_R2=("R2", "std"),
            MeanCV_MeanBias=("MeanBias", "mean"),
            StdCV_MeanBias=("MeanBias", "std"),
        )
    )

    summary["MAE_Range"] = (
        summary["WorstFold_MAE"] - summary["BestFold_MAE"]
    )
    summary["MAE_CV_Percent"] = np.where(
        summary["MeanCV_MAE"].ne(0),
        summary["StdCV_MAE"] / summary["MeanCV_MAE"] * 100,
        np.nan,
    )

    summary = summary.sort_values(
        ["MeanCV_MAE", "StdCV_MAE", "MeanCV_RMSE"]
    ).reset_index(drop=True)
    summary.insert(0, "CV_Rank", np.arange(1, len(summary) + 1))

    return fold_results, summary, oof_predictions


def evaluate_on_fixed_holdout(
    model_data: pd.DataFrame,
    train_positions: np.ndarray,
    test_positions: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Pipeline]]:
    X = model_data[FEATURE_COLUMNS]
    y = model_data[TARGET_COLUMN]

    X_train = X.iloc[train_positions]
    X_test = X.iloc[test_positions]
    y_train = y.iloc[train_positions]
    y_test = y.iloc[test_positions]

    metric_rows: list[dict[str, float | int | str]] = []
    prediction_frames: list[pd.DataFrame] = []
    fitted_models: dict[str, Pipeline] = {}

    for model_name, model in create_candidate_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = calculate_metrics(y_test, predictions)

        metric_rows.append(
            {
                "FeatureSet": FEATURE_SET_NAME,
                "Model": model_name,
                "FeatureCount": len(FEATURE_COLUMNS),
                "TrainRows": len(train_positions),
                "TestRows": len(test_positions),
                **metrics,
            }
        )

        prediction_frames.append(
            build_prediction_frame(
                source_data=model_data,
                row_positions=test_positions,
                model_name=model_name,
                dataset_name="FixedHoldoutTest",
                y_true=y_test,
                y_pred=predictions,
            )
        )
        fitted_models[model_name] = model

    results = pd.DataFrame(metric_rows).sort_values(
        ["MAE", "RMSE"]
    ).reset_index(drop=True)
    results.insert(0, "HoldoutRank", np.arange(1, len(results) + 1))

    predictions = pd.concat(prediction_frames, ignore_index=True)
    return results, predictions, fitted_models


def calculate_strength_band_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    result = (
        predictions.groupby(
            ["Dataset", "FeatureSet", "Model", "StrengthBand"],
            observed=True,
            sort=False,
        )
        .agg(
            RowCount=("ActualStrength28_psi", "size"),
            MAE=("AbsoluteError_psi", "mean"),
            MedianAE=("AbsoluteError_psi", "median"),
            RMSE=(
                "Residual_psi",
                lambda values: float(
                    np.sqrt(np.mean(np.square(values)))
                ),
            ),
            MeanBias=("Residual_psi", "mean"),
        )
        .reset_index()
    )
    result["StrengthBand"] = result["StrengthBand"].astype(str)
    return result


def get_largest_errors(
    predictions: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    return (
        predictions.sort_values(
            ["Dataset", "Model", "AbsoluteError_psi"],
            ascending=[True, True, False],
        )
        .groupby(
            ["Dataset", "Model"],
            sort=False,
            group_keys=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )


def get_model_parameters(
    models: dict[str, Pipeline],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for model_name, pipeline in models.items():
        estimator = pipeline.named_steps["model"]
        rows.append(
            {
                "FeatureSet": FEATURE_SET_NAME,
                "Model": model_name,
                "Parameters": repr(estimator.get_params(deep=False)),
            }
        )

    return pd.DataFrame(rows)


def print_dataframe(
    title: str,
    dataframe: pd.DataFrame,
) -> None:
    print(f"\n{title}")
    print(
        dataframe.to_string(
            index=False,
            formatters={
                column: "{:,.2f}".format
                for column in [
                    "MAE",
                    "MedianAE",
                    "RMSE",
                    "MeanBias",
                    "MeanCV_MAE",
                    "StdCV_MAE",
                    "BestFold_MAE",
                    "WorstFold_MAE",
                    "MeanCV_MedianAE",
                    "StdCV_MedianAE",
                    "MeanCV_RMSE",
                    "StdCV_RMSE",
                    "MeanCV_MeanBias",
                    "StdCV_MeanBias",
                    "MAE_Range",
                    "MAE_CV_Percent",
                ]
                if column in dataframe.columns
            }
            | {
                column: "{:.4f}".format
                for column in [
                    "R2",
                    "MeanCV_R2",
                    "StdCV_R2",
                ]
                if column in dataframe.columns
            },
        )
    )


def main() -> None:
    dataframe = pd.read_csv(DATA_FILE)
    validate_required_columns(dataframe)
    model_data = prepare_data(dataframe)

    if model_data.empty:
        raise ValueError("No rows remain after prepare_data().")

    y = model_data[TARGET_COLUMN]
    groups = create_groups(model_data)

    print(f"Original rows: {len(dataframe):,}")
    print(f"Model rows:    {len(model_data):,}")
    print(f"Unique groups: {groups.nunique():,}")
    print(f"Feature set:   {FEATURE_SET_NAME}")
    print(f"Feature count: {len(FEATURE_COLUMNS)}")

    # Outer grouped holdout split. This is exactly the same split design as
    # the earlier baseline comparison and is not used to select CV winners.
    outer_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_positions, test_positions = next(
        outer_splitter.split(model_data, y, groups=groups)
    )

    train_groups = groups.iloc[train_positions]
    test_groups = groups.iloc[test_positions]
    validate_group_separation(
        train_groups,
        test_groups,
        context="fixed outer train/test split",
    )

    print(f"Outer training rows:   {len(train_positions):,}")
    print(f"Outer testing rows:    {len(test_positions):,}")
    print(f"Outer training groups: {train_groups.nunique():,}")
    print(f"Outer testing groups:  {test_groups.nunique():,}")
    print("Outer group overlap:   0")

    (
        cv_fold_results,
        cv_summary,
        cv_oof_predictions,
    ) = run_grouped_cross_validation(
        model_data=model_data,
        outer_train_positions=train_positions,
        groups=groups,
    )

    (
        holdout_results,
        holdout_predictions,
        fitted_holdout_models,
    ) = evaluate_on_fixed_holdout(
        model_data=model_data,
        train_positions=train_positions,
        test_positions=test_positions,
    )

    all_predictions = pd.concat(
        [cv_oof_predictions, holdout_predictions],
        ignore_index=True,
    )
    strength_band_results = calculate_strength_band_metrics(
        all_predictions
    )
    largest_errors = get_largest_errors(
        all_predictions,
        top_n=LARGEST_ERRORS_PER_MODEL,
    )
    model_parameters = get_model_parameters(fitted_holdout_models)

    print_dataframe("Grouped CV fold results", cv_fold_results)
    print_dataframe("Grouped CV stability summary", cv_summary)
    print_dataframe("Fixed holdout results", holdout_results)
    print_dataframe("Strength-band results", strength_band_results)

    largest_error_display_columns = [
        column
        for column in [
            "Dataset",
            "Model",
            "Fold",
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
        if column in largest_errors.columns
    ]
    print_dataframe(
        f"Largest {LARGEST_ERRORS_PER_MODEL} errors per model and dataset",
        largest_errors[largest_error_display_columns],
    )

    cv_winner = cv_summary.iloc[0]
    holdout_winner = holdout_results.iloc[0]

    print(
        "\nCV winner by mean MAE: "
        f"{cv_winner['Model']} "
        f"(Mean MAE={cv_winner['MeanCV_MAE']:,.2f}, "
        f"Std MAE={cv_winner['StdCV_MAE']:,.2f})"
    )
    print(
        "Holdout winner by MAE: "
        f"{holdout_winner['Model']} "
        f"(MAE={holdout_winner['MAE']:,.2f}, "
        f"RMSE={holdout_winner['RMSE']:,.2f}, "
        f"R2={holdout_winner['R2']:.4f})"
    )

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    cv_fold_results.to_csv(
        OUTPUT_PATH / "grouped_cv_fold_results.csv",
        index=False,
    )
    cv_summary.to_csv(
        OUTPUT_PATH / "grouped_cv_stability_summary.csv",
        index=False,
    )
    cv_oof_predictions.to_csv(
        OUTPUT_PATH / "grouped_cv_oof_predictions.csv",
        index=False,
    )
    holdout_results.to_csv(
        OUTPUT_PATH / "fixed_holdout_results.csv",
        index=False,
    )
    holdout_predictions.to_csv(
        OUTPUT_PATH / "fixed_holdout_predictions.csv",
        index=False,
    )
    strength_band_results.to_csv(
        OUTPUT_PATH / "cv_and_holdout_strength_band_results.csv",
        index=False,
    )
    largest_errors.to_csv(
        OUTPUT_PATH / "cv_and_holdout_largest_errors.csv",
        index=False,
    )
    model_parameters.to_csv(
        OUTPUT_PATH / "candidate_model_parameters.csv",
        index=False,
    )

    print(f"\nResults saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()