from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import re
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import GroupShuffleSplit

# Feature configuration (unchanged)
FIELD_FEATURES = [
    "EffectiveSlump_in",
    "EffectiveAir_percent",
    "EffectiveUnitWeight_lb_ft3",
    "EffectiveConcreteTemp_F",
    "AmbientTemp_F",
    "WaterAdded_gal_per_yd3",
    "BatchToSampleMinutes",
    "BatchToCastMinutes",
    "HasWaterAdded",
    "HasAnyAfterSPMeasurement",
    "ApplicableSpecifiedStrength28"
]
FIELD_FEATURES_7DAYS = FIELD_FEATURES + ["AverageActualStrength7_psi", "ActualStrength7SpecimenCount"]

TARGET = "AverageActualStrength28_psi"
TEST_SIZE = 0.20
RANDOM_STATE = 42
GROUP_COLUMN = "projectId"
RQUESTED_FOLDS = 5

def _smoothed_mapping(
    category: pd.Series,
    target: pd.Series,
    global_mean: float,
    smoothing: float,
) -> pd.Series:
    stats = pd.DataFrame({"category": category, "target": target}).groupby(
        "category",
        dropna=False,
    )["target"].agg(["sum", "count"])
    return (stats["sum"] + smoothing * global_mean) / (
        stats["count"] + smoothing
    )

def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\\n"
            "Run code_field/01_clean_field_core_dataset.py first."
        )
    return pd.read_csv(path, low_memory=False)

def split_data(df: pd.DataFrame, group_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Ensure the grouping column exists
    if group_column not in df.columns:
        raise KeyError(f"Grouping column '{group_column}' not found in the DataFrame.")

    # Drop rows where group or target is missing
    clean_df = df.dropna(subset=[GROUP_COLUMN, TARGET]).copy()
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # Use a single grouping column without NaNs
    train_positions, test_positions = next(
        splitter.split(clean_df, y=clean_df[TARGET], groups=clean_df[group_column])
    )

    train_data = clean_df.iloc[train_positions].copy()
    test_data = clean_df.iloc[test_positions].copy()
    return train_data, test_data

def build_models() -> dict[str, object]:
    return {
        "DummyMean": DummyRegressor(strategy="mean"),
        "Ridge": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=5,
                        max_features=0.8,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=300,
                        learning_rate=0.05,
                        max_leaf_nodes=31,
                        min_samples_leaf=20,
                        l2_regularization=1.0,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - actual
    absolute_error = np.abs(residual)
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "MedianAE": float(median_absolute_error(actual, predicted)),
        "RMSE": float(math.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
        "MeanBias": float(np.mean(residual)),
        "Within300PsiPercent": float(np.mean(absolute_error <= 300.0) * 100.0),
        "Within500PsiPercent": float(np.mean(absolute_error <= 500.0) * 100.0),
    }

def numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Required model columns are missing: {missing}")

    result = pd.DataFrame(index=df.index)
    for column in columns:
        result[column] = pd.to_numeric(df[column], errors="coerce")
    return result

def normalize_category(series: pd.Series) -> pd.Series:
    values = series.astype("string").fillna("__MISSING__")
    values = values.str.strip().str.upper()
    values = values.str.replace(r"\s+", " ", regex=True)
    values = values.replace("", "__MISSING__")
    return values
def build_context_categories(
    df: pd.DataFrame,
    sources: dict,
) -> pd.DataFrame:
    context = pd.DataFrame(index=df.index)
    context["SupplierCategory"] = normalize_category(df[sources.supplier])
    context["PlantCategory"] = normalize_category(df[sources.plant])
    context["MixCategory"] = normalize_category(df[sources.mix])
    context["TestSubtypeCategory"] = normalize_category(df[sources.test_subtype])

    context["SupplierPlantCategory"] = (
        context["SupplierCategory"] + "|" + context["PlantCategory"]
    )
    context["SupplierPlantMixCategory"] = (
        context["SupplierCategory"]
        + "|"
        + context["PlantCategory"]
        + "|"
        + context["MixCategory"]
    )
    return context

def cross_fitted_target_encode(
    train_categories: pd.DataFrame,
    test_categories: pd.DataFrame,
    y_train: pd.Series,
    smoothing: float = 20.0,
    max_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create numeric context features without using each training row's own target.

    For every categorical field:
      * training target means are created out-of-fold by project group;
      * test mappings are learned from the complete training set only;
      * unknown test categories fall back to the training global mean;
      * log-frequency features are also produced.
    """
    y_train = pd.to_numeric(y_train, errors="coerce")
    if y_train.isna().any():
        raise ValueError("Target encoding received missing training targets.")

    groups = train_categories[GROUP_COLUMN].astype("string").fillna("__MISSING_GROUP__")
    unique_groups = int(groups.nunique())
    n_splits = min(max_splits, unique_groups)
    if n_splits < 2:
        raise ValueError("At least two project groups are required for target encoding.")

    splitter = GroupKFold(n_splits=n_splits)
    global_mean = float(y_train.mean())

    encoded_train = pd.DataFrame(index=train_categories.index)
    encoded_test = pd.DataFrame(index=test_categories.index)
    metadata_rows: list[dict[str, object]] = []

    for column in train_categories.columns:
        train_values = normalize_category(train_categories[column])
        test_values = normalize_category(test_categories[column])
        oof = pd.Series(np.nan, index=train_categories.index, dtype=float)

        for fit_positions, validation_positions in splitter.split(
            train_categories,
            y_train,
            groups,
        ):
            fit_index = train_categories.index[fit_positions]
            validation_index = train_categories.index[validation_positions]
            mapping = _smoothed_mapping(
                train_values.loc[fit_index],
                y_train.loc[fit_index],
                global_mean,
                smoothing,
            )
            oof.loc[validation_index] = (
                train_values.loc[validation_index].map(mapping).fillna(global_mean)
            )

        full_mapping = _smoothed_mapping(
            train_values,
            y_train,
            global_mean,
            smoothing,
        )
        train_counts = train_values.value_counts(dropna=False)

        encoded_train[f"{column}_TargetMean"] = oof.fillna(global_mean)
        encoded_test[f"{column}_TargetMean"] = (
            test_values.map(full_mapping).fillna(global_mean)
        )
        encoded_train[f"{column}_LogCount"] = np.log1p(
            train_values.map(train_counts).fillna(0).astype(float)
        )
        encoded_test[f"{column}_LogCount"] = np.log1p(
            test_values.map(train_counts).fillna(0).astype(float)
        )
        encoded_test[f"{column}_Unknown"] = (~test_values.isin(full_mapping.index)).astype(
            int
        )
        encoded_train[f"{column}_Unknown"] = 0

        metadata_rows.append(
            {
                "ContextColumn": column,
                "TrainUniqueCategories": int(train_values.nunique()),
                "TestUniqueCategories": int(test_values.nunique()),
                "UnknownTestRows": int((~test_values.isin(full_mapping.index)).sum()),
                "UnknownTestPercent": float(
                    (~test_values.isin(full_mapping.index)).mean() * 100.0
                ),
                "Smoothing": smoothing,
                "GroupFolds": n_splits,
            }
        )

    return encoded_train, encoded_test

def save_json(data: dict[str, object], path: Path) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
)