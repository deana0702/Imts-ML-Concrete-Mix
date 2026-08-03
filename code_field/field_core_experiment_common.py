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


# -----------------------------------------------------------------------------
# Shared hardcoded project paths.
# Put every script in the repository root and run it with only:
#     python <script_name>.py
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT_DIR / "data" / "field_core_outputs"
SPLIT_DIR = DATA_ROOT / "field_core_splits"
TRAIN_FILE = SPLIT_DIR / "comparison_train.csv"
TEST_FILE = SPLIT_DIR / "comparison_test.csv"
NEXT_OUTPUT_ROOT = DATA_ROOT / "next_model_experiments"

TARGET = "AverageActualStrength28_psi"
REQUIRED_STRENGTH = "ApplicableSpecifiedStrength28"
RANDOM_STATE = 42

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
]

FIELD_PLUS_REQUIRED_FEATURES = FIELD_FEATURES + [REQUIRED_STRENGTH]

DAY7_AVERAGE_CANDIDATES = [
    "AverageActualStrength7_psi",
    "AverageActualStrength7",
]
DAY7_COUNT_CANDIDATES = [
    "ActualStrength7SpecimenCount",
    "StandardCuredStrength7SpecimenCount",
]

CONTEXT_COLUMN_CANDIDATES = {
    "Supplier": ["SupplierId", "supplierId", "SupplierName"],
    "Plant": ["PlantNumber", "plantNumber"],
    "Mix": ["MixNumber", "mixNumber"],
    "TestSubtype": ["testSubTypeId", "TestSubTypeId", "testSubtypeId"],
}

GROUP_COLUMN_CANDIDATES = ["SplitGroup", "projectId", "projectNo"]


@dataclass(frozen=True)
class ContextSources:
    supplier: str
    plant: str
    mix: str
    test_subtype: str


# -----------------------------------------------------------------------------
# General I/O and validation
# -----------------------------------------------------------------------------
def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            "Run 01_clean_field_core_dataset.py and "
            "02_create_grouped_splits.py first."
        )
    return pd.read_csv(path, low_memory=False)


def load_comparison_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = read_csv(TRAIN_FILE)
    test = read_csv(TEST_FILE)
    require_columns(train, [TARGET, REQUIRED_STRENGTH])
    require_columns(test, [TARGET, REQUIRED_STRENGTH])
    return train, test


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Required columns are missing: {missing}")


def resolve_first_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def resolve_group_column(df: pd.DataFrame) -> str:
    column = resolve_first_column(df, GROUP_COLUMN_CANDIDATES)
    if column is None:
        raise KeyError(
            f"No grouping column found. Expected one of {GROUP_COLUMN_CANDIDATES}."
        )
    return column


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    series = df[column]
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(
        series.astype("string").str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Required model columns are missing: {missing}")
    result = pd.DataFrame(index=df.index)
    for column in columns:
        result[column] = numeric_series(df, column)
    return result


def save_json(data: dict[str, object], path: Path) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


# -----------------------------------------------------------------------------
# Regression models and metrics
# -----------------------------------------------------------------------------
def build_regression_models() -> dict[str, object]:
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


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
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


# -----------------------------------------------------------------------------
# Context preparation and leakage-controlled target encoding
# -----------------------------------------------------------------------------
def normalize_category(series: pd.Series) -> pd.Series:
    values = series.astype("string").fillna("__MISSING__")
    values = values.str.strip().str.upper()
    values = values.str.replace(r"\s+", " ", regex=True)
    values = values.replace("", "__MISSING__")
    return values


def resolve_context_sources(df: pd.DataFrame) -> ContextSources:
    resolved: dict[str, str] = {}
    for logical_name, candidates in CONTEXT_COLUMN_CANDIDATES.items():
        column = resolve_first_column(df, candidates)
        if column is None:
            raise KeyError(
                f"Context field '{logical_name}' was not found. "
                f"Expected one of {candidates}."
            )
        resolved[logical_name] = column
    return ContextSources(
        supplier=resolved["Supplier"],
        plant=resolved["Plant"],
        mix=resolved["Mix"],
        test_subtype=resolved["TestSubtype"],
    )


def build_context_categories(
    df: pd.DataFrame,
    sources: ContextSources,
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


def cross_fitted_target_encode(
    train_categories: pd.DataFrame,
    test_categories: pd.DataFrame,
    y_train: pd.Series,
    train_groups: pd.Series,
    *,
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

    groups = train_groups.astype("string").fillna("__MISSING_GROUP__")
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

    return encoded_train, encoded_test, pd.DataFrame(metadata_rows)


# -----------------------------------------------------------------------------
# Seven-day feature preparation
# -----------------------------------------------------------------------------
def add_day7_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | None]]:
    result = df.copy()
    average_column = resolve_first_column(result, DAY7_AVERAGE_CANDIDATES)
    count_column = resolve_first_column(result, DAY7_COUNT_CANDIDATES)
    if average_column is None:
        raise KeyError(
            "No 7-day average-strength column was found. Expected one of "
            f"{DAY7_AVERAGE_CANDIDATES}."
        )

    result["Day7AverageStrength_psi"] = numeric_series(result, average_column)
    if count_column is None:
        result["Day7SpecimenCount"] = np.nan
    else:
        result["Day7SpecimenCount"] = numeric_series(result, count_column)

    required = numeric_series(result, REQUIRED_STRENGTH)
    result["Day7MarginToRequired_psi"] = (
        result["Day7AverageStrength_psi"] - required
    )
    result["Day7ToRequiredRatio"] = (
        result["Day7AverageStrength_psi"] / required.replace(0, np.nan)
    )

    metadata = {
        "source_average_column": average_column,
        "source_count_column": count_column,
    }
    return result, metadata


DAY7_FEATURES = [
    "Day7AverageStrength_psi",
    "Day7SpecimenCount",
    "Day7MarginToRequired_psi",
    "Day7ToRequiredRatio",
]


# -----------------------------------------------------------------------------
# Classification models and metrics
# -----------------------------------------------------------------------------
def build_classification_models() -> dict[str, object]:
    return {
        "DummyPrior": DummyClassifier(strategy="prior"),
        "LogisticRegression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2_000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=5,
                        max_features=0.8,
                        class_weight="balanced_subsample",
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
                    HistGradientBoostingClassifier(
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


def fit_classifier(
    model_name: str,
    model: object,
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> object:
    if model_name == "HistGradientBoosting":
        weights = compute_sample_weight(class_weight="balanced", y=y_train)
        model.fit(x_train, y_train, model__sample_weight=weights)
    else:
        model.fit(x_train, y_train)
    return model


def positive_probability(model: object, x: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(x)
    classes = list(model.classes_)
    positive_index = classes.index(1)
    return probabilities[:, positive_index]


def classification_metrics(
    actual: np.ndarray,
    probability: np.ndarray,
    threshold: float = 0.50,
) -> dict[str, float | int]:
    predicted = (probability >= threshold).astype(int)
    true_positive = int(np.sum((actual == 1) & (predicted == 1)))
    true_negative = int(np.sum((actual == 0) & (predicted == 0)))
    false_positive = int(np.sum((actual == 0) & (predicted == 1)))
    false_negative = int(np.sum((actual == 1) & (predicted == 0)))

    roc_auc = np.nan
    if len(np.unique(actual)) == 2:
        roc_auc = float(roc_auc_score(actual, probability))

    return {
        "Threshold": float(threshold),
        "FailureRatePercent": float(np.mean(actual) * 100.0),
        "Precision": float(precision_score(actual, predicted, zero_division=0)),
        "Recall": float(recall_score(actual, predicted, zero_division=0)),
        "F1": float(f1_score(actual, predicted, zero_division=0)),
        "AveragePrecision_PR_AUC": float(average_precision_score(actual, probability)),
        "ROC_AUC": roc_auc,
        "BrierScore": float(brier_score_loss(actual, probability)),
        "TruePositive": true_positive,
        "TrueNegative": true_negative,
        "FalsePositive": false_positive,
        "FalseNegative": false_negative,
    }


def threshold_metrics_table(
    actual: np.ndarray,
    probability: np.ndarray,
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.arange(0.05, 0.55, 0.05)
    rows = [classification_metrics(actual, probability, float(t)) for t in thresholds]
    return pd.DataFrame(rows)
