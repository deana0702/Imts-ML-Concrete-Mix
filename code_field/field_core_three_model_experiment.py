from __future__ import annotations

"""
IMTS Field Core - Consolidated 28-Day Strength Experiment

This single file replaces the experiment logic that was previously split across:
    field_core_experiment_common.py
    06_compare_required_only.py
    07_compare_day0_context.py
    08_compare_day7_updated.py

It intentionally DOES NOT run a Field-Only model.

The three feature sets are:

1) Day0_FieldPlusRequired
   - Field measurements available at/near placement
   - Applicable 28-day required/design strength

2) Day7_FieldPlusRequired
   - Day-0 features
   - 7-day standard-cured strength information

3) Full_ContextPlusDay7
   - Day-0 features
   - Supplier / Plant / Mix context
   - 7-day strength information

For a fair comparison, all three feature sets are evaluated on the SAME rows:
only tests with a valid 7-day strength result are included in the comparison.

Run with:
    python code_field/field_core_three_model_experiment.py

No command-line arguments are required.
"""

from dataclasses import dataclass
from pathlib import Path
import json
import math
import re
import time
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent


def find_repo_root() -> Path:
    """
    Find the repository root whether this script is stored in:
        <repo>/
    or:
        <repo>/code_field/
    """
    for candidate in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        if (candidate / "data").exists():
            return candidate
    return SCRIPT_DIR.parent


REPO_ROOT = find_repo_root()
DATA_DIR = REPO_ROOT / "data"
FIELD_CORE_OUTPUT_ROOT = DATA_DIR / "field_core_outputs"

INPUT_CANDIDATES = [
    FIELD_CORE_OUTPUT_ROOT / "field_core_clean" / "field_core_clean_with_required.csv",
    DATA_DIR / "field_core_clean" / "field_core_clean_with_required.csv",
]

OUTPUT_DIR = FIELD_CORE_OUTPUT_ROOT / "consolidated_three_model_experiment"
MODEL_DIR = OUTPUT_DIR / "saved_models"

TARGET = "AverageActualStrength28_psi"
REQUIRED_STRENGTH = "ApplicableSpecifiedStrength28"

TEST_SIZE = 0.20
RANDOM_STATE = 42
TARGET_ENCODING_SMOOTHING = 20.0
TARGET_ENCODING_FOLDS = 5


# -----------------------------------------------------------------------------
# Day-0 field features
# -----------------------------------------------------------------------------
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

DAY0_FEATURES = FIELD_FEATURES + [REQUIRED_STRENGTH]


# -----------------------------------------------------------------------------
# 7-day source columns
# -----------------------------------------------------------------------------
DAY7_AVERAGE_CANDIDATES = [
    "AverageActualStrength7_psi",
    "AverageActualStrength7",
]

DAY7_COUNT_CANDIDATES = [
    "ActualStrength7SpecimenCount",
    "StandardCuredStrength7SpecimenCount",
]


# -----------------------------------------------------------------------------
# Supplier / Plant / Mix context only
# Test subtype is intentionally excluded from this consolidated experiment.
# -----------------------------------------------------------------------------
CONTEXT_COLUMN_CANDIDATES = {
    "Supplier": ["SupplierId", "supplierId", "SupplierName", "supplierName"],
    "Plant": ["PlantNumber", "plantNumber", "PlantNo", "plantNo"],
    "Mix": ["MixNumber", "mixNumber", "MixNo", "mixNo"],
}

GROUP_COLUMN_CANDIDATES = [
    "projectId",
    "projectNo",
    "ProjectId",
    "ProjectNo",
]


@dataclass(frozen=True)
class ContextSources:
    supplier: str
    plant: str
    mix: str


# =============================================================================
# 2. GENERAL HELPERS
# =============================================================================

def first_existing_path(paths: Iterable[Path]) -> Path:
    paths = list(paths)
    for path in paths:
        if path.exists():
            return path

    expected = "\n".join(f"  - {path}" for path in paths)
    raise FileNotFoundError(
        "Could not find the cleaned Field Core dataset.\n"
        "Expected one of:\n"
        f"{expected}\n\n"
        "Run the cleaning step first."
    )


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue

    raise UnicodeError(f"Could not determine CSV encoding for: {path}")


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Required columns are missing: {missing}")


def resolve_first_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)

    series = df[column]

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    return pd.to_numeric(
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def numeric_frame(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    require_columns(df, columns)

    result = pd.DataFrame(index=df.index)

    for column in columns:
        result[column] = numeric_series(df, column)

    return result


def save_json(data: dict[str, object], path: Path) -> None:
    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )


# =============================================================================
# 3. PROJECT GROUP SPLIT
# =============================================================================

def resolve_group_column(df: pd.DataFrame) -> str:
    column = resolve_first_column(df, GROUP_COLUMN_CANDIDATES)

    if column is None:
        raise KeyError(
            "No project grouping column was found. "
            f"Expected one of: {GROUP_COLUMN_CANDIDATES}"
        )

    return column


def make_project_groups(
    df: pd.DataFrame,
    group_column: str,
) -> pd.Series:
    """
    Missing project IDs are converted to unique per-test groups so that all
    missing projects do not accidentally become one giant group.
    """
    groups = df[group_column].astype("string").str.strip()

    missing = groups.isna() | groups.eq("")

    if "testId" in df.columns:
        fallback = "MISSING_PROJECT_TEST_" + df["testId"].astype("string")
    else:
        fallback = (
            "MISSING_PROJECT_ROW_"
            + pd.Series(df.index, index=df.index).astype("string")
        )

    return groups.mask(missing, fallback)


def create_grouped_train_test_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    group_column = resolve_group_column(df)
    groups = make_project_groups(df, group_column)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_positions, test_positions = next(
        splitter.split(df, groups=groups)
    )

    train = df.iloc[train_positions].copy()
    test = df.iloc[test_positions].copy()

    train_groups = set(
        make_project_groups(train, group_column).astype(str)
    )
    test_groups = set(
        make_project_groups(test, group_column).astype(str)
    )

    overlap = train_groups.intersection(test_groups)

    if overlap:
        raise RuntimeError(
            f"Project-group leakage detected: {len(overlap)} overlapping groups."
        )

    return train, test, group_column


# =============================================================================
# 4. 7-DAY FEATURES
# =============================================================================

def add_day7_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    result = df.copy()

    average_column = resolve_first_column(
        result,
        DAY7_AVERAGE_CANDIDATES,
    )

    count_column = resolve_first_column(
        result,
        DAY7_COUNT_CANDIDATES,
    )

    if average_column is None:
        raise KeyError(
            "No 7-day average-strength column was found. "
            f"Expected one of: {DAY7_AVERAGE_CANDIDATES}"
        )

    result["Day7AverageStrength_psi"] = numeric_series(
        result,
        average_column,
    )

    if count_column is None:
        result["Day7SpecimenCount"] = np.nan
    else:
        result["Day7SpecimenCount"] = numeric_series(
            result,
            count_column,
        )

    required = numeric_series(result, REQUIRED_STRENGTH)

    result["Day7MarginToRequired_psi"] = (
        result["Day7AverageStrength_psi"] - required
    )

    result["Day7ToRequiredRatio"] = (
        result["Day7AverageStrength_psi"]
        / required.replace(0, np.nan)
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


# =============================================================================
# 5. SUPPLIER / PLANT / MIX CONTEXT
# =============================================================================

def normalize_category(series: pd.Series) -> pd.Series:
    values = series.astype("string").fillna("__MISSING__")

    values = (
        values.str.strip()
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
        .replace("", "__MISSING__")
    )

    return values


def resolve_context_sources(df: pd.DataFrame) -> ContextSources:
    resolved: dict[str, str] = {}

    for logical_name, candidates in CONTEXT_COLUMN_CANDIDATES.items():
        column = resolve_first_column(df, candidates)

        if column is None:
            raise KeyError(
                f"Context field '{logical_name}' was not found. "
                f"Expected one of: {candidates}"
            )

        resolved[logical_name] = column

    return ContextSources(
        supplier=resolved["Supplier"],
        plant=resolved["Plant"],
        mix=resolved["Mix"],
    )


def build_context_categories(
    df: pd.DataFrame,
    sources: ContextSources,
) -> pd.DataFrame:
    context = pd.DataFrame(index=df.index)

    context["SupplierCategory"] = normalize_category(
        df[sources.supplier]
    )
    context["PlantCategory"] = normalize_category(
        df[sources.plant]
    )
    context["MixCategory"] = normalize_category(
        df[sources.mix]
    )

    context["SupplierPlantCategory"] = (
        context["SupplierCategory"]
        + "|"
        + context["PlantCategory"]
    )

    context["SupplierPlantMixCategory"] = (
        context["SupplierCategory"]
        + "|"
        + context["PlantCategory"]
        + "|"
        + context["MixCategory"]
    )

    return context


def smoothed_target_mapping(
    category: pd.Series,
    target: pd.Series,
    global_mean: float,
    smoothing: float,
) -> pd.Series:
    stats = pd.DataFrame(
        {
            "category": category,
            "target": target,
        }
    ).groupby(
        "category",
        dropna=False,
    )["target"].agg(["sum", "count"])

    return (
        stats["sum"] + smoothing * global_mean
    ) / (
        stats["count"] + smoothing
    )


def cross_fitted_target_encode(
    train_categories: pd.DataFrame,
    test_categories: pd.DataFrame,
    y_train: pd.Series,
    train_groups: pd.Series,
    *,
    smoothing: float = TARGET_ENCODING_SMOOTHING,
    max_splits: int = TARGET_ENCODING_FOLDS,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
]:
    """
    Leakage-controlled target encoding.

    Training rows:
        target means are generated out-of-fold by project group.

    Test rows:
        mappings are learned from the complete training data only.

    Unknown test categories:
        fall back to the training global mean.
    """
    y_train = pd.to_numeric(y_train, errors="coerce")

    if y_train.isna().any():
        raise ValueError(
            "Target encoding received missing training targets."
        )

    groups = train_groups.astype("string").fillna(
        "__MISSING_GROUP__"
    )

    unique_groups = int(groups.nunique())
    n_splits = min(max_splits, unique_groups)

    if n_splits < 2:
        raise ValueError(
            "At least two project groups are required "
            "for cross-fitted target encoding."
        )

    splitter = GroupKFold(n_splits=n_splits)
    global_mean = float(y_train.mean())

    encoded_train = pd.DataFrame(index=train_categories.index)
    encoded_test = pd.DataFrame(index=test_categories.index)

    metadata_rows: list[dict[str, object]] = []
    encoder_state: dict[str, object] = {
        "global_mean": global_mean,
        "smoothing": smoothing,
        "group_folds": n_splits,
        "columns": {},
    }

    for column in train_categories.columns:
        train_values = normalize_category(
            train_categories[column]
        )
        test_values = normalize_category(
            test_categories[column]
        )

        oof = pd.Series(
            np.nan,
            index=train_categories.index,
            dtype=float,
        )

        for fit_positions, validation_positions in splitter.split(
            train_categories,
            y_train,
            groups,
        ):
            fit_index = train_categories.index[fit_positions]
            validation_index = train_categories.index[
                validation_positions
            ]

            mapping = smoothed_target_mapping(
                train_values.loc[fit_index],
                y_train.loc[fit_index],
                global_mean,
                smoothing,
            )

            oof.loc[validation_index] = (
                train_values.loc[validation_index]
                .map(mapping)
                .fillna(global_mean)
            )

        full_mapping = smoothed_target_mapping(
            train_values,
            y_train,
            global_mean,
            smoothing,
        )

        train_counts = train_values.value_counts(
            dropna=False
        )

        encoded_train[
            f"{column}_TargetMean"
        ] = oof.fillna(global_mean)

        encoded_test[
            f"{column}_TargetMean"
        ] = (
            test_values
            .map(full_mapping)
            .fillna(global_mean)
        )

        encoded_train[
            f"{column}_LogCount"
        ] = np.log1p(
            train_values
            .map(train_counts)
            .fillna(0)
            .astype(float)
        )

        encoded_test[
            f"{column}_LogCount"
        ] = np.log1p(
            test_values
            .map(train_counts)
            .fillna(0)
            .astype(float)
        )

        unknown_test = ~test_values.isin(
            full_mapping.index
        )

        encoded_train[
            f"{column}_Unknown"
        ] = 0

        encoded_test[
            f"{column}_Unknown"
        ] = unknown_test.astype(int)

        metadata_rows.append(
            {
                "ContextColumn": column,
                "TrainUniqueCategories": int(
                    train_values.nunique()
                ),
                "TestUniqueCategories": int(
                    test_values.nunique()
                ),
                "UnknownTestRows": int(
                    unknown_test.sum()
                ),
                "UnknownTestPercent": float(
                    unknown_test.mean() * 100.0
                ),
                "Smoothing": smoothing,
                "GroupFolds": n_splits,
            }
        )

        encoder_state["columns"][column] = {
            "target_mean_mapping": {
                str(key): float(value)
                for key, value in full_mapping.items()
            },
            "count_mapping": {
                str(key): int(value)
                for key, value in train_counts.items()
            },
        }

    return (
        encoded_train,
        encoded_test,
        pd.DataFrame(metadata_rows),
        encoder_state,
    )


# =============================================================================
# 6. MODELS AND METRICS
# =============================================================================

def build_regression_models() -> dict[str, object]:
    models: dict[str, object] = {
        "DummyMean": DummyRegressor(
            strategy="mean"
        ),

        "Ridge": Pipeline(
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
                    Ridge(alpha=10.0),
                ),
            ]
        ),

        "RandomForest": Pipeline(
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

    if XGBRegressor is not None:
        models["XGBoost"] = Pipeline(
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
                        n_estimators=400,
                        learning_rate=0.05,
                        max_depth=6,
                        min_child_weight=5,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        tree_method="hist",
                    ),
                ),
            ]
        )

    return models


def regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    residual = predicted - actual
    absolute_error = np.abs(residual)

    return {
        "MAE": float(
            mean_absolute_error(actual, predicted)
        ),
        "MedianAE": float(
            median_absolute_error(actual, predicted)
        ),
        "RMSE": float(
            math.sqrt(
                mean_squared_error(actual, predicted)
            )
        ),
        "R2": float(
            r2_score(actual, predicted)
        ),
        "MeanBias": float(
            np.mean(residual)
        ),
        "Within300PsiPercent": float(
            np.mean(absolute_error <= 300.0) * 100.0
        ),
        "Within500PsiPercent": float(
            np.mean(absolute_error <= 500.0) * 100.0
        ),
    }


# =============================================================================
# 7. MAIN EXPERIMENT
# =============================================================================

def main() -> int:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "IMTS Field Core consolidated experiment started"
    )

    # -------------------------------------------------------------------------
    # Read the cleaned dataset.
    # -------------------------------------------------------------------------
    input_file = first_existing_path(
        INPUT_CANDIDATES
    )

    print(f"Input: {input_file}")

    df = read_csv(input_file)

    require_columns(
        df,
        [
            TARGET,
            REQUIRED_STRENGTH,
            *FIELD_FEATURES,
        ],
    )

    # Target and required strength must be valid.
    target = numeric_series(df, TARGET)
    required = numeric_series(
        df,
        REQUIRED_STRENGTH,
    )

    eligible_mask = (
        target.gt(0)
        & required.gt(0)
    )

    df = df.loc[
        eligible_mask
    ].copy()

    print(
        f"Eligible clean rows: {len(df):,}"
    )

    # -------------------------------------------------------------------------
    # Create ONE reusable project-grouped holdout split.
    # -------------------------------------------------------------------------
    train_raw, test_raw, group_column = (
        create_grouped_train_test_split(df)
    )

    print(
        f"Grouping column: {group_column}"
    )
    print(
        f"Initial train rows: {len(train_raw):,}"
    )
    print(
        f"Initial test rows: {len(test_raw):,}"
    )

    # -------------------------------------------------------------------------
    # Add 7-day derived features.
    # -------------------------------------------------------------------------
    train_with_day7, day7_metadata = (
        add_day7_features(train_raw)
    )
    test_with_day7, _ = (
        add_day7_features(test_raw)
    )

    # -------------------------------------------------------------------------
    # FAIR COMPARISON:
    # All three feature sets use the same rows with a valid 7-day result.
    # -------------------------------------------------------------------------
    train_day7_mask = numeric_series(
        train_with_day7,
        "Day7AverageStrength_psi",
    ).gt(0)

    test_day7_mask = numeric_series(
        test_with_day7,
        "Day7AverageStrength_psi",
    ).gt(0)

    train = train_with_day7.loc[
        train_day7_mask
    ].copy()

    test = test_with_day7.loc[
        test_day7_mask
    ].copy()

    print(
        "Common comparison rows with valid 7-day strength:"
    )
    print(
        f"  Train: {len(train):,}"
    )
    print(
        f"  Test : {len(test):,}"
    )

    y_train = numeric_series(
        train,
        TARGET,
    )
    y_test = numeric_series(
        test,
        TARGET,
    )

    # -------------------------------------------------------------------------
    # Feature Set 1:
    # Day-0 = field measurements + required 28-day strength
    # -------------------------------------------------------------------------
    day0_train = numeric_frame(
        train,
        DAY0_FEATURES,
    )
    day0_test = numeric_frame(
        test,
        DAY0_FEATURES,
    )

    # -------------------------------------------------------------------------
    # Feature Set 2:
    # Day-7 = Day-0 + 7-day strength features
    # -------------------------------------------------------------------------
    day7_train = numeric_frame(
        train,
        DAY7_FEATURES,
    )
    day7_test = numeric_frame(
        test,
        DAY7_FEATURES,
    )

    day7_full_train = pd.concat(
        [
            day0_train,
            day7_train,
        ],
        axis=1,
    )

    day7_full_test = pd.concat(
        [
            day0_test,
            day7_test,
        ],
        axis=1,
    )

    # -------------------------------------------------------------------------
    # Feature Set 3:
    # Full = Day-0 + Supplier/Plant/Mix context + Day-7
    # -------------------------------------------------------------------------
    context_sources = resolve_context_sources(
        train
    )

    # Ensure the same raw context columns exist in test data.
    for source_column in (
        context_sources.supplier,
        context_sources.plant,
        context_sources.mix,
    ):
        if source_column not in test.columns:
            raise KeyError(
                f"Test data is missing context column: {source_column}"
            )

    train_context = build_context_categories(
        train,
        context_sources,
    )
    test_context = build_context_categories(
        test,
        context_sources,
    )

    train_groups = make_project_groups(
        train,
        group_column,
    )

    (
        encoded_context_train,
        encoded_context_test,
        context_metadata,
        context_encoder_state,
    ) = cross_fitted_target_encode(
        train_context,
        test_context,
        y_train,
        train_groups,
    )

    full_train = pd.concat(
        [
            day0_train,
            encoded_context_train,
            day7_train,
        ],
        axis=1,
    )

    full_test = pd.concat(
        [
            day0_test,
            encoded_context_test,
            day7_test,
        ],
        axis=1,
    )

    # -------------------------------------------------------------------------
    # Exactly THREE feature sets.
    # No Field-Only experiment.
    # -------------------------------------------------------------------------
    feature_sets = {
        "Day0_FieldPlusRequired": (
            day0_train,
            day0_test,
        ),

        "Day7_FieldPlusRequired": (
            day7_full_train,
            day7_full_test,
        ),

        "Full_ContextPlusDay7": (
            full_train,
            full_test,
        ),
    }

    # -------------------------------------------------------------------------
    # Train all regression models on each feature set.
    # -------------------------------------------------------------------------
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for feature_set_name, (
        x_train,
        x_test,
    ) in feature_sets.items():

        for model_name, model in (
            build_regression_models().items()
        ):
            print(
                f"Training "
                f"{feature_set_name} / {model_name} ..."
            )

            start = time.perf_counter()

            model.fit(
                x_train,
                y_train,
            )

            predicted = model.predict(
                x_test
            )

            elapsed = (
                time.perf_counter() - start
            )

            metrics = regression_metrics(
                y_test.to_numpy(),
                predicted,
            )

            metric_rows.append(
                {
                    "FeatureSet": feature_set_name,
                    "Model": model_name,
                    "TrainRows": len(train),
                    "TestRows": len(test),
                    "FeatureCount": x_train.shape[1],
                    "TrainingSeconds": elapsed,
                    **metrics,
                }
            )

            prediction_frame = pd.DataFrame(
                {
                    "FeatureSet": feature_set_name,
                    "Model": model_name,
                    "ActualStrength28_psi": (
                        y_test.to_numpy()
                    ),
                    "PredictedStrength28_psi": predicted,
                    "ResidualPsi": (
                        predicted
                        - y_test.to_numpy()
                    ),
                    "AbsoluteErrorPsi": np.abs(
                        predicted
                        - y_test.to_numpy()
                    ),
                }
            )

            for identifier in [
                "testId",
                "projectId",
                "projectNo",
                "OfficeName",
                "officeId",
            ]:
                if identifier in test.columns:
                    prediction_frame[
                        identifier
                    ] = test[
                        identifier
                    ].to_numpy()

            prediction_frames.append(
                prediction_frame
            )

            model_bundle = {
                "model": model,
                "feature_set": feature_set_name,
                "model_name": model_name,
                "model_input_columns": list(
                    x_train.columns
                ),
                "target": TARGET,
                "required_strength_column": (
                    REQUIRED_STRENGTH
                ),
                "field_features": FIELD_FEATURES,
                "day7_features": DAY7_FEATURES,
                "day7_source_columns": (
                    day7_metadata
                ),
                "context_sources": (
                    context_sources.__dict__
                    if feature_set_name
                    == "Full_ContextPlusDay7"
                    else None
                ),
                "context_encoder_state": (
                    context_encoder_state
                    if feature_set_name
                    == "Full_ContextPlusDay7"
                    else None
                ),
                "group_column": group_column,
                "random_state": RANDOM_STATE,
            }

            joblib.dump(
                model_bundle,
                MODEL_DIR
                / (
                    f"{feature_set_name}"
                    f"__{model_name}.joblib"
                ),
            )

    # -------------------------------------------------------------------------
    # Save outputs.
    # -------------------------------------------------------------------------
    metrics_df = (
        pd.DataFrame(metric_rows)
        .sort_values(
            ["MAE", "RMSE"],
            ascending=True,
        )
        .reset_index(drop=True)
    )

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    best_by_feature_set = (
        metrics_df
        .sort_values("MAE")
        .groupby(
            "FeatureSet",
            as_index=False,
        )
        .first()
        .sort_values("MAE")
        .reset_index(drop=True)
    )

    metrics_df.to_csv(
        OUTPUT_DIR / "model_metrics.csv",
        index=False,
    )

    best_by_feature_set.to_csv(
        OUTPUT_DIR
        / "best_model_by_feature_set.csv",
        index=False,
    )

    predictions_df.to_csv(
        OUTPUT_DIR / "predictions.csv",
        index=False,
    )

    context_metadata.to_csv(
        OUTPUT_DIR
        / "context_encoding_metadata.csv",
        index=False,
    )

    # Save the exact split membership for reproducibility.
    split_frames: list[pd.DataFrame] = []

    for split_name, split_df in [
        ("Train", train),
        ("Test", test),
    ]:
        split_frame = pd.DataFrame(
            {
                "Split": split_name,
                "RowIndex": split_df.index,
            }
        )

        for identifier in [
            "testId",
            group_column,
        ]:
            if identifier in split_df.columns:
                split_frame[
                    identifier
                ] = split_df[
                    identifier
                ].to_numpy()

        split_frames.append(split_frame)

    split_assignments = pd.concat(
        split_frames,
        ignore_index=True,
    )

    split_assignments.to_csv(
        OUTPUT_DIR
        / "split_assignments_day7_common_rows.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Improvement summary between the three feature sets.
    # -------------------------------------------------------------------------
    best_lookup = (
        best_by_feature_set
        .set_index("FeatureSet")
    )

    day0_mae = float(
        best_lookup.loc[
            "Day0_FieldPlusRequired",
            "MAE",
        ]
    )

    day7_mae = float(
        best_lookup.loc[
            "Day7_FieldPlusRequired",
            "MAE",
        ]
    )

    full_mae = float(
        best_lookup.loc[
            "Full_ContextPlusDay7",
            "MAE",
        ]
    )

    day7_improvement = (
        (day0_mae - day7_mae)
        / day0_mae
        * 100.0
    )

    context_improvement = (
        (day7_mae - full_mae)
        / day7_mae
        * 100.0
    )

    total_improvement = (
        (day0_mae - full_mae)
        / day0_mae
        * 100.0
    )

    summary = {
        "input_file": str(input_file),
        "eligible_clean_rows_before_split": len(df),
        "group_column": group_column,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "comparison_train_rows_with_7_day": (
            len(train)
        ),
        "comparison_test_rows_with_7_day": (
            len(test)
        ),
        "day7_source_columns": day7_metadata,
        "context_sources": context_sources.__dict__,
        "feature_sets": {
            "Day0_FieldPlusRequired": list(
                day0_train.columns
            ),
            "Day7_FieldPlusRequired": list(
                day7_full_train.columns
            ),
            "Full_ContextPlusDay7": list(
                full_train.columns
            ),
        },
        "best_models": best_by_feature_set[
            [
                "FeatureSet",
                "Model",
                "MAE",
                "RMSE",
                "R2",
            ]
        ].to_dict(orient="records"),
        "best_day0_mae": day0_mae,
        "best_day7_mae": day7_mae,
        "best_full_mae": full_mae,
        "day7_improvement_vs_day0_percent": (
            day7_improvement
        ),
        "context_improvement_over_day7_percent": (
            context_improvement
        ),
        "full_improvement_vs_day0_percent": (
            total_improvement
        ),
        "important_note": (
            "All three feature sets were evaluated on the "
            "same rows with a valid 7-day strength result "
            "so that the comparison is fair."
        ),
    }

    save_json(
        summary,
        OUTPUT_DIR / "experiment_summary.json",
    )

    # -------------------------------------------------------------------------
    # Terminal report.
    # -------------------------------------------------------------------------
    print()
    print(
        "Consolidated experiment completed."
    )
    print()

    print(
        metrics_df[
            [
                "FeatureSet",
                "Model",
                "MAE",
                "RMSE",
                "R2",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        "Best model by feature set:"
    )
    print(
        best_by_feature_set[
            [
                "FeatureSet",
                "Model",
                "MAE",
                "RMSE",
                "R2",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        f"Day-7 improvement over Day-0: "
        f"{day0_mae - day7_mae:,.1f} psi "
        f"({day7_improvement:.1f}%)"
    )

    print(
        f"Context improvement over Day-7: "
        f"{day7_mae - full_mae:,.1f} psi "
        f"({context_improvement:.1f}%)"
    )

    print(
        f"Full-model improvement over Day-0: "
        f"{day0_mae - full_mae:,.1f} psi "
        f"({total_improvement:.1f}%)"
    )

    print()
    print(
        f"Output: {OUTPUT_DIR}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
