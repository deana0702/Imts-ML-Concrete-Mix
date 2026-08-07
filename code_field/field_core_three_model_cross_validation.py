from __future__ import annotations

"""
IMTS Field Core - Consolidated 5-Fold Project-Grouped Cross-Validation

This file cross-validates the SAME three feature sets used in
field_core_three_model_experiment.py:

1) Day0_FieldPlusRequired
   - Day-0 field measurements
   - Applicable 28-day required/design strength

2) Day7_FieldPlusRequired
   - Day-0 features
   - 7-day strength features

3) Full_ContextPlusDay7
   - Day-0 features
   - Supplier / Plant / Mix context
   - 7-day strength features

Important:
- Field-only is intentionally NOT included.
- All three feature sets are evaluated on the SAME rows with a valid 7-day
  strength result so their CV metrics are directly comparable.
- Outer CV is grouped by project.
- Supplier/Plant/Mix target encoding is rebuilt INSIDE EACH OUTER FOLD.
  Therefore the validation fold is never used to build its own context encoding.
- XGBoost is included in build_regression_models().

Run:
    python code_field/field_core_three_model_cross_validation.py

Dependency:
    pip install xgboost
"""

from dataclasses import dataclass
from pathlib import Path
import json
import math
import time
from typing import Iterable

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
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError(
        "xgboost is required for this script.\n"
        "Install it with:\n"
        "    pip install xgboost"
    ) from exc


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent


def find_repo_root() -> Path:
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

OUTPUT_DIR = FIELD_CORE_OUTPUT_ROOT / "consolidated_three_model_cross_validation"

TARGET = "AverageActualStrength28_psi"
REQUIRED_STRENGTH = "ApplicableSpecifiedStrength28"

RANDOM_STATE = 42
OUTER_CV_FOLDS = 5
TARGET_ENCODING_FOLDS = 5
TARGET_ENCODING_SMOOTHING = 20.0


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
# Day-7 source columns
# -----------------------------------------------------------------------------

DAY7_AVERAGE_CANDIDATES = [
    "AverageActualStrength7_psi",
    "AverageActualStrength7",
]

DAY7_COUNT_CANDIDATES = [
    "ActualStrength7SpecimenCount",
    "StandardCuredStrength7SpecimenCount",
]

DAY7_FEATURES = [
    "Day7AverageStrength_psi",
    "Day7SpecimenCount",
    "Day7MarginToRequired_psi",
    "Day7ToRequiredRatio",
]


# -----------------------------------------------------------------------------
# Supplier / Plant / Mix context
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
        f"{expected}"
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
# 3. PROJECT GROUPING
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
    Missing project IDs are converted to unique per-test groups so that
    unrelated missing-project records are not forced into one group.
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


# =============================================================================
# 4. DAY-7 FEATURES
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


# =============================================================================
# 5. SUPPLIER / PLANT / MIX CONTEXT
# =============================================================================

def normalize_category(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("__MISSING__")
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
        .replace("", "__MISSING__")
    )


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

    context["SupplierCategory"] = normalize_category(df[sources.supplier])
    context["PlantCategory"] = normalize_category(df[sources.plant])
    context["MixCategory"] = normalize_category(df[sources.mix])

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
    validation_categories: pd.DataFrame,
    y_train: pd.Series,
    train_groups: pd.Series,
    *,
    smoothing: float = TARGET_ENCODING_SMOOTHING,
    max_splits: int = TARGET_ENCODING_FOLDS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Target encoding for ONE OUTER CV FOLD.

    Critical leakage rule:
    - The outer validation fold is never used to build mappings.
    - Training rows get out-of-fold target encodings using only the
      outer-training data.
    - Validation rows are mapped from all outer-training rows only.
    """
    y_train = pd.to_numeric(y_train, errors="coerce")

    if y_train.isna().any():
        raise ValueError("Target encoding received missing training targets.")

    groups = train_groups.astype("string").fillna("__MISSING_GROUP__")

    unique_groups = int(groups.nunique())
    n_splits = min(max_splits, unique_groups)

    if n_splits < 2:
        raise ValueError(
            "At least two project groups are required "
            "for cross-fitted target encoding."
        )

    inner_splitter = GroupKFold(n_splits=n_splits)
    global_mean = float(y_train.mean())

    encoded_train = pd.DataFrame(index=train_categories.index)
    encoded_validation = pd.DataFrame(index=validation_categories.index)

    metadata_rows: list[dict[str, object]] = []

    for column in train_categories.columns:
        train_values = normalize_category(train_categories[column])
        validation_values = normalize_category(validation_categories[column])

        oof = pd.Series(
            np.nan,
            index=train_categories.index,
            dtype=float,
        )

        # Inner grouped CV creates leakage-controlled encodings
        # for the outer-training rows.
        for fit_positions, encoding_positions in inner_splitter.split(
            train_categories,
            y_train,
            groups,
        ):
            fit_index = train_categories.index[fit_positions]
            encoding_index = train_categories.index[encoding_positions]

            mapping = smoothed_target_mapping(
                train_values.loc[fit_index],
                y_train.loc[fit_index],
                global_mean,
                smoothing,
            )

            oof.loc[encoding_index] = (
                train_values.loc[encoding_index]
                .map(mapping)
                .fillna(global_mean)
            )

        # Mapping for the OUTER validation fold:
        # learned from outer-training rows only.
        full_train_mapping = smoothed_target_mapping(
            train_values,
            y_train,
            global_mean,
            smoothing,
        )

        train_counts = train_values.value_counts(dropna=False)

        encoded_train[f"{column}_TargetMean"] = oof.fillna(global_mean)

        encoded_validation[f"{column}_TargetMean"] = (
            validation_values
            .map(full_train_mapping)
            .fillna(global_mean)
        )

        encoded_train[f"{column}_LogCount"] = np.log1p(
            train_values
            .map(train_counts)
            .fillna(0)
            .astype(float)
        )

        encoded_validation[f"{column}_LogCount"] = np.log1p(
            validation_values
            .map(train_counts)
            .fillna(0)
            .astype(float)
        )

        unknown_validation = ~validation_values.isin(
            full_train_mapping.index
        )

        encoded_train[f"{column}_Unknown"] = 0
        encoded_validation[f"{column}_Unknown"] = (
            unknown_validation.astype(int)
        )

        metadata_rows.append(
            {
                "ContextColumn": column,
                "TrainUniqueCategories": int(train_values.nunique()),
                "ValidationUniqueCategories": int(
                    validation_values.nunique()
                ),
                "UnknownValidationRows": int(
                    unknown_validation.sum()
                ),
                "UnknownValidationPercent": float(
                    unknown_validation.mean() * 100.0
                ),
                "InnerEncodingFolds": n_splits,
                "Smoothing": smoothing,
            }
        )

    return (
        encoded_train,
        encoded_validation,
        pd.DataFrame(metadata_rows),
    )


# =============================================================================
# 6. REGRESSION MODELS
# =============================================================================

def build_regression_models() -> dict[str, object]:
    """
    Candidate regressors.

    XGBoost is included here as requested.

    These are deliberately reasonable baseline settings rather than an
    aggressive hyperparameter search. Cross-validation should first tell us
    which model family is strongest and most stable. Hyperparameter tuning
    can be done after that.
    """
    return {
        "DummyMean": DummyRegressor(
            strategy="mean",
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

        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            n_estimators=700,
            learning_rate=0.04,
            max_depth=6,
            min_child_weight=5,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="rmse",
        ),
    }


# =============================================================================
# 7. METRICS
# =============================================================================

def regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    residual = predicted - actual
    absolute_error = np.abs(residual)

    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "MedianAE": float(median_absolute_error(actual, predicted)),
        "RMSE": float(
            math.sqrt(mean_squared_error(actual, predicted))
        ),
        "R2": float(r2_score(actual, predicted)),
        "MeanBias": float(np.mean(residual)),
        "Within300PsiPercent": float(
            np.mean(absolute_error <= 300.0) * 100.0
        ),
        "Within500PsiPercent": float(
            np.mean(absolute_error <= 500.0) * 100.0
        ),
    }


# =============================================================================
# 8. CROSS-VALIDATION
# =============================================================================

def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("IMTS Field Core 5-fold project-grouped cross-validation")
    print()

    input_file = first_existing_path(INPUT_CANDIDATES)
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

    # -------------------------------------------------------------------------
    # Basic eligibility
    # -------------------------------------------------------------------------
    target = numeric_series(df, TARGET)
    required = numeric_series(df, REQUIRED_STRENGTH)

    eligible_mask = target.gt(0) & required.gt(0)

    df = df.loc[eligible_mask].copy()

    # -------------------------------------------------------------------------
    # Add Day-7 features and use common rows for all three feature sets.
    # -------------------------------------------------------------------------
    df, day7_metadata = add_day7_features(df)

    valid_day7_mask = numeric_series(
        df,
        "Day7AverageStrength_psi",
    ).gt(0)

    df = df.loc[valid_day7_mask].copy()

    print(f"Common rows with valid Day-7 strength: {len(df):,}")

    # -------------------------------------------------------------------------
    # Grouping
    # -------------------------------------------------------------------------
    group_column = resolve_group_column(df)
    groups = make_project_groups(df, group_column)

    unique_group_count = int(groups.nunique())

    if unique_group_count < OUTER_CV_FOLDS:
        raise ValueError(
            f"Only {unique_group_count} project groups are available, "
            f"but OUTER_CV_FOLDS={OUTER_CV_FOLDS}."
        )

    print(f"Grouping column: {group_column}")
    print(f"Unique project groups: {unique_group_count:,}")
    print(f"Outer CV folds: {OUTER_CV_FOLDS}")
    print()

    # Resolve context fields once.
    context_sources = resolve_context_sources(df)

    outer_splitter = GroupKFold(n_splits=OUTER_CV_FOLDS)

    fold_metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    context_metadata_frames: list[pd.DataFrame] = []
    fold_summary_rows: list[dict[str, object]] = []

    # =========================================================================
    # OUTER GROUPED CV
    # =========================================================================
    for fold_number, (train_positions, validation_positions) in enumerate(
        outer_splitter.split(df, groups=groups),
        start=1,
    ):
        print("=" * 78)
        print(f"OUTER FOLD {fold_number}/{OUTER_CV_FOLDS}")
        print("=" * 78)

        train = df.iloc[train_positions].copy()
        validation = df.iloc[validation_positions].copy()

        train_groups = make_project_groups(train, group_column)
        validation_groups = make_project_groups(validation, group_column)

        overlap = set(train_groups.astype(str)).intersection(
            set(validation_groups.astype(str))
        )

        if overlap:
            raise RuntimeError(
                f"Fold {fold_number}: project leakage detected "
                f"({len(overlap)} overlapping groups)."
            )

        y_train = numeric_series(train, TARGET)
        y_validation = numeric_series(validation, TARGET)

        print(f"Train rows     : {len(train):,}")
        print(f"Validation rows: {len(validation):,}")
        print(f"Train projects : {train_groups.nunique():,}")
        print(f"Valid projects : {validation_groups.nunique():,}")

        fold_summary_rows.append(
            {
                "Fold": fold_number,
                "TrainRows": len(train),
                "ValidationRows": len(validation),
                "TrainProjectGroups": int(train_groups.nunique()),
                "ValidationProjectGroups": int(
                    validation_groups.nunique()
                ),
                "TrainTargetMean": float(y_train.mean()),
                "TrainTargetStd": float(y_train.std()),
                "ValidationTargetMean": float(y_validation.mean()),
                "ValidationTargetStd": float(y_validation.std()),
            }
        )

        # ---------------------------------------------------------------------
        # Feature Set 1: Day-0 Field + Required
        # ---------------------------------------------------------------------
        day0_train = numeric_frame(train, DAY0_FEATURES)
        day0_validation = numeric_frame(validation, DAY0_FEATURES)

        # ---------------------------------------------------------------------
        # Feature Set 2: Day-7 Field + Required
        # ---------------------------------------------------------------------
        day7_train = numeric_frame(train, DAY7_FEATURES)
        day7_validation = numeric_frame(validation, DAY7_FEATURES)

        day7_full_train = pd.concat(
            [day0_train, day7_train],
            axis=1,
        )

        day7_full_validation = pd.concat(
            [day0_validation, day7_validation],
            axis=1,
        )

        # ---------------------------------------------------------------------
        # Feature Set 3: Full Context + Day-7
        #
        # Context target encoding is learned INSIDE this outer fold.
        # ---------------------------------------------------------------------
        train_context = build_context_categories(
            train,
            context_sources,
        )

        validation_context = build_context_categories(
            validation,
            context_sources,
        )

        (
            encoded_context_train,
            encoded_context_validation,
            context_metadata,
        ) = cross_fitted_target_encode(
            train_context,
            validation_context,
            y_train,
            train_groups,
        )

        context_metadata.insert(0, "OuterFold", fold_number)

        context_metadata_frames.append(context_metadata)

        full_train = pd.concat(
            [
                day0_train,
                encoded_context_train,
                day7_train,
            ],
            axis=1,
        )

        full_validation = pd.concat(
            [
                day0_validation,
                encoded_context_validation,
                day7_validation,
            ],
            axis=1,
        )

        # Exactly three feature sets.
        feature_sets = {
            "Day0_FieldPlusRequired": (
                day0_train,
                day0_validation,
            ),
            "Day7_FieldPlusRequired": (
                day7_full_train,
                day7_full_validation,
            ),
            "Full_ContextPlusDay7": (
                full_train,
                full_validation,
            ),
        }

        # ---------------------------------------------------------------------
        # Train every regressor on every feature set.
        # ---------------------------------------------------------------------
        for feature_set_name, (
            x_train,
            x_validation,
        ) in feature_sets.items():

            for model_name, model in build_regression_models().items():
                print(
                    f"Fold {fold_number} | "
                    f"{feature_set_name} | {model_name}"
                )

                start = time.perf_counter()

                model.fit(
                    x_train,
                    y_train,
                )

                predicted = model.predict(
                    x_validation
                )

                elapsed = time.perf_counter() - start

                metrics = regression_metrics(
                    y_validation.to_numpy(),
                    predicted,
                )

                fold_metric_rows.append(
                    {
                        "Fold": fold_number,
                        "FeatureSet": feature_set_name,
                        "Model": model_name,
                        "TrainRows": len(train),
                        "ValidationRows": len(validation),
                        "FeatureCount": x_train.shape[1],
                        "TrainingSeconds": elapsed,
                        **metrics,
                    }
                )

                prediction_frame = pd.DataFrame(
                    {
                        "Fold": fold_number,
                        "FeatureSet": feature_set_name,
                        "Model": model_name,
                        "ActualStrength28_psi": (
                            y_validation.to_numpy()
                        ),
                        "PredictedStrength28_psi": predicted,
                        "ResidualPsi": (
                            predicted
                            - y_validation.to_numpy()
                        ),
                        "AbsoluteErrorPsi": np.abs(
                            predicted
                            - y_validation.to_numpy()
                        ),
                    },
                    index=validation.index,
                )

                for identifier in [
                    "testId",
                    "projectId",
                    "projectNo",
                    "officeId",
                    "OfficeName",
                ]:
                    if identifier in validation.columns:
                        prediction_frame[identifier] = (
                            validation[identifier]
                            .reindex(prediction_frame.index)
                            .to_numpy()
                        )

                prediction_frames.append(
                    prediction_frame.reset_index(drop=True)
                )

        print()

    # =========================================================================
    # SUMMARIZE CV RESULTS
    # =========================================================================
    fold_metrics = pd.DataFrame(fold_metric_rows)

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    context_metadata_all = pd.concat(
        context_metadata_frames,
        ignore_index=True,
    )

    fold_summary = pd.DataFrame(
        fold_summary_rows
    )

    summary = (
        fold_metrics.groupby(
            ["FeatureSet", "Model"],
            as_index=False,
        )
        .agg(
            MeanCV_MAE=("MAE", "mean"),
            StdCV_MAE=("MAE", "std"),
            MeanCV_RMSE=("RMSE", "mean"),
            StdCV_RMSE=("RMSE", "std"),
            MeanCV_R2=("R2", "mean"),
            StdCV_R2=("R2", "std"),
            MeanCV_MedianAE=("MedianAE", "mean"),
            MeanCV_Bias=("MeanBias", "mean"),
            MeanWithin300PsiPercent=(
                "Within300PsiPercent",
                "mean",
            ),
            MeanWithin500PsiPercent=(
                "Within500PsiPercent",
                "mean",
            ),
            MeanTrainingSeconds=(
                "TrainingSeconds",
                "mean",
            ),
        )
        .sort_values(
            ["MeanCV_MAE", "MeanCV_RMSE"],
            ascending=True,
        )
        .reset_index(drop=True)
    )

    best_by_feature_set = (
        summary.sort_values(
            ["MeanCV_MAE", "MeanCV_RMSE"],
            ascending=True,
        )
        .groupby(
            "FeatureSet",
            as_index=False,
        )
        .first()
        .sort_values(
            "MeanCV_MAE",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Save CSV outputs.
    # -------------------------------------------------------------------------
    fold_metrics.to_csv(
        OUTPUT_DIR / "cv_fold_metrics.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR / "cv_summary.csv",
        index=False,
    )

    best_by_feature_set.to_csv(
        OUTPUT_DIR / "best_model_by_feature_set_cv.csv",
        index=False,
    )

    predictions.to_csv(
        OUTPUT_DIR / "cv_predictions.csv",
        index=False,
    )

    context_metadata_all.to_csv(
        OUTPUT_DIR / "cv_context_encoding_metadata.csv",
        index=False,
    )

    fold_summary.to_csv(
        OUTPUT_DIR / "cv_fold_data_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # JSON run summary.
    # -------------------------------------------------------------------------
    run_summary = {
        "input_file": str(input_file),
        "rows_used": len(df),
        "group_column": group_column,
        "unique_project_groups": unique_group_count,
        "outer_cv_folds": OUTER_CV_FOLDS,
        "inner_target_encoding_folds": TARGET_ENCODING_FOLDS,
        "target_encoding_smoothing": TARGET_ENCODING_SMOOTHING,
        "day7_source_columns": day7_metadata,
        "context_sources": context_sources.__dict__,
        "feature_sets": {
            "Day0_FieldPlusRequired": DAY0_FEATURES,
            "Day7_FieldPlusRequired": (
                DAY0_FEATURES + DAY7_FEATURES
            ),
            "Full_ContextPlusDay7": (
                "Day-0 + leakage-controlled Supplier/Plant/Mix "
                "context encoding + Day-7 features"
            ),
        },
        "candidate_models": list(
            build_regression_models().keys()
        ),
        "best_models_by_feature_set": (
            best_by_feature_set[
                [
                    "FeatureSet",
                    "Model",
                    "MeanCV_MAE",
                    "StdCV_MAE",
                    "MeanCV_RMSE",
                    "MeanCV_R2",
                ]
            ].to_dict(orient="records")
        ),
        "important_methodology_note": (
            "Supplier/Plant/Mix target encoding is recreated inside each "
            "outer project-grouped CV fold. Outer validation targets are "
            "never used to construct their context encodings."
        ),
        "comparison_note": (
            "All three feature sets use the same records with valid "
            "7-day strength so that Day-0, Day-7, and Full comparisons "
            "are directly comparable."
        ),
    }

    save_json(
        run_summary,
        OUTPUT_DIR / "cv_run_summary.json",
    )

    # =========================================================================
    # TERMINAL REPORT
    # =========================================================================
    print()
    print("=" * 100)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 100)

    print(
        summary[
            [
                "FeatureSet",
                "Model",
                "MeanCV_MAE",
                "StdCV_MAE",
                "MeanCV_RMSE",
                "StdCV_RMSE",
                "MeanCV_R2",
                "StdCV_R2",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 100)
    print("BEST MODEL BY FEATURE SET")
    print("=" * 100)

    print(
        best_by_feature_set[
            [
                "FeatureSet",
                "Model",
                "MeanCV_MAE",
                "StdCV_MAE",
                "MeanCV_RMSE",
                "StdCV_RMSE",
                "MeanCV_R2",
                "StdCV_R2",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"Output directory: {OUTPUT_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
