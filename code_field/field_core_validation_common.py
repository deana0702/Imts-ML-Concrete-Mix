from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from field_core_experiment_common import (
    DAY7_FEATURES,
    FIELD_PLUS_REQUIRED_FEATURES,
    REQUIRED_STRENGTH,
    TARGET,
    add_day7_features,
    build_context_categories,
    build_regression_models,
    build_classification_models,
    classification_metrics,
    cross_fitted_target_encode,
    fit_classifier,
    numeric_frame,
    numeric_series,
    positive_probability,
    regression_metrics,
    require_columns,
    resolve_context_sources,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def find_repo_root() -> Path:
    """Find the repository root whether scripts live in root or code_field/."""
    candidates = [SCRIPT_DIR, *SCRIPT_DIR.parents]
    for candidate in candidates:
        if (candidate / "data" / "field_core_outputs").exists():
            return candidate
        if (candidate / "data" / "prepared_field_core_US").exists():
            return candidate
    # Most users keep these scripts in <repo>/code_field/.
    return SCRIPT_DIR.parent


REPO_ROOT = find_repo_root()
DATA_DIR = REPO_ROOT / "data"
FIELD_CORE_OUTPUT_ROOT = DATA_DIR / "field_core_outputs"
VALIDATION_OUTPUT_ROOT = FIELD_CORE_OUTPUT_ROOT / "validation"

CLEAN_TRAIN_CANDIDATES = [
    FIELD_CORE_OUTPUT_ROOT / "field_core_clean" / "field_core_clean_with_required.csv",
    DATA_DIR / "field_core_clean" / "field_core_clean_with_required.csv",
]

EXTERNAL_DIR = DATA_DIR / "external_2022_2026"
EXTERNAL_RAW_FILE = EXTERNAL_DIR / "field_core_US_data_current.csv"
EXTERNAL_PREPARED_FILE = (
    EXTERNAL_DIR / "prepared" / "field_core_external_2022_2026_clean.csv"
)

PROJECT_GROUP_CANDIDATES = ["projectId", "projectNo", "SplitGroup"]
CAST_DATE_CANDIDATES = ["castDate", "CastDate"]
UNIT_SYSTEM_CANDIDATES = ["ConcreteTestUnitSystem", "unitSystem"]


def first_existing_path(paths: Iterable[Path]) -> Path:
    paths = list(paths)
    for path in paths:
        if path.exists():
            return path
    expected = "\n".join(f"  - {path}" for path in paths)
    raise FileNotFoundError(f"No expected input file was found:\n{expected}")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Could not determine CSV encoding for {path}")


def load_clean_training_data() -> tuple[pd.DataFrame, Path]:
    path = first_existing_path(CLEAN_TRAIN_CANDIDATES)
    df = read_csv(path)
    require_columns(df, ["testId", TARGET, REQUIRED_STRENGTH])
    return df, path


def resolve_first_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def resolve_cast_date_column(df: pd.DataFrame) -> str:
    column = resolve_first_column(df, CAST_DATE_CANDIDATES)
    if column is None:
        raise KeyError(f"Expected one of cast-date columns: {CAST_DATE_CANDIDATES}")
    return column


def resolve_unit_system_column(df: pd.DataFrame) -> str:
    column = resolve_first_column(df, UNIT_SYSTEM_CANDIDATES)
    if column is None:
        raise KeyError(f"Expected one of unit-system columns: {UNIT_SYSTEM_CANDIDATES}")
    return column


def resolve_project_group_column(df: pd.DataFrame) -> str:
    for column in PROJECT_GROUP_CANDIDATES:
        if column in df.columns and df[column].notna().any():
            return column
    raise KeyError(f"Expected one of project grouping columns: {PROJECT_GROUP_CANDIDATES}")


def make_groups(
    df: pd.DataFrame,
    column: str,
    *,
    missing_prefix: str,
) -> pd.Series:
    values = df[column].astype("string").str.strip()
    missing = values.isna() | values.eq("")
    if "testId" in df.columns:
        fallback = missing_prefix + df["testId"].astype("string")
    else:
        fallback = missing_prefix + pd.Series(df.index, index=df.index).astype("string")
    return values.mask(missing, fallback)


def project_groups(df: pd.DataFrame) -> tuple[pd.Series, str]:
    column = resolve_project_group_column(df)
    return make_groups(df, column, missing_prefix="MISSING_PROJECT_TEST_"), column


def prepare_failure_target(df: pd.DataFrame) -> pd.Series:
    actual = numeric_series(df, TARGET)
    required = numeric_series(df, REQUIRED_STRENGTH)
    if actual.isna().any() or required.isna().any():
        raise ValueError("Failure target requires non-missing actual and required strengths.")
    return actual.lt(required).astype(int)


def prepare_context_encoded_features(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    y_train: pd.Series,
    *,
    include_day7: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build Field + Required + Context, optionally adding 7-day features.

    Context target encodings are leakage-controlled:
      - train rows receive project-group out-of-fold encodings;
      - validation rows are mapped from train only;
      - unknown validation categories fall back to the train global mean.
    """
    sources = resolve_context_sources(train)
    for source_column in sources.__dict__.values():
        if source_column not in validation.columns:
            raise KeyError(f"Validation data is missing context column: {source_column}")

    train_context = build_context_categories(train, sources)
    validation_context = build_context_categories(validation, sources)
    groups, _ = project_groups(train)

    encoded_train, encoded_validation, metadata = cross_fitted_target_encode(
        train_context,
        validation_context,
        y_train,
        groups,
    )

    x_train = numeric_frame(train, FIELD_PLUS_REQUIRED_FEATURES)
    x_validation = numeric_frame(validation, FIELD_PLUS_REQUIRED_FEATURES)

    if include_day7:
        train_with_day7, _ = add_day7_features(train)
        validation_with_day7, _ = add_day7_features(validation)
        x_train = pd.concat(
            [x_train, encoded_train, numeric_frame(train_with_day7, DAY7_FEATURES)],
            axis=1,
        )
        x_validation = pd.concat(
            [
                x_validation,
                encoded_validation,
                numeric_frame(validation_with_day7, DAY7_FEATURES),
            ],
            axis=1,
        )
    else:
        x_train = pd.concat([x_train, encoded_train], axis=1)
        x_validation = pd.concat([x_validation, encoded_validation], axis=1)

    return x_train, x_validation, metadata


def has_valid_day7(df: pd.DataFrame) -> pd.Series:
    with_day7, _ = add_day7_features(df)
    return numeric_series(with_day7, "Day7AverageStrength_psi").gt(0)


def selected_regression_model(model_name: str) -> object:
    models = build_regression_models()
    if model_name not in models:
        raise KeyError(f"Unknown regression model: {model_name}")
    return models[model_name]


def selected_classification_model(model_name: str) -> object:
    models = build_classification_models()
    if model_name not in models:
        raise KeyError(f"Unknown classification model: {model_name}")
    return models[model_name]


def summarize_regression_folds(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "MAE",
        "MedianAE",
        "RMSE",
        "R2",
        "MeanBias",
        "Within300PsiPercent",
        "Within500PsiPercent",
    ]
    rows: list[dict[str, object]] = []
    for keys, group in fold_metrics.groupby(["Stage", "Model"], dropna=False):
        stage, model = keys
        row: dict[str, object] = {
            "Stage": stage,
            "Model": model,
            "FoldCount": int(len(group)),
            "TotalValidationRows": int(group["ValidationRows"].sum()),
        }
        for metric in metrics:
            row[f"MeanCV_{metric}"] = float(group[metric].mean())
            row[f"StdCV_{metric}"] = float(group[metric].std(ddof=0))
            row[f"BestFold_{metric}"] = float(group[metric].min()) if metric != "R2" else float(group[metric].max())
            row[f"WorstFold_{metric}"] = float(group[metric].max()) if metric != "R2" else float(group[metric].min())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["Stage", "MeanCV_MAE", "MeanCV_RMSE"])


def summarize_classification_folds(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "AveragePrecision_PR_AUC",
        "ROC_AUC",
        "Recall",
        "Precision",
        "F1",
        "BrierScore",
        "FalseNegative",
        "FalsePositive",
    ]
    rows: list[dict[str, object]] = []
    for keys, group in fold_metrics.groupby(["Stage", "Model"], dropna=False):
        stage, model = keys
        row: dict[str, object] = {
            "Stage": stage,
            "Model": model,
            "FoldCount": int(len(group)),
            "TotalValidationRows": int(group["ValidationRows"].sum()),
            "TotalValidationFailures": int(group["ValidationFailures"].sum()),
        }
        for metric in metrics:
            row[f"MeanCV_{metric}"] = float(group[metric].mean())
            row[f"StdCV_{metric}"] = float(group[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["Stage", "MeanCV_AveragePrecision_PR_AUC", "MeanCV_Recall"],
        ascending=[True, False, False],
    )


def category_keys(df: pd.DataFrame) -> pd.DataFrame:
    sources = resolve_context_sources(df)
    return build_context_categories(df, sources)


def valid_context_mask(categories: pd.DataFrame, scenario: str) -> pd.Series:
    if scenario == "UnseenSupplier":
        return categories["SupplierCategory"].ne("__MISSING__")
    if scenario == "UnseenSupplierPlantMix":
        return (
            categories["SupplierCategory"].ne("__MISSING__")
            & categories["PlantCategory"].ne("__MISSING__")
            & categories["MixCategory"].ne("__MISSING__")
        )
    raise ValueError(f"Unknown validation scenario: {scenario}")


def scenario_group_values(categories: pd.DataFrame, scenario: str) -> pd.Series:
    if scenario == "UnseenSupplier":
        return categories["SupplierCategory"]
    if scenario == "UnseenSupplierPlantMix":
        return categories["SupplierPlantMixCategory"]
    raise ValueError(f"Unknown validation scenario: {scenario}")


def metric_by_year_regression(
    frame: pd.DataFrame,
    *,
    actual_column: str,
    predicted_column: str,
    year_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year, group in frame.groupby(year_column):
        if len(group) < 2:
            continue
        metrics = regression_metrics(
            group[actual_column].to_numpy(dtype=float),
            group[predicted_column].to_numpy(dtype=float),
        )
        rows.append({"Year": int(year), "RowCount": int(len(group)), **metrics})
    return pd.DataFrame(rows)


def metric_by_year_classification(
    frame: pd.DataFrame,
    *,
    actual_column: str,
    probability_column: str,
    year_column: str,
    threshold: float = 0.50,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year, group in frame.groupby(year_column):
        if len(group) < 2:
            continue
        actual = group[actual_column].to_numpy(dtype=int)
        probability = group[probability_column].to_numpy(dtype=float)
        metrics = classification_metrics(actual, probability, threshold)
        rows.append({"Year": int(year), "RowCount": int(len(group)), **metrics})
    return pd.DataFrame(rows)
