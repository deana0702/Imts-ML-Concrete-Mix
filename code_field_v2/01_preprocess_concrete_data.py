"""Prepare the IMTS Field-Core dataset for regression and classification.

What this script does
---------------------
* Reads a CSV or Parquet export from the IMTS Field-Core SQL query.
* Validates required columns and the one-row-per-testId grain.
* Converts model columns to numeric values.
* Replaces clearly invalid numeric values with NaN and records audit counts.
* Recomputes/validates FailureFlag28 from the two strength columns.
* Creates specification-deviation and Day-7 derived features.
* Creates separate Day-0 and Day-7 eligible datasets.
* Writes data-quality, coverage, and cohort summaries.

What this script deliberately does NOT do
-----------------------------------------
* It does not impute missing values, scale features, encode categories, select
  features, resample failures, split data, or train a model. Those operations
  must occur inside each training fold to prevent data leakage.

Examples
--------
python 01_preprocess_concrete_data.py --input IMTS_concrete_field_core.csv
python 01_preprocess_concrete_data.py --input data.parquet --output-dir outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import config


COMPLIANCE_COMPONENT_FLAGS = [
    "SlumpOutOfSpecFlag",
    "SpreadOutOfSpecFlag",
    "AirOutOfSpecFlag",
    "UnitWeightOutOfSpecFlag",
    "ConcreteTempOutOfSpecFlag",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess the IMTS concrete Field-Core ML extract."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=config.DEFAULT_INPUT_PATH,
        help="Input .csv or .parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.DEFAULT_OUTPUT_DIR,
        help="Directory for cleaned datasets and quality reports.",
    )
    parser.add_argument(
        "--unit-system",
        type=int,
        default=config.US_UNIT_SYSTEM_ID,
        help="ConcreteTestUnitSystem to retain when that column exists.",
    )
    parser.add_argument(
        "--keep-invalid-dates",
        action="store_true",
        help="Keep IsValidCastDate=0 rows. Default behavior removes them.",
    )
    parser.add_argument(
        "--allow-duplicate-testid",
        action="store_true",
        help="Keep the first duplicate testId instead of stopping.",
    )
    return parser.parse_args()


def read_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input type: {suffix}. Use CSV or Parquet.")


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError("Required columns are missing: " + ", ".join(missing))


def existing(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def numeric_candidates() -> list[str]:
    columns: list[str] = []
    for values in config.FEATURE_SETS.values():
        columns.extend(values)
    columns.extend(
        [
            config.REGRESSION_TARGET,
            config.CLASSIFICATION_TARGET,
            "IsValidCastDate",
            "ConcreteTestUnitSystem",
            "AverageActualStrength7_psi",
            "MinimumActualStrength7_psi",
            "MaximumActualStrength7_psi",
            "ActualStrengthRange7_psi",
            "ActualStrength7SpecimenCount",
        ]
    )
    return list(dict.fromkeys(columns))


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for column in existing(df, numeric_candidates()):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def replace_invalid_ranges(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_rows: list[dict[str, object]] = []
    for column, (minimum, maximum) in config.VALID_RANGES.items():
        if column not in df.columns:
            continue
        invalid = df[column].notna() & ~df[column].between(minimum, maximum)
        audit_rows.append(
            {
                "column": column,
                "valid_min": minimum,
                "valid_max": maximum,
                "invalid_value_count_replaced_with_null": int(invalid.sum()),
            }
        )
        df.loc[invalid, column] = np.nan
    return df, pd.DataFrame(audit_rows)


def deviation(
    actual: pd.Series, minimum: pd.Series | None, maximum: pd.Series | None
) -> tuple[pd.Series, pd.Series]:
    below = pd.Series(np.nan, index=actual.index, dtype="float64")
    above = pd.Series(np.nan, index=actual.index, dtype="float64")
    if minimum is not None:
        known = actual.notna() & minimum.notna()
        below.loc[known] = (minimum.loc[known] - actual.loc[known]).clip(lower=0)
    if maximum is not None:
        known = actual.notna() & maximum.notna()
        above.loc[known] = (actual.loc[known] - maximum.loc[known]).clip(lower=0)
    return below, above


def add_compliance_deviations(df: pd.DataFrame) -> pd.DataFrame:
    mappings = [
        ("Slump", "EffectiveSlump_in", "uwSlump_specMin", "uwSlump_specMax"),
        ("Spread", "EffectiveSpread_in", "uwSpread_specMin", "uwSpread_specMax"),
        ("Air", "EffectiveAir_percent", "uwAir_specMin", "uwAir_specMax"),
        (
            "UnitWeight",
            "EffectiveUnitWeight_lb_ft3",
            "uwWeight_specMin",
            "uwWeight_specMax",
        ),
        (
            "ConcreteTemp",
            "EffectiveConcreteTemp_F",
            "uwConcreteTemp_specMin",
            "uwConcreteTemp_specMax",
        ),
    ]
    for prefix, actual_col, min_col, max_col in mappings:
        if actual_col not in df.columns:
            df[f"{prefix}BelowMinAmount"] = np.nan
            df[f"{prefix}AboveMaxAmount"] = np.nan
            continue
        minimum = df[min_col] if min_col in df.columns else None
        maximum = df[max_col] if max_col in df.columns else None
        below, above = deviation(df[actual_col], minimum, maximum)
        df[f"{prefix}BelowMinAmount"] = below
        df[f"{prefix}AboveMaxAmount"] = above

    flags = existing(df, COMPLIANCE_COMPONENT_FLAGS)
    if flags:
        known_count = df[flags].notna().sum(axis=1)
        df["FieldOutOfSpecCount"] = df[flags].eq(1).sum(axis=1).astype(float)
        df.loc[known_count.eq(0), "FieldOutOfSpecCount"] = np.nan
    else:
        df["FieldOutOfSpecCount"] = np.nan
    return df


def add_day7_features(df: pd.DataFrame) -> pd.DataFrame:
    day7 = df.get("AverageActualStrength7_psi")
    required = df.get("ApplicableSpecifiedStrength28")
    if day7 is None or required is None:
        df["Strength7ToSpecifiedStrength28Ratio"] = np.nan
        df["Strength7Margin_psi"] = np.nan
        return df
    valid = day7.notna() & required.notna() & required.gt(0)
    df["Strength7ToSpecifiedStrength28Ratio"] = np.nan
    df.loc[valid, "Strength7ToSpecifiedStrength28Ratio"] = (
        day7.loc[valid] / required.loc[valid]
    )
    df["Strength7Margin_psi"] = day7 - required
    return df


def validate_target(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    actual = df[config.REGRESSION_TARGET]
    required = df["ApplicableSpecifiedStrength28"]
    calculated = pd.Series(np.nan, index=df.index, dtype="float64")
    known = actual.notna() & required.notna()
    calculated.loc[known] = (actual.loc[known] < required.loc[known]).astype(int)

    mismatch_count = 0
    if config.CLASSIFICATION_TARGET in df.columns:
        stored = pd.to_numeric(df[config.CLASSIFICATION_TARGET], errors="coerce")
        mismatch = known & stored.notna() & stored.ne(calculated)
        mismatch_count = int(mismatch.sum())

    # The deterministic definition is authoritative for this ML extract.
    df[config.CLASSIFICATION_TARGET] = calculated.astype("Int64")
    return df, mismatch_count


def apply_filters(
    df: pd.DataFrame, unit_system: int, keep_invalid_dates: bool
) -> tuple[pd.DataFrame, dict[str, int]]:
    counts = {"input_rows": len(df)}
    if config.UNIT_SYSTEM_COLUMN in df.columns:
        df = df[df[config.UNIT_SYSTEM_COLUMN].eq(unit_system)].copy()
    counts["after_unit_system_filter"] = len(df)

    if not keep_invalid_dates and "IsValidCastDate" in df.columns:
        df = df[df["IsValidCastDate"].eq(1)].copy()
    counts["after_valid_date_filter"] = len(df)
    return df, counts


def resolve_duplicates(
    df: pd.DataFrame, allow_duplicate_testid: bool
) -> tuple[pd.DataFrame, int]:
    duplicate_mask = df["testId"].duplicated(keep=False)
    duplicate_row_count = int(duplicate_mask.sum())
    if duplicate_row_count and not allow_duplicate_testid:
        examples = df.loc[duplicate_mask, "testId"].head(10).tolist()
        raise ValueError(
            "The extract is not one row per testId. "
            f"Duplicate rows: {duplicate_row_count}; example testIds: {examples}. "
            "Fix the SQL join or rerun with --allow-duplicate-testid only for audit."
        )
    if duplicate_row_count:
        df = df.drop_duplicates("testId", keep="first").copy()
    return df, duplicate_row_count


def feature_coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature_set, columns in config.FEATURE_SETS.items():
        for column in columns:
            present = column in df.columns
            non_null = int(df[column].notna().sum()) if present else 0
            rows.append(
                {
                    "feature_set": feature_set,
                    "column": column,
                    "column_present": int(present),
                    "non_null_count": non_null,
                    "coverage_percent": round(100.0 * non_null / len(df), 2)
                    if len(df)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows).drop_duplicates(["feature_set", "column"])


def cohort_summary(df: pd.DataFrame) -> pd.DataFrame:
    regression_eligible = (
        df[config.REGRESSION_TARGET].notna()
        & df["ApplicableSpecifiedStrength28"].notna()
    )
    classification_eligible = df[config.CLASSIFICATION_TARGET].notna()
    day7_eligible = classification_eligible & df.get(
        "AverageActualStrength7_psi", pd.Series(np.nan, index=df.index)
    ).notna()
    failure_count = int(df.loc[classification_eligible, config.CLASSIFICATION_TARGET].sum())
    classification_count = int(classification_eligible.sum())
    return pd.DataFrame(
        [
            {"metric": "all_cleaned_rows", "value": len(df)},
            {"metric": "regression_eligible_rows", "value": int(regression_eligible.sum())},
            {"metric": "classification_eligible_rows", "value": classification_count},
            {"metric": "day7_eligible_rows", "value": int(day7_eligible.sum())},
            {"metric": "failure_rows", "value": failure_count},
            {
                "metric": "failure_rate_percent",
                "value": round(100.0 * failure_count / classification_count, 3)
                if classification_count
                else np.nan,
            },
        ]
    )


def write_dataset(df: pd.DataFrame, path_without_suffix: Path) -> Path:
    """Prefer Parquet; fall back to CSV when a Parquet engine is unavailable."""
    parquet_path = path_without_suffix.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except (ImportError, ModuleNotFoundError):
        csv_path = path_without_suffix.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = read_input(args.input)
    require_columns(df, config.BASE_REQUIRED_COLUMNS)
    df, duplicate_row_count = resolve_duplicates(df, args.allow_duplicate_testid)
    # SQL/CSV exports sometimes store filter flags as text (for example "0"
    # and "1"). Convert first so those valid rows are not filtered out.
    df = coerce_numeric(df)
    df, filter_counts = apply_filters(df, args.unit_system, args.keep_invalid_dates)
    df, range_audit = replace_invalid_ranges(df)
    df, target_mismatch_count = validate_target(df)
    df = add_compliance_deviations(df)
    df = add_day7_features(df)

    regression_mask = (
        df[config.REGRESSION_TARGET].notna()
        & df["ApplicableSpecifiedStrength28"].notna()
    )
    classification_mask = df[config.CLASSIFICATION_TARGET].notna()
    day7_mask = classification_mask & df.get(
        "AverageActualStrength7_psi", pd.Series(np.nan, index=df.index)
    ).notna()

    # Preserve all audit and context columns. Model scripts will select only the
    # configured feature lists and fit fold-specific transformations.
    cleaned_path = write_dataset(df, args.output_dir / "concrete_field_core_cleaned")
    regression_path = write_dataset(
        df.loc[regression_mask].copy(),
        args.output_dir / "concrete_regression_eligible",
    )
    classification_path = write_dataset(
        df.loc[classification_mask].copy(),
        args.output_dir / "concrete_classification_eligible",
    )
    day7_path = write_dataset(
        df.loc[day7_mask].copy(), args.output_dir / "concrete_day7_eligible"
    )

    coverage = feature_coverage(df)
    coverage.to_csv(args.output_dir / "feature_coverage.csv", index=False)
    range_audit.to_csv(args.output_dir / "invalid_range_audit.csv", index=False)
    cohorts = cohort_summary(df)
    cohorts.to_csv(args.output_dir / "cohort_summary.csv", index=False)

    run_summary = {
        **filter_counts,
        "duplicate_testid_rows_found": duplicate_row_count,
        "failure_flag_mismatches_recomputed": target_mismatch_count,
        "cleaned_rows": len(df),
        "cleaned_dataset": str(cleaned_path),
        "regression_dataset": str(regression_path),
        "classification_dataset": str(classification_path),
        "day7_dataset": str(day7_path),
        "global_imputation_applied": False,
        "global_scaling_applied": False,
        "global_resampling_applied": False,
    }
    with (args.output_dir / "preprocessing_run_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(run_summary, handle, indent=2)

    print("Preprocessing completed.")
    print(cohorts.to_string(index=False))
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
