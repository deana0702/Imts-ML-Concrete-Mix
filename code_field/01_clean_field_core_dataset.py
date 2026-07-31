from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Hardcoded paths: run with
#     python 01_clean_field_core_dataset.py
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT_DIR / "data/prepared_field_core_US/field_core_model_candidates.csv"
OUTPUT_DIR = ROOT_DIR / "data/field_core_outputs/field_core_clean"

EXPECTED_UNIT_SYSTEM_ID = 0
MIN_VALID_DATE = pd.Timestamp("2000-01-01")
MAX_VALID_DATE = pd.Timestamp.now().normalize()

TARGET = "AverageActualStrength28_psi"
REQUIRED_STRENGTH = "ApplicableSpecifiedStrength28"

CORE_FRESH_MEASUREMENTS = [
    "EffectiveSlump_in",
    "EffectiveAir_percent",
    "EffectiveUnitWeight_lb_ft3",
    "EffectiveConcreteTemp_F",
]

# These are deliberately broad sanity ranges. Values outside these limits are
# changed to missing, not used to delete the entire concrete test.
FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "EffectiveSlump_in": (0.0, 30.0),
    "EffectiveAir_percent": (0.0, 20.0),
    "EffectiveUnitWeight_lb_ft3": (40.0, 220.0),
    "EffectiveConcreteTemp_F": (-20.0, 160.0),
    "AmbientTemp_F": (-80.0, 160.0),
    "WaterAdded_gal_per_yd3": (0.0, 20.0),
    "BatchToSampleMinutes": (0.0, 720.0),
    "BatchToCastMinutes": (0.0, 720.0),
    "EffectiveSpread_in": (0.0, 40.0),
}

INDICATOR_COLUMNS = [
    "HasWaterAdded",
    "HasAnyAfterSPMeasurement",
]

MAX_TARGET_PSI = 20_000.0
MAX_REQUIRED_STRENGTH_PSI = 20_000.0


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            "Run prepare_field_core_US.py first and confirm the output path."
        )

    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Could not determine CSV encoding for {path}")


def normalize_blank_strings(df: pd.DataFrame) -> None:
    object_columns = df.select_dtypes(include=["object", "string"]).columns
    for column in object_columns:
        df[column] = df[column].replace(r"^\s*$", pd.NA, regex=True)


def numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(
        series.astype("string").str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Required columns are missing: {missing}")


def append_reason(reason_series: pd.Series, mask: pd.Series, reason: str) -> pd.Series:
    mask = mask.fillna(False)
    current = reason_series.fillna("")
    updated = np.where(
        mask,
        np.where(current == "", reason, current + ";" + reason),
        current,
    )
    return pd.Series(updated, index=reason_series.index, dtype="string").replace("", pd.NA)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 1 started: clean Field Core modeling dataset")
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_DIR}")

    df = read_csv(INPUT_FILE)
    normalize_blank_strings(df)

    require_columns(df, ["testId", TARGET])

    # Parse core audit columns.
    df[TARGET] = numeric(df[TARGET])
    if REQUIRED_STRENGTH in df.columns:
        df[REQUIRED_STRENGTH] = numeric(df[REQUIRED_STRENGTH])

    if "castDate" in df.columns:
        df["castDate"] = pd.to_datetime(df["castDate"], errors="coerce")
    else:
        df["castDate"] = pd.NaT

    if "ConcreteTestUnitSystem" in df.columns:
        df["ConcreteTestUnitSystem"] = numeric(df["ConcreteTestUnitSystem"])
    else:
        df["ConcreteTestUnitSystem"] = EXPECTED_UNIT_SYSTEM_ID

    # Convert model features to numeric where present.
    for column in list(FEATURE_RANGES) + INDICATOR_COLUMNS:
        if column in df.columns:
            df[column] = numeric(df[column])

    # ------------------------------------------------------------------
    # Row-level exclusions: these remove a test from the clean base data.
    # ------------------------------------------------------------------
    exclusion_reasons = pd.Series(pd.NA, index=df.index, dtype="string")

    duplicate_test = df["testId"].duplicated(keep=False)
    invalid_unit = df["ConcreteTestUnitSystem"].ne(EXPECTED_UNIT_SYSTEM_ID)
    invalid_date = ~df["castDate"].between(
        MIN_VALID_DATE,
        MAX_VALID_DATE,
        inclusive="both",
    )
    invalid_target = (
        df[TARGET].isna()
        | df[TARGET].le(0)
        | df[TARGET].gt(MAX_TARGET_PSI)
    )

    exclusion_reasons = append_reason(exclusion_reasons, duplicate_test, "DuplicateTestId")
    exclusion_reasons = append_reason(exclusion_reasons, invalid_unit, "NonUSUnitSystem")
    exclusion_reasons = append_reason(exclusion_reasons, invalid_date, "InvalidCastDate")
    exclusion_reasons = append_reason(
        exclusion_reasons,
        invalid_target,
        "Invalid28DayTarget",
    )

    # ------------------------------------------------------------------
    # Feature-level cleaning: impossible values become missing, but the
    # concrete test remains available for imputation-based models.
    # ------------------------------------------------------------------
    adjustment_rows: list[dict[str, object]] = []
    review_flags = pd.DataFrame(index=df.index)

    for column, (minimum, maximum) in FEATURE_RANGES.items():
        if column not in df.columns:
            adjustment_rows.append(
                {
                    "Column": column,
                    "PresentInInput": 0,
                    "MinimumAllowed": minimum,
                    "MaximumAllowed": maximum,
                    "ValuesSetToMissing": 0,
                }
            )
            continue

        out_of_range = df[column].notna() & (
            df[column].lt(minimum) | df[column].gt(maximum)
        )
        review_flags[f"{column}_OutOfRange"] = out_of_range.astype(int)
        count = int(out_of_range.sum())
        df.loc[out_of_range, column] = np.nan

        adjustment_rows.append(
            {
                "Column": column,
                "PresentInInput": 1,
                "MinimumAllowed": minimum,
                "MaximumAllowed": maximum,
                "ValuesSetToMissing": count,
            }
        )

    for column in INDICATOR_COLUMNS:
        if column not in df.columns:
            continue
        invalid_indicator = df[column].notna() & ~df[column].isin([0, 1])
        review_flags[f"{column}_InvalidIndicator"] = invalid_indicator.astype(int)
        df.loc[invalid_indicator, column] = np.nan

    # Retain unusual low strengths for review instead of deleting them.
    review_flags["TargetBelow1000Psi_Review"] = df[TARGET].between(
        0,
        1_000,
        inclusive="neither",
    ).astype(int)

    if REQUIRED_STRENGTH in df.columns:
        required_invalid = df[REQUIRED_STRENGTH].notna() & (
            df[REQUIRED_STRENGTH].le(0)
            | df[REQUIRED_STRENGTH].gt(MAX_REQUIRED_STRENGTH_PSI)
        )
        review_flags["RequiredStrengthInvalid"] = required_invalid.astype(int)
        df.loc[required_invalid, REQUIRED_STRENGTH] = np.nan
        review_flags["RequiredStrengthBelow1000Psi_Review"] = (
            df[REQUIRED_STRENGTH].between(0, 1_000, inclusive="neither")
        ).astype(int)
    else:
        df[REQUIRED_STRENGTH] = np.nan

    # A base row needs at least one core fresh-concrete measurement after
    # cleaning. Missing fields are allowed and will be imputed during training.
    available_core_columns = [
        column for column in CORE_FRESH_MEASUREMENTS if column in df.columns
    ]
    if not available_core_columns:
        raise KeyError(
            "None of the expected core fresh-measurement columns were found."
        )

    has_any_core_measurement = df[available_core_columns].notna().any(axis=1)
    exclusion_reasons = append_reason(
        exclusion_reasons,
        ~has_any_core_measurement,
        "NoUsableCoreFreshMeasurement",
    )

    df["CleaningExclusionReasons"] = exclusion_reasons
    df["HasAnyCoreMeasurementAfterCleaning"] = has_any_core_measurement.astype(int)

    if not review_flags.empty:
        df["CleaningReviewFlagCount"] = review_flags.sum(axis=1)
        df["HasCleaningReviewFlag"] = df["CleaningReviewFlagCount"].gt(0).astype(int)
        for column in review_flags.columns:
            df[column] = review_flags[column]
    else:
        df["CleaningReviewFlagCount"] = 0
        df["HasCleaningReviewFlag"] = 0

    clean_base_mask = df["CleaningExclusionReasons"].isna()
    clean_base = df.loc[clean_base_mask].copy()
    clean_with_required = clean_base.loc[
        clean_base[REQUIRED_STRENGTH].notna()
    ].copy()
    excluded = df.loc[~clean_base_mask].copy()

    # Save outputs.
    clean_base.to_csv(OUTPUT_DIR / "field_core_clean_base.csv", index=False)
    clean_with_required.to_csv(
        OUTPUT_DIR / "field_core_clean_with_required.csv",
        index=False,
    )
    excluded.to_csv(OUTPUT_DIR / "cleaning_exclusion_audit.csv", index=False)

    adjustments = pd.DataFrame(adjustment_rows)
    adjustments.to_csv(
        OUTPUT_DIR / "cleaning_feature_adjustment_summary.csv",
        index=False,
    )

    review_columns = [
        "testId",
        "OfficeName",
        "projectNo",
        "castDate",
        TARGET,
        REQUIRED_STRENGTH,
        "CleaningReviewFlagCount",
    ]
    review_columns += list(review_flags.columns)
    review_columns = [column for column in review_columns if column in df.columns]
    df.loc[df["HasCleaningReviewFlag"].eq(1), review_columns].to_csv(
        OUTPUT_DIR / "cleaning_review_rows.csv",
        index=False,
    )

    summary = {
        "input_records": int(len(df)),
        "unique_tests": int(df["testId"].nunique(dropna=True)),
        "clean_base_records": int(len(clean_base)),
        "clean_with_required_records": int(len(clean_with_required)),
        "excluded_records": int(len(excluded)),
        "records_with_review_flags": int(df["HasCleaningReviewFlag"].sum()),
        "expected_unit_system_id": EXPECTED_UNIT_SYSTEM_ID,
        "minimum_valid_date": MIN_VALID_DATE.date().isoformat(),
        "maximum_valid_date": MAX_VALID_DATE.date().isoformat(),
        "maximum_target_psi": MAX_TARGET_PSI,
        "maximum_required_strength_psi": MAX_REQUIRED_STRENGTH_PSI,
    }
    (OUTPUT_DIR / "cleaning_run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    report = f"""# Field Core Cleaning Report

## Results

- Input records: **{len(df):,}**
- Clean base records: **{len(clean_base):,}**
- Clean records with applicable required strength: **{len(clean_with_required):,}**
- Excluded records: **{len(excluded):,}**
- Records retained but marked for review: **{int(df['HasCleaningReviewFlag'].sum()):,}**

## Cleaning behavior

- Invalid target, duplicate test ID, non-US unit system, invalid cast date, or no usable core fresh measurement removes the row.
- Out-of-range feature values are changed to missing rather than deleting the row.
- Missing feature values are retained for later median imputation with missing indicators.
- A missing or invalid required strength excludes a row only from `field_core_clean_with_required.csv`; it can remain in the field-only dataset.
- The ranges in `cleaning_feature_adjustment_summary.csv` are broad sanity limits, not acceptance specifications.

## Main outputs

- `field_core_clean_base.csv`
- `field_core_clean_with_required.csv`
- `cleaning_exclusion_audit.csv`
- `cleaning_review_rows.csv`
- `cleaning_feature_adjustment_summary.csv`
"""
    (OUTPUT_DIR / "cleaning_report.md").write_text(report, encoding="utf-8")

    print("Step 1 completed.")
    print(f"Input records: {len(df):,}")
    print(f"Clean base records: {len(clean_base):,}")
    print(f"Clean with required strength: {len(clean_with_required):,}")
    print(f"Excluded records: {len(excluded):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
