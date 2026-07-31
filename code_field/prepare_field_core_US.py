#!/usr/bin/env python3
"""
Profile and prepare the IMTS US-unit Field Core concrete dataset.

This script is designed for the CSV produced by:
    imts_field_core_predictive_28day_us_units.sql

It does NOT train a machine-learning model. It performs the data-readiness work
that should happen first:

1. Confirms row count, unique concrete-test count, and duplicate test IDs.
2. Confirms the US unit system (ConcreteTestUnitSystem = 0), when available.
3. Profiles missing values and numeric distributions.
4. Audits standard-cured versus field-cured 7-day and 28-day results.
5. Identifies why a 28-day target is unavailable.
6. Audits legacy N/A ambiguity, especially zero values created before N/A flags
   were added to IMTS.
7. Profiles initial-curing free text without using it as a model feature yet.
8. Writes model-candidate CSV files for the first Field Core model.
9. Saves a Markdown report, audit CSV files, and a few simple charts.

Important curing rule
---------------------
AverageActualStrength7_psi and AverageActualStrength28_psi are expected to have
been calculated by SQL from STANDARD-CURED specimens only
(FieldConcreteTestRows.wasFieldCured = 0).

Field-cured specimens are not treated as the primary 28-day target. They are
reported separately using FieldCuredStrength7SpecimenCount and
FieldCuredStrength28SpecimenCount. A specimen marked field-cured measures the
jobsite/structure curing condition and should be analyzed separately.

Legacy N/A rule
---------------
Some IMTS N/A flags were added after historical data already existed. Therefore:

* raw value = 0 and N/A flag = NULL can be ambiguous;
* this script reports those records instead of automatically changing every zero
  to missing;
* zero slump can be physically valid, and zero water added can mean no water was
  added, so blanket replacement of zero with NaN would be unsafe;
* physically impossible values such as unit weight <= 0 and load volume <= 0 are
  flagged for review.

Install dependencies
--------------------
    pip install pandas numpy matplotlib openpyxl

Run
---
    python prepare_field_core_US.py

No command-line arguments are required. The input CSV and output directory are
hardcoded below relative to the folder containing this Python file.
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


US_UNIT_SYSTEM_ID = 0
DEFAULT_MIN_VALID_DATE = "2000-01-01"

# -----------------------------------------------------------------------------
# Hardcoded local run configuration
# -----------------------------------------------------------------------------
# The script can be started without command-line arguments:
#     python prepare_field_core_US.py
#
# These paths are resolved relative to this Python file, so the script works
# even when it is launched from a different current working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / "data" / "field_core_US_data.csv"
OUTPUT_DIR = SCRIPT_DIR / "data" / "prepared_field_core_US"

CSV_ENCODING: str | None = None  # Auto-detect UTF-8/UTF-8-SIG/CP1252/Latin-1.
MIN_VALID_DATE = DEFAULT_MIN_VALID_DATE
MAX_VALID_DATE: str | None = None  # None means the current date/time.
EXPECTED_UNIT_SYSTEM_ID = US_UNIT_SYSTEM_ID  # IMTS: 0 = US customary units.

# First Field Core model: measurements available at or near placement time.
BASE_FIELD_NUMERIC_FEATURES = [
    "EffectiveSlump_in",
    "EffectiveAir_percent",
    "EffectiveUnitWeight_lb_ft3",
    "EffectiveConcreteTemp_F",
    "AmbientTemp_F",
    "WaterAdded_gal_per_yd3",
    "BatchToSampleMinutes",
    "BatchToCastMinutes",
]

BASE_FIELD_INDICATOR_FEATURES = [
    "HasWaterAdded",
    "HasAnyAfterSPMeasurement",
]

# Spread is useful for SCC-type tests but is not mixed into the basic slump
# feature. It is retained as an optional field and profiled separately.
OPTIONAL_FIELD_FEATURES = [
    "EffectiveSpread_in",
]

MODEL_TARGET = "AverageActualStrength28_psi"
REQUIRED_STRENGTH_FEATURE = "ApplicableSpecifiedStrength28"

IDENTIFIER_AND_AUDIT_COLUMNS = [
    "officeId",
    "OfficeName",
    "projectId",
    "projectNo",
    "SampleId",
    "labNo",
    "testId",
    "ConcreteTestDataId",
    "testSubTypeId",
    "castDate",
    "supplierId",
    "SupplierName",
    "plantNumber",
    "mixNumber",
    "ConcreteTestUnitSystem",
]

TARGET_AUDIT_COLUMNS = [
    "RequiredStrength28",
    "DesignStrength28",
    "ApplicableSpecifiedStrength28",
    "ApplicableStrengthType28",
    "AverageActualStrength7_psi",
    "ActualStrength7SpecimenCount",
    "AverageActualStrength28_psi",
    "ActualStrength28SpecimenCount",
    "ActualStrengthRange28_psi",
    "FieldCuredStrength7SpecimenCount",
    "FieldCuredStrength28SpecimenCount",
    "StrengthMargin28_psi",
    "FailureFlag28",
]

CATEGORICAL_PROFILE_COLUMNS = [
    "OfficeName",
    "testSubTypeId",
    "placementType",
    "sampledFrom",
    "CloudType",
    "PrecipitationType",
    "WindType",
    "ApplicableStrengthType28",
    "SlumpMeasurementSource",
    "AirMeasurementSource",
    "UnitWeightMeasurementSource",
    "ConcreteTempMeasurementSource",
    "SpreadMeasurementSource",
    "FinalCure_AuditOnly",
]

# Raw numeric field and its later-added N/A flag. Missing pairs are skipped.
LEGACY_NA_PAIRS = [
    ("WaterAddedRaw", "waterAddedNA", "Water added"),
    ("LoadBatchVolumeRaw", "LoadBatchVolumeNA", "Load/batch volume"),
    ("AmbientTempRaw", "ambientTempNA", "Ambient temperature"),
    ("uwSlump_actual", "uwSlump_actualNA", "Slump actual"),
    ("uwSlump_afterSP", "uwSlump_afterSPNA", "Slump after SP"),
    ("uwSlump_specMin", "uwSlump_specMinNA", "Slump spec minimum"),
    ("uwSlump_specMax", "uwSlump_specMaxNA", "Slump spec maximum"),
    ("uwSpread_actual", "uwSpread_actualNA", "Spread actual"),
    ("uwSpread_afterSP", "uwSpread_afterSPNA", "Spread after SP"),
    ("uwSpread_specMin", "uwSpread_specMinNA", "Spread spec minimum"),
    ("uwSpread_specMax", "uwSpread_specMaxNA", "Spread spec maximum"),
    ("uwAir_actual", "uwAir_actualNA", "Air actual"),
    ("uwAir_afterSP", "uwAir_afterSPNA", "Air after SP"),
    ("uwAir_specMin", "uwAir_specMinNA", "Air spec minimum"),
    ("uwAir_specMax", "uwAir_specMaxNA", "Air spec maximum"),
    ("uwWeight_actual", "uwWeight_actualNA", "Unit weight actual"),
    ("uwWeight_afterSP", "uwWeight_afterSPNA", "Unit weight after SP"),
    ("uwWeight_specMin", "uwWeight_specMinNA", "Unit weight spec minimum"),
    ("uwWeight_specMax", "uwWeight_specMaxNA", "Unit weight spec maximum"),
    ("uwConcreteTemp_actual", "uwConcreteTemp_actualNA", "Concrete temperature actual"),
    ("uwConcreteTemp_afterSP", "uwConcreteTemp_afterSPNA", "Concrete temperature after SP"),
    ("uwConcreteTemp_specMin", "uwConcreteTemp_specMinNA", "Concrete temperature spec minimum"),
    ("uwConcreteTemp_specMax", "uwConcreteTemp_specMaxNA", "Concrete temperature spec maximum"),
]


@dataclass(frozen=True)
class ResolvedColumns:
    test_id: str | None
    cast_date: str | None
    unit_system: str | None
    target_28: str | None
    target_7: str | None
    required_28: str | None
    standard_count_28: str | None
    standard_count_7: str | None
    field_count_28: str | None
    field_count_7: str | None


def read_input(path: Path, csv_encoding: str | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path)

    if suffix not in {".csv", ".txt"}:
        raise ValueError("Input file must be CSV, TXT, XLSX, XLSM, or XLS.")

    if csv_encoding:
        return pd.read_csv(path, encoding=csv_encoding, low_memory=False)

    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            last_error = exc

    if last_error:
        raise last_error
    raise RuntimeError("Could not read the input file.")


def find_column(df: pd.DataFrame, aliases: Sequence[str]) -> str | None:
    lookup = {str(column).strip().lower(): str(column) for column in df.columns}
    for alias in aliases:
        match = lookup.get(alias.strip().lower())
        if match is not None:
            return match
    return None


def resolve_columns(df: pd.DataFrame) -> ResolvedColumns:
    return ResolvedColumns(
        test_id=find_column(df, ["testId", "TestId", "ConcreteTestId"]),
        cast_date=find_column(df, ["castDate", "CastDate"]),
        unit_system=find_column(
            df,
            ["ConcreteTestUnitSystem", "unitSystem", "unit_system"],
        ),
        target_28=find_column(
            df,
            ["AverageActualStrength28_psi", "AverageActualStrength28"],
        ),
        target_7=find_column(
            df,
            ["AverageActualStrength7_psi", "AverageActualStrength7"],
        ),
        required_28=find_column(
            df,
            ["ApplicableSpecifiedStrength28", "RequiredStrength28"],
        ),
        standard_count_28=find_column(
            df,
            ["ActualStrength28SpecimenCount", "StandardCuredStrength28SpecimenCount"],
        ),
        standard_count_7=find_column(
            df,
            ["ActualStrength7SpecimenCount", "StandardCuredStrength7SpecimenCount"],
        ),
        field_count_28=find_column(df, ["FieldCuredStrength28SpecimenCount"]),
        field_count_7=find_column(df, ["FieldCuredStrength7SpecimenCount"]),
    )


def normalize_blank_strings(df: pd.DataFrame) -> None:
    object_columns = df.select_dtypes(include=["object", "string"]).columns
    for column in object_columns:
        series = df[column].astype("string")
        stripped = series.str.strip()
        df[column] = stripped.mask(stripped.eq(""), pd.NA)


def coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")


def parse_flag(series: pd.Series) -> pd.Series:
    """Convert common bit/boolean representations to nullable Float64 0/1."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype("Int64").astype("Float64")

    numeric = pd.to_numeric(series, errors="coerce")
    result = numeric.where(numeric.isin([0, 1]))

    unresolved = result.isna() & series.notna()
    if unresolved.any():
        text = series.astype("string").str.strip().str.lower()
        mapped = text.map(
            {
                "true": 1.0,
                "false": 0.0,
                "yes": 1.0,
                "no": 0.0,
                "y": 1.0,
                "n": 0.0,
            }
        )
        result = result.where(~unresolved, mapped)

    return result.astype("Float64")


def safe_percent(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return float("nan")
    return round(100.0 * float(numerator) / float(denominator), 2)


def present_mask(df: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None or column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column].notna()


def numeric_present_mask(df: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None or column not in df.columns:
        return pd.Series(False, index=df.index)
    values = pd.to_numeric(df[column], errors="coerce")
    return values.notna()


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def column_missingness_profile(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    rows: list[dict[str, object]] = []

    for column in df.columns:
        series = df[column]
        missing_count = int(series.isna().sum())
        unique_non_null = int(series.nunique(dropna=True))

        numeric = pd.to_numeric(series, errors="coerce")
        numeric_non_null = numeric.notna().sum()
        zero_count = int(numeric.eq(0).sum()) if numeric_non_null else 0

        rows.append(
            {
                "Column": column,
                "DType": str(series.dtype),
                "RowCount": total,
                "NonMissingCount": total - missing_count,
                "MissingCount": missing_count,
                "MissingPercent": safe_percent(missing_count, total),
                "UniqueNonNullCount": unique_non_null,
                "ZeroCountNumericInterpretation": zero_count,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["MissingPercent", "Column"], ascending=[False, True]
    )


def numeric_feature_profile(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total = len(df)

    for column in columns:
        if column not in df.columns:
            rows.append(
                {
                    "Column": column,
                    "PresentInFile": 0,
                    "RowCount": total,
                }
            )
            continue

        values = pd.to_numeric(df[column], errors="coerce")
        non_missing = values.dropna()

        row: dict[str, object] = {
            "Column": column,
            "PresentInFile": 1,
            "RowCount": total,
            "NonMissingCount": int(non_missing.size),
            "MissingCount": int(values.isna().sum()),
            "MissingPercent": safe_percent(int(values.isna().sum()), total),
            "ZeroCount": int(values.eq(0).sum()),
            "NegativeCount": int(values.lt(0).sum()),
        }

        if not non_missing.empty:
            quantiles = non_missing.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
            row.update(
                {
                    "Min": float(non_missing.min()),
                    "P01": float(quantiles.loc[0.01]),
                    "P05": float(quantiles.loc[0.05]),
                    "P25": float(quantiles.loc[0.25]),
                    "Median": float(quantiles.loc[0.5]),
                    "Mean": float(non_missing.mean()),
                    "P75": float(quantiles.loc[0.75]),
                    "P95": float(quantiles.loc[0.95]),
                    "P99": float(quantiles.loc[0.99]),
                    "Max": float(non_missing.max()),
                    "Std": float(non_missing.std(ddof=1)) if non_missing.size > 1 else np.nan,
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def clean_initial_curing_text(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip().str.lower()
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True)
    cleaned = cleaned.str.replace(r"[^a-z0-9%+./ -]", "", regex=True)
    return cleaned.mask(cleaned.eq(""), pd.NA)


def initial_curing_profile(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    raw_col = find_column(
        df,
        ["InitialCuringConditionRaw", "initialCuringCondition"],
    )
    clean_col = find_column(df, ["InitialCuringConditionClean"])

    if clean_col is not None:
        cleaned = clean_initial_curing_text(df[clean_col])
    elif raw_col is not None:
        cleaned = clean_initial_curing_text(df[raw_col])
    else:
        return pd.DataFrame(), {
            "InitialCuringColumnAvailable": False,
            "InitialCuringNonMissingCount": 0,
            "InitialCuringUniqueCleanCount": 0,
        }

    counts = (
        cleaned.fillna("<missing>")
        .value_counts(dropna=False)
        .rename_axis("InitialCuringConditionNormalizedText")
        .reset_index(name="TestCount")
    )
    counts["Percent"] = counts["TestCount"].map(lambda n: safe_percent(n, len(df)))

    stats = {
        "InitialCuringColumnAvailable": True,
        "InitialCuringNonMissingCount": int(cleaned.notna().sum()),
        "InitialCuringMissingCount": int(cleaned.isna().sum()),
        "InitialCuringUniqueCleanCount": int(cleaned.nunique(dropna=True)),
        "InitialCuringSingleOccurrenceValueCount": int(
            (cleaned.value_counts(dropna=True) == 1).sum()
        ),
    }
    return counts, stats


def legacy_na_audit(
    df: pd.DataFrame,
    cast_date_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    rows: list[dict[str, object]] = []
    yearly_frames: list[pd.DataFrame] = []
    row_ambiguity_count = pd.Series(0, index=df.index, dtype="int64")

    cast_year: pd.Series | None = None
    if cast_date_col and cast_date_col in df.columns:
        cast_year = pd.to_datetime(df[cast_date_col], errors="coerce").dt.year

    for raw_col, flag_col, label in LEGACY_NA_PAIRS:
        if raw_col not in df.columns and flag_col not in df.columns:
            continue

        raw = (
            pd.to_numeric(df[raw_col], errors="coerce")
            if raw_col in df.columns
            else pd.Series(np.nan, index=df.index)
        )
        flag = (
            parse_flag(df[flag_col])
            if flag_col in df.columns
            else pd.Series(pd.NA, index=df.index, dtype="Float64")
        )

        raw_zero = raw.eq(0)
        flag_null = flag.isna()
        flag_zero = flag.eq(0).fillna(False)
        flag_one = flag.eq(1).fillna(False)

        ambiguous_zero = raw_zero & flag_null
        row_ambiguity_count = row_ambiguity_count.add(
            ambiguous_zero.astype("int64"), fill_value=0
        ).astype("int64")

        rows.append(
            {
                "Field": label,
                "RawColumn": raw_col if raw_col in df.columns else None,
                "NAFlagColumn": flag_col if flag_col in df.columns else None,
                "TotalRows": len(df),
                "RawNonMissingCount": int(raw.notna().sum()),
                "RawMissingCount": int(raw.isna().sum()),
                "NAFlagPopulatedCount": int(flag.notna().sum()),
                "NAFlagNullCount": int(flag_null.sum()),
                "NAFlagTrueCount": int(flag_one.sum()),
                "NAFlagFalseCount": int(flag_zero.sum()),
                "RawZeroCount": int(raw_zero.sum()),
                "ZeroWithNAFlagTrueCount": int((raw_zero & flag_one).sum()),
                "ZeroWithNAFlagFalseCount": int((raw_zero & flag_zero).sum()),
                "ZeroWithNAFlagNull_AmbiguousCount": int(ambiguous_zero.sum()),
                "PositiveValueWithNAFlagTrue_ConflictCount": int(
                    (raw.gt(0) & flag_one).sum()
                ),
                "MissingRawWithNAFlagFalseCount": int((raw.isna() & flag_zero).sum()),
            }
        )

        if cast_year is not None:
            temp = pd.DataFrame(
                {
                    "CastYear": cast_year,
                    "RawPresent": raw.notna().astype(int),
                    "NAFlagPresent": flag.notna().astype(int),
                    "NAFlagTrue": flag_one.astype(int),
                    "RawZero": raw_zero.astype(int),
                    "ZeroWithNAFlagNull_Ambiguous": ambiguous_zero.astype(int),
                }
            ).dropna(subset=["CastYear"])

            if not temp.empty:
                grouped = (
                    temp.groupby("CastYear", as_index=False)
                    .agg(
                        RowCount=("RawPresent", "size"),
                        RawPresentCount=("RawPresent", "sum"),
                        NAFlagPresentCount=("NAFlagPresent", "sum"),
                        NAFlagTrueCount=("NAFlagTrue", "sum"),
                        RawZeroCount=("RawZero", "sum"),
                        ZeroWithNAFlagNull_AmbiguousCount=(
                            "ZeroWithNAFlagNull_Ambiguous",
                            "sum",
                        ),
                    )
                    .assign(Field=label, RawColumn=raw_col, NAFlagColumn=flag_col)
                )
                yearly_frames.append(grouped)

    summary = pd.DataFrame(rows)
    yearly = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    return summary, yearly, row_ambiguity_count


def curing_profile(df: pd.DataFrame, cols: ResolvedColumns) -> tuple[pd.DataFrame, pd.DataFrame]:
    std28 = (
        pd.to_numeric(df[cols.standard_count_28], errors="coerce").fillna(0).gt(0)
        if cols.standard_count_28
        else present_mask(df, cols.target_28)
    )
    fld28 = (
        pd.to_numeric(df[cols.field_count_28], errors="coerce").fillna(0).gt(0)
        if cols.field_count_28
        else pd.Series(False, index=df.index)
    )

    std7 = (
        pd.to_numeric(df[cols.standard_count_7], errors="coerce").fillna(0).gt(0)
        if cols.standard_count_7
        else present_mask(df, cols.target_7)
    )
    fld7 = (
        pd.to_numeric(df[cols.field_count_7], errors="coerce").fillna(0).gt(0)
        if cols.field_count_7
        else pd.Series(False, index=df.index)
    )

    def status(std: pd.Series, fld: pd.Series) -> pd.Series:
        return pd.Series(
            np.select(
                [std & fld, std & ~fld, ~std & fld],
                ["Both standard and field cured", "Standard cured only", "Field cured only"],
                default="No strength result",
            ),
            index=df.index,
            dtype="string",
        )

    status28 = status(std28, fld28)
    status7 = status(std7, fld7)

    summary_rows: list[dict[str, object]] = []
    for age, series in ((7, status7), (28, status28)):
        counts = series.value_counts(dropna=False)
        for category in [
            "Standard cured only",
            "Field cured only",
            "Both standard and field cured",
            "No strength result",
        ]:
            count = int(counts.get(category, 0))
            summary_rows.append(
                {
                    "BreakAgeDays": age,
                    "CuringResultAvailability": category,
                    "ConcreteTestCount": count,
                    "Percent": safe_percent(count, len(df)),
                }
            )

    row_level = pd.DataFrame(
        {
            "CuringAvailability7": status7,
            "CuringAvailability28": status28,
            "HasStandardCured7": std7.astype(int),
            "HasFieldCured7": fld7.astype(int),
            "HasStandardCured28": std28.astype(int),
            "HasFieldCured28": fld28.astype(int),
        },
        index=df.index,
    )
    return pd.DataFrame(summary_rows), row_level


def categorical_profiles(df: pd.DataFrame, output_dir: Path) -> None:
    category_dir = output_dir / "categorical_profiles"
    category_dir.mkdir(parents=True, exist_ok=True)

    for column in CATEGORICAL_PROFILE_COLUMNS:
        if column not in df.columns:
            continue
        counts = (
            df[column]
            .astype("string")
            .fillna("<missing>")
            .value_counts(dropna=False)
            .rename_axis("Value")
            .reset_index(name="TestCount")
        )
        counts["Percent"] = counts["TestCount"].map(lambda n: safe_percent(n, len(df)))
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", column)
        write_dataframe(counts, category_dir / f"{safe_name}_frequency.csv")


def curing_breakdowns(
    df: pd.DataFrame,
    curing_rows: pd.DataFrame,
    cols: ResolvedColumns,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.DataFrame(index=df.index)
    combined["CuringAvailability7"] = curing_rows["CuringAvailability7"]
    combined["CuringAvailability28"] = curing_rows["CuringAvailability28"]

    if cols.cast_date and cols.cast_date in df.columns:
        combined["CastYear"] = pd.to_datetime(df[cols.cast_date], errors="coerce").dt.year
        by_year = (
            combined.dropna(subset=["CastYear"])
            .groupby(["CastYear", "CuringAvailability28"], as_index=False)
            .size()
            .rename(columns={"size": "ConcreteTestCount"})
        )
        year_totals = by_year.groupby("CastYear")["ConcreteTestCount"].transform("sum")
        by_year["PercentWithinYear"] = (
            100.0 * by_year["ConcreteTestCount"] / year_totals
        ).round(2)
    else:
        by_year = pd.DataFrame()

    office_col = find_column(df, ["OfficeName", "officeName", "officeId"])
    if office_col:
        combined["Office"] = df[office_col].astype("string").fillna("<missing>")
        by_office = (
            combined.groupby(["Office", "CuringAvailability28"], as_index=False)
            .size()
            .rename(columns={"size": "ConcreteTestCount"})
        )
        office_totals = by_office.groupby("Office")["ConcreteTestCount"].transform("sum")
        by_office["PercentWithinOffice"] = (
            100.0 * by_office["ConcreteTestCount"] / office_totals
        ).round(2)
    else:
        by_office = pd.DataFrame()

    return by_year, by_office


def test_subtype_readiness_profile(
    df: pd.DataFrame,
    cols: ResolvedColumns,
    curing_rows: pd.DataFrame,
) -> pd.DataFrame:
    subtype_col = find_column(df, ["testSubTypeId", "TestSubTypeId"])
    if subtype_col is None:
        return pd.DataFrame()

    work = pd.DataFrame(index=df.index)
    work["testSubTypeId"] = df[subtype_col].astype("string").fillna("<missing>")
    work["HasActual28"] = numeric_present_mask(df, cols.target_28).astype(int)
    work["HasActual7"] = numeric_present_mask(df, cols.target_7).astype(int)
    work["HasRequired28"] = numeric_present_mask(df, cols.required_28).astype(int)
    work["HasStandardCured28"] = curing_rows["HasStandardCured28"].astype(int)
    work["HasFieldCured28"] = curing_rows["HasFieldCured28"].astype(int)

    fresh_columns = [
        column
        for column in [
            "EffectiveSlump_in",
            "EffectiveSpread_in",
            "EffectiveAir_percent",
            "EffectiveUnitWeight_lb_ft3",
            "EffectiveConcreteTemp_F",
        ]
        if column in df.columns
    ]
    if fresh_columns:
        work["HasAnyFreshMeasurement"] = (
            df[fresh_columns].apply(pd.to_numeric, errors="coerce").notna().any(axis=1).astype(int)
        )
    else:
        work["HasAnyFreshMeasurement"] = 0

    grouped = (
        work.groupby("testSubTypeId", as_index=False)
        .agg(
            ConcreteTestCount=("testSubTypeId", "size"),
            TestsWithActual7=("HasActual7", "sum"),
            TestsWithActual28=("HasActual28", "sum"),
            TestsWithRequired28=("HasRequired28", "sum"),
            TestsWithAnyFreshMeasurement=("HasAnyFreshMeasurement", "sum"),
            TestsWithStandardCured28=("HasStandardCured28", "sum"),
            TestsWithFieldCured28=("HasFieldCured28", "sum"),
        )
        .sort_values("ConcreteTestCount", ascending=False)
    )
    grouped["Actual28CoveragePercent"] = (
        100.0 * grouped["TestsWithActual28"] / grouped["ConcreteTestCount"]
    ).round(2)
    grouped["FreshMeasurementCoveragePercent"] = (
        100.0 * grouped["TestsWithAnyFreshMeasurement"] / grouped["ConcreteTestCount"]
    ).round(2)
    grouped["FieldCured28Percent"] = (
        100.0 * grouped["TestsWithFieldCured28"] / grouped["ConcreteTestCount"]
    ).round(2)
    return grouped


def build_suspicious_value_flags(df: pd.DataFrame) -> pd.DataFrame:
    flags = pd.DataFrame(index=df.index)

    def numeric(column: str) -> pd.Series:
        if column not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return pd.to_numeric(df[column], errors="coerce")

    flags["NonPositive28DayStrength"] = numeric(MODEL_TARGET).le(0).fillna(False)
    flags["NonPositiveRequiredStrength"] = numeric(REQUIRED_STRENGTH_FEATURE).le(0).fillna(False)
    flags["NegativeSlump"] = numeric("EffectiveSlump_in").lt(0).fillna(False)
    flags["NegativeSpread"] = numeric("EffectiveSpread_in").lt(0).fillna(False)
    flags["NegativeAir"] = numeric("EffectiveAir_percent").lt(0).fillna(False)
    flags["NonPositiveUnitWeight"] = numeric("EffectiveUnitWeight_lb_ft3").le(0).fillna(False)
    flags["NegativeWaterAddedPerVolume"] = numeric("WaterAdded_gal_per_yd3").lt(0).fillna(False)
    flags["NonPositiveLoadBatchVolume"] = numeric("LoadBatchVolumeRaw").le(0).fillna(False)
    flags["BatchToSampleOver24Hours"] = numeric("BatchToSampleMinutes").gt(1440).fillna(False)
    flags["BatchToCastOver24Hours"] = numeric("BatchToCastMinutes").gt(1440).fillna(False)

    numeric_target = numeric(MODEL_TARGET).dropna()
    if len(numeric_target) >= 8:
        q1 = numeric_target.quantile(0.25)
        q3 = numeric_target.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 3.0 * iqr
        upper = q3 + 3.0 * iqr
        target_values = numeric(MODEL_TARGET)
        flags["Extreme28DayStrength_IQRReview"] = (
            target_values.lt(lower) | target_values.gt(upper)
        ).fillna(False)
    else:
        flags["Extreme28DayStrength_IQRReview"] = False

    flags = flags.astype(bool)
    flags["SuspiciousValueFlagCount"] = flags.sum(axis=1)
    flags["HasSuspiciousValue"] = flags["SuspiciousValueFlagCount"].gt(0)
    return flags


def prepare_model_candidates(
    df: pd.DataFrame,
    cols: ResolvedColumns,
    min_valid_date: pd.Timestamp,
    max_valid_date: pd.Timestamp,
    expected_unit_system: int,
    curing_rows: pd.DataFrame,
    legacy_ambiguity_count: pd.Series,
    suspicious_flags: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = df.copy()

    # Canonical date and unit checks.
    if cols.cast_date:
        cast_date = pd.to_datetime(work[cols.cast_date], errors="coerce")
    else:
        cast_date = pd.Series(pd.NaT, index=work.index, dtype="datetime64[ns]")

    valid_date = cast_date.between(min_valid_date, max_valid_date, inclusive="both")

    if cols.unit_system:
        unit_values = pd.to_numeric(work[cols.unit_system], errors="coerce")
        valid_unit = unit_values.eq(expected_unit_system)
    else:
        valid_unit = pd.Series(True, index=work.index)

    target = (
        pd.to_numeric(work[cols.target_28], errors="coerce")
        if cols.target_28
        else pd.Series(np.nan, index=work.index)
    )
    target_available = target.notna() & target.gt(0)

    required = (
        pd.to_numeric(work[cols.required_28], errors="coerce")
        if cols.required_28
        else pd.Series(np.nan, index=work.index)
    )
    required_available = required.notna() & required.gt(0)

    fresh_columns = [
        column
        for column in [
            "EffectiveSlump_in",
            "EffectiveSpread_in",
            "EffectiveAir_percent",
            "EffectiveUnitWeight_lb_ft3",
            "EffectiveConcreteTemp_F",
        ]
        if column in work.columns
    ]
    if fresh_columns:
        fresh_count = work[fresh_columns].apply(pd.to_numeric, errors="coerce").notna().sum(axis=1)
    else:
        fresh_count = pd.Series(0, index=work.index)
    any_fresh = fresh_count.gt(0)

    # Duplicate IDs are excluded from the prepared model files until investigated.
    if cols.test_id:
        duplicate_test = work[cols.test_id].duplicated(keep=False)
    else:
        duplicate_test = pd.Series(False, index=work.index)

    work["ParsedCastDate"] = cast_date
    work["AvailableFreshMeasurementCount"] = fresh_count
    work["HasLegacyZeroNAAmbiguity"] = legacy_ambiguity_count.gt(0).astype(int)
    work["LegacyZeroNAAmbiguityCount"] = legacy_ambiguity_count

    for column in curing_rows.columns:
        work[column] = curing_rows[column]
    for column in suspicious_flags.columns:
        work[column] = suspicious_flags[column]

    work["ValidUSUnitSystem"] = valid_unit.fillna(False).astype(int)
    work["ValidCastDateForModel"] = valid_date.fillna(False).astype(int)
    work["ValidStandardCured28Target"] = target_available.astype(int)
    work["HasRequiredStrength28"] = required_available.astype(int)
    work["HasAnyCoreFreshMeasurement"] = any_fresh.astype(int)
    work["DuplicateTestId"] = duplicate_test.astype(int)

    base_eligible = valid_unit & valid_date & target_available & any_fresh & ~duplicate_test
    required_eligible = base_eligible & required_available

    work["FieldCoreBaseEligible"] = base_eligible.astype(int)
    work["FieldCoreWithRequiredEligible"] = required_eligible.astype(int)

    reason_masks = {
        "DuplicateTestId": duplicate_test,
        "NonUSUnitSystem": ~valid_unit.fillna(False),
        "InvalidOrMissingCastDate": ~valid_date.fillna(False),
        "MissingOrNonPositiveStandardCured28Strength": ~target_available,
        "NoEffectiveFreshMeasurement": ~any_fresh,
    }

    exclusion_reason = pd.Series("", index=work.index, dtype="string")
    for reason, mask in reason_masks.items():
        exclusion_reason = exclusion_reason.mask(
            mask,
            exclusion_reason.where(exclusion_reason.eq(""), exclusion_reason + ";") + reason,
        )
    work["BaseExclusionReasons"] = exclusion_reason.mask(exclusion_reason.eq(""), pd.NA)

    # Do not silently modify ambiguous historical zero values. Model-ready files
    # keep them and expose the audit flags. A later domain-reviewed rule can change them.
    feature_columns = [
        column
        for column in BASE_FIELD_NUMERIC_FEATURES
        + BASE_FIELD_INDICATOR_FEATURES
        + OPTIONAL_FIELD_FEATURES
        if column in work.columns
    ]
    id_columns = [column for column in IDENTIFIER_AND_AUDIT_COLUMNS if column in work.columns]
    target_columns = [column for column in TARGET_AUDIT_COLUMNS if column in work.columns]
    extra_audit = [
        "ParsedCastDate",
        "AvailableFreshMeasurementCount",
        "HasLegacyZeroNAAmbiguity",
        "LegacyZeroNAAmbiguityCount",
        "CuringAvailability7",
        "CuringAvailability28",
        "HasStandardCured7",
        "HasFieldCured7",
        "HasStandardCured28",
        "HasFieldCured28",
        "HasSuspiciousValue",
        "SuspiciousValueFlagCount",
        "FieldCoreBaseEligible",
        "FieldCoreWithRequiredEligible",
    ]
    extra_audit = [column for column in extra_audit if column in work.columns]

    selected = list(dict.fromkeys(id_columns + feature_columns + target_columns + extra_audit))
    base = work.loc[base_eligible, selected].copy()
    with_required = work.loc[required_eligible, selected].copy()

    complete_feature_columns = [
        column
        for column in BASE_FIELD_NUMERIC_FEATURES + BASE_FIELD_INDICATOR_FEATURES
        if column in base.columns
    ]
    if complete_feature_columns:
        complete_mask = base[complete_feature_columns].notna().all(axis=1)
        complete = base.loc[complete_mask].copy()
    else:
        complete = base.iloc[0:0].copy()

    excluded_columns = list(dict.fromkeys(id_columns + ["BaseExclusionReasons"] + extra_audit))
    excluded = work.loc[~base_eligible, excluded_columns].copy()
    return base, with_required, complete, excluded


def create_overview(
    df: pd.DataFrame,
    cols: ResolvedColumns,
    curing_rows: pd.DataFrame,
    base: pd.DataFrame,
    with_required: pd.DataFrame,
    complete: pd.DataFrame,
    legacy_ambiguity_count: pd.Series,
    suspicious_flags: pd.DataFrame,
    min_valid_date: pd.Timestamp,
    max_valid_date: pd.Timestamp,
    expected_unit_system: int,
) -> pd.DataFrame:
    total = len(df)
    unique_tests = int(df[cols.test_id].nunique(dropna=True)) if cols.test_id else total
    duplicate_rows = int(df[cols.test_id].duplicated(keep=False).sum()) if cols.test_id else 0

    if cols.cast_date:
        cast_dates = pd.to_datetime(df[cols.cast_date], errors="coerce")
        valid_dates = cast_dates.between(min_valid_date, max_valid_date, inclusive="both")
    else:
        cast_dates = pd.Series(pd.NaT, index=df.index)
        valid_dates = pd.Series(False, index=df.index)

    if cols.unit_system:
        units = pd.to_numeric(df[cols.unit_system], errors="coerce")
        us_units = units.eq(expected_unit_system)
    else:
        us_units = pd.Series(True, index=df.index)

    actual7 = numeric_present_mask(df, cols.target_7)
    actual28 = numeric_present_mask(df, cols.target_28)
    required28 = numeric_present_mask(df, cols.required_28)

    core_columns = [
        column
        for column in [
            "EffectiveSlump_in",
            "EffectiveSpread_in",
            "EffectiveAir_percent",
            "EffectiveUnitWeight_lb_ft3",
            "EffectiveConcreteTemp_F",
        ]
        if column in df.columns
    ]
    if core_columns:
        core_values = df[core_columns].apply(pd.to_numeric, errors="coerce")
        any_core = core_values.notna().any(axis=1)
        all_core = core_values.notna().all(axis=1)
    else:
        any_core = pd.Series(False, index=df.index)
        all_core = pd.Series(False, index=df.index)

    rows: list[dict[str, object]] = []

    def add(metric: str, count: int | float | str, denominator: int | None = total, note: str = "") -> None:
        percent: float | str = ""
        if denominator is not None and isinstance(count, (int, float, np.integer, np.floating)):
            percent = safe_percent(float(count), denominator)
        rows.append(
            {
                "Metric": metric,
                "CountOrValue": count,
                "PercentOfAllRows": percent,
                "Note": note,
            }
        )

    add("Total records", total)
    add("Unique concrete tests", unique_tests)
    add("Rows belonging to duplicated test IDs", duplicate_rows)
    add("US unit-system rows", int(us_units.sum()), note=f"Expected unit system ID = {expected_unit_system}")
    add("Rows with valid cast date", int(valid_dates.sum()), note=f"Range {min_valid_date.date()} through {max_valid_date.date()}")
    add("Rows with actual 7-day standard-cured strength", int(actual7.sum()))
    add("Rows with actual 28-day standard-cured strength", int(actual28.sum()))
    add("Rows with applicable required/design 28-day strength", int(required28.sum()))
    add("Rows with any effective fresh measurement", int(any_core.sum()))
    add("Rows with all effective fresh measurements", int(all_core.sum()))
    add("Field Core base model candidates", len(base))
    add("Field Core candidates with required strength", len(with_required))
    add("Complete-case Field Core candidates", len(complete))
    add("Rows with at least one ambiguous legacy zero/N/A pair", int(legacy_ambiguity_count.gt(0).sum()))
    add("Rows with suspicious-value review flags", int(suspicious_flags["HasSuspiciousValue"].sum()))

    if "CuringAvailability28" in curing_rows.columns:
        counts = curing_rows["CuringAvailability28"].value_counts()
        add("28-day standard-cured only", int(counts.get("Standard cured only", 0)))
        add("28-day field-cured only", int(counts.get("Field cured only", 0)))
        add("28-day both standard and field cured", int(counts.get("Both standard and field cured", 0)))
        add("28-day no strength result", int(counts.get("No strength result", 0)))

    valid_non_null_dates = cast_dates[valid_dates]
    add(
        "Earliest valid cast date",
        valid_non_null_dates.min().date().isoformat() if not valid_non_null_dates.empty else "",
        denominator=None,
    )
    add(
        "Latest valid cast date",
        valid_non_null_dates.max().date().isoformat() if not valid_non_null_dates.empty else "",
        denominator=None,
    )

    return pd.DataFrame(rows)


def save_plots(
    df: pd.DataFrame,
    output_dir: Path,
    cols: ResolvedColumns,
    missingness: pd.DataFrame,
    curing_summary: pd.DataFrame,
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Target distribution.
    if cols.target_28 and cols.target_28 in df.columns:
        target = pd.to_numeric(df[cols.target_28], errors="coerce").dropna()
        if not target.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.hist(target, bins=50)
            ax.set_title("Standard-Cured 28-Day Strength Distribution")
            ax.set_xlabel("Average actual 28-day strength (psi)")
            ax.set_ylabel("Concrete test count")
            fig.tight_layout()
            fig.savefig(plot_dir / "actual_strength_28_distribution.png", dpi=160)
            plt.close(fig)

    # Top missing columns.
    if not missingness.empty:
        top = missingness.sort_values("MissingPercent", ascending=False).head(20).sort_values("MissingPercent")
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(top["Column"], top["MissingPercent"])
        ax.set_title("Top 20 Columns by Missing Percentage")
        ax.set_xlabel("Missing (%)")
        fig.tight_layout()
        fig.savefig(plot_dir / "top_20_missing_columns.png", dpi=160)
        plt.close(fig)

    # Tests by cast year.
    if cols.cast_date and cols.cast_date in df.columns:
        years = pd.to_datetime(df[cols.cast_date], errors="coerce").dt.year.dropna().astype(int)
        if not years.empty:
            counts = years.value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(counts.index, counts.values, marker="o")
            ax.set_title("Concrete Tests by Cast Year")
            ax.set_xlabel("Cast year")
            ax.set_ylabel("Concrete test count")
            fig.tight_layout()
            fig.savefig(plot_dir / "tests_by_cast_year.png", dpi=160)
            plt.close(fig)

    # 28-day curing availability.
    if not curing_summary.empty:
        subset = curing_summary[curing_summary["BreakAgeDays"].eq(28)].copy()
        if not subset.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.bar(subset["CuringResultAvailability"], subset["ConcreteTestCount"])
            ax.set_title("28-Day Strength Availability by Curing Type")
            ax.set_ylabel("Concrete test count")
            ax.tick_params(axis="x", rotation=20)
            fig.tight_layout()
            fig.savefig(plot_dir / "curing_availability_28_day.png", dpi=160)
            plt.close(fig)


def markdown_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df.empty:
        return "_No data available._"

    shown = df.head(max_rows).copy()
    columns = [str(column) for column in shown.columns]

    def format_value(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            if math.isfinite(value):
                return f"{value:,.2f}"
            return ""
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(format_value(row[column]) for column in shown.columns) + " |")
    return "\n".join(lines)


def write_markdown_report(
    output_dir: Path,
    input_file: Path,
    overview: pd.DataFrame,
    curing_summary: pd.DataFrame,
    model_profile: pd.DataFrame,
    initial_stats: dict[str, object],
    legacy_summary: pd.DataFrame,
) -> None:
    ambiguous_total = int(
        legacy_summary.get("ZeroWithNAFlagNull_AmbiguousCount", pd.Series(dtype=float)).sum()
    ) if not legacy_summary.empty else 0

    report = f"""# IMTS Field Core US Data Profile

**Input:** `{input_file}`  
**Generated:** `{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}`

## Purpose

This report evaluates readiness for the first model:

> Field measurements available at or near placement time → standard-cured 28-day compressive strength.

The primary target is `AverageActualStrength28_psi`. The SQL extract is expected to calculate this target from specimens where `wasFieldCured = 0` only.

## Dataset overview

{markdown_table(overview)}

## Curing status

`wasFieldCured` matters. A checked/true value means the specimen was field-cured. A false value means the specimen is treated as standard-cured for this extract. Standard-cured 28-day results are the primary model target; field-cured results are retained only for audit or a future separate model.

{markdown_table(curing_summary)}

## First-model feature coverage

{markdown_table(model_profile)}

## Initial curing text

The free-text initial-curing field is profiled but is **not automatically used as a model feature**. It should first be mapped to reviewed categories such as blanket, curing box, indoor, outdoor protected, and other. Final curing is not used in the placement-time model because it may be known later and could cause data leakage.

* Initial-curing column available: {initial_stats.get('InitialCuringColumnAvailable', False)}
* Non-missing initial-curing values: {initial_stats.get('InitialCuringNonMissingCount', 0):,}
* Unique normalized text values: {initial_stats.get('InitialCuringUniqueCleanCount', 0):,}
* Values occurring only once: {initial_stats.get('InitialCuringSingleOccurrenceValueCount', 0):,}

## Legacy N/A ambiguity

N/A flags such as slump N/A and load/batch-volume N/A were added after historical data already existed. A raw zero with a NULL N/A flag may mean a real zero, no entry, or historical N/A. This script does not silently convert all such zeros to missing.

Total ambiguous zero/NULL-flag occurrences across audited field pairs: **{ambiguous_total:,}**

Use `legacy_na_zero_audit.csv` and `legacy_na_flag_usage_by_year.csv` to identify when each N/A flag began being populated and to decide field-specific cleanup rules.

## Output files

* `dataset_overview.csv` — headline counts and model-readiness metrics
* `column_missingness.csv` — missing-value profile for every column
* `model_feature_profile.csv` — numeric distribution and missingness for first-model fields
* `curing_summary.csv` — standard-cured and field-cured result availability
* `curing_by_year.csv` and `curing_by_office.csv` — whether curing-checkbox usage changes by time or office
* `test_subtype_readiness.csv` — target, fresh-measurement, and curing coverage by test subtype
* `initial_curing_text_frequency.csv` — normalized free-text frequency
* `legacy_na_zero_audit.csv` — legacy zero/N/A ambiguity by field
* `legacy_na_flag_usage_by_year.csv` — flag usage and ambiguity by cast year
* `suspicious_value_summary.csv` and `suspicious_rows.csv` — values requiring review
* `field_core_model_candidates.csv` — base Field Core model candidate rows
* `field_core_model_candidates_with_required_strength.csv` — candidates with 28-day required/design strength
* `field_core_complete_cases.csv` — rows with every first-model feature populated
* `excluded_rows.csv` — records excluded from the base model and reasons

## What comes next

Profiling is necessary, but it is not the model test itself. After reviewing these outputs, the next step is to train and compare:

1. a dummy mean baseline;
2. field measurements only;
3. field measurements plus applicable 28-day required/design strength.

The split should be grouped by project or by supplier/plant/mix rather than a simple random row split.
"""
    (output_dir / "field_core_profile_report.md").write_text(report, encoding="utf-8")


def main() -> int:
    input_file = INPUT_FILE
    output_dir = OUTPUT_DIR
    csv_encoding = CSV_ENCODING
    expected_unit_system_id = EXPECTED_UNIT_SYSTEM_ID

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        min_valid_date = pd.Timestamp(MIN_VALID_DATE)
        max_valid_date = (
            pd.Timestamp.now()
            if MAX_VALID_DATE is None
            else pd.Timestamp(MAX_VALID_DATE)
        )
    except (TypeError, ValueError) as exc:
        print(f"Invalid configured date: {exc}", file=sys.stderr)
        return 2

    if min_valid_date > max_valid_date:
        print("MIN_VALID_DATE cannot be later than MAX_VALID_DATE", file=sys.stderr)
        return 2

    print("IMTS Field Core profiling started.")
    print(f"Input file: {input_file.resolve()}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Expected unit system ID: {expected_unit_system_id} (US)")

    try:
        df = read_input(input_file, csv_encoding)
    except Exception as exc:  # noqa: BLE001 - local script should show a useful message.
        print(f"Could not read input data: {exc}", file=sys.stderr)
        return 1

    if df.empty:
        print("Input file contains no data rows.", file=sys.stderr)
        return 1

    normalize_blank_strings(df)
    cols = resolve_columns(df)

    if cols.target_28 is None:
        print(
            "Required target column not found. Expected AverageActualStrength28_psi.",
            file=sys.stderr,
        )
        return 1

    # Coerce known numeric fields.
    numeric_columns = set(
        BASE_FIELD_NUMERIC_FEATURES
        + BASE_FIELD_INDICATOR_FEATURES
        + OPTIONAL_FIELD_FEATURES
        + TARGET_AUDIT_COLUMNS
        + [
            "ConcreteTestUnitSystem",
            "ActualStrength28SpecimenCount",
            "ActualStrength7SpecimenCount",
            "FieldCuredStrength28SpecimenCount",
            "FieldCuredStrength7SpecimenCount",
            "WaterAddedRaw",
            "waterAddedNA",
            "LoadBatchVolumeRaw",
            "LoadBatchVolumeNA",
            "AmbientTempRaw",
            "ambientTempNA",
        ]
    )
    for raw_col, flag_col, _ in LEGACY_NA_PAIRS:
        numeric_columns.add(raw_col)
        numeric_columns.add(flag_col)
    coerce_numeric_columns(df, numeric_columns)

    # Parse cast date once in the source DataFrame for profiles by year.
    if cols.cast_date:
        df[cols.cast_date] = pd.to_datetime(df[cols.cast_date], errors="coerce")

    missingness = column_missingness_profile(df)

    model_profile_columns = list(
        dict.fromkeys(
            [REQUIRED_STRENGTH_FEATURE]
            + BASE_FIELD_NUMERIC_FEATURES
            + BASE_FIELD_INDICATOR_FEATURES
            + OPTIONAL_FIELD_FEATURES
            + [MODEL_TARGET]
        )
    )
    model_profile = numeric_feature_profile(df, model_profile_columns)

    initial_counts, initial_stats = initial_curing_profile(df)
    legacy_summary, legacy_yearly, legacy_ambiguity_count = legacy_na_audit(
        df,
        cols.cast_date,
    )
    curing_summary, curing_rows = curing_profile(df, cols)
    curing_by_year, curing_by_office = curing_breakdowns(df, curing_rows, cols)
    subtype_readiness = test_subtype_readiness_profile(df, cols, curing_rows)
    suspicious_flags = build_suspicious_value_flags(df)

    base, with_required, complete, excluded = prepare_model_candidates(
        df=df,
        cols=cols,
        min_valid_date=min_valid_date,
        max_valid_date=max_valid_date,
        expected_unit_system=expected_unit_system_id,
        curing_rows=curing_rows,
        legacy_ambiguity_count=legacy_ambiguity_count,
        suspicious_flags=suspicious_flags,
    )

    overview = create_overview(
        df=df,
        cols=cols,
        curing_rows=curing_rows,
        base=base,
        with_required=with_required,
        complete=complete,
        legacy_ambiguity_count=legacy_ambiguity_count,
        suspicious_flags=suspicious_flags,
        min_valid_date=min_valid_date,
        max_valid_date=max_valid_date,
        expected_unit_system=expected_unit_system_id,
    )

    # Suspicious summary and rows.
    suspicious_summary_rows: list[dict[str, object]] = []
    flag_columns = [
        column
        for column in suspicious_flags.columns
        if column not in {"SuspiciousValueFlagCount", "HasSuspiciousValue"}
    ]
    for column in flag_columns:
        count = int(suspicious_flags[column].sum())
        suspicious_summary_rows.append(
            {
                "ReviewFlag": column,
                "RowCount": count,
                "Percent": safe_percent(count, len(df)),
            }
        )
    suspicious_summary = pd.DataFrame(suspicious_summary_rows)

    suspicious_ids = [
        column
        for column in ["testId", "OfficeName", "projectNo", "labNo", "castDate"]
        if column in df.columns
    ]
    suspicious_rows = pd.concat([df[suspicious_ids], suspicious_flags], axis=1)
    suspicious_rows = suspicious_rows.loc[suspicious_flags["HasSuspiciousValue"]].copy()

    # Duplicate rows for direct review.
    if cols.test_id:
        duplicate_mask = df[cols.test_id].duplicated(keep=False)
        duplicate_columns = [
            column
            for column in [
                cols.test_id,
                "ConcreteTestDataId",
                "OfficeName",
                "projectNo",
                "labNo",
                cols.cast_date,
                MODEL_TARGET,
            ]
            if column and column in df.columns
        ]
        duplicates = df.loc[duplicate_mask, duplicate_columns].sort_values(cols.test_id)
    else:
        duplicates = pd.DataFrame()

    # Save all outputs.
    write_dataframe(overview, output_dir / "dataset_overview.csv")
    write_dataframe(missingness, output_dir / "column_missingness.csv")
    write_dataframe(model_profile, output_dir / "model_feature_profile.csv")
    write_dataframe(curing_summary, output_dir / "curing_summary.csv")
    write_dataframe(curing_by_year, output_dir / "curing_by_year.csv")
    write_dataframe(curing_by_office, output_dir / "curing_by_office.csv")
    write_dataframe(subtype_readiness, output_dir / "test_subtype_readiness.csv")
    write_dataframe(initial_counts, output_dir / "initial_curing_text_frequency.csv")
    write_dataframe(legacy_summary, output_dir / "legacy_na_zero_audit.csv")
    write_dataframe(legacy_yearly, output_dir / "legacy_na_flag_usage_by_year.csv")
    write_dataframe(suspicious_summary, output_dir / "suspicious_value_summary.csv")
    write_dataframe(suspicious_rows, output_dir / "suspicious_rows.csv")
    write_dataframe(duplicates, output_dir / "duplicate_test_rows.csv")
    write_dataframe(base, output_dir / "field_core_model_candidates.csv")
    write_dataframe(
        with_required,
        output_dir / "field_core_model_candidates_with_required_strength.csv",
    )
    write_dataframe(complete, output_dir / "field_core_complete_cases.csv")
    write_dataframe(excluded, output_dir / "excluded_rows.csv")

    categorical_profiles(df, output_dir)
    save_plots(df, output_dir, cols, missingness, curing_summary)
    write_markdown_report(
        output_dir=output_dir,
        input_file=input_file,
        overview=overview,
        curing_summary=curing_summary,
        model_profile=model_profile,
        initial_stats=initial_stats,
        legacy_summary=legacy_summary,
    )

    run_metadata = {
        "input_file": str(input_file.resolve()),
        "output_dir": str(output_dir.resolve()),
        "generated_at": pd.Timestamp.now().isoformat(),
        "expected_unit_system_id": expected_unit_system_id,
        "expected_unit_system_label": "US",
        "min_valid_date": min_valid_date.date().isoformat(),
        "max_valid_date": max_valid_date.date().isoformat(),
        "total_records": len(df),
        "base_model_candidates": len(base),
        "with_required_strength_candidates": len(with_required),
        "complete_case_candidates": len(complete),
        "resolved_columns": cols.__dict__,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print("IMTS Field Core profiling completed.")
    print(f"Input records: {len(df):,}")
    if cols.test_id:
        print(f"Unique tests: {df[cols.test_id].nunique(dropna=True):,}")
    print(f"Base model candidates: {len(base):,}")
    print(f"Candidates with required strength: {len(with_required):,}")
    print(f"Complete-case candidates: {len(complete):,}")
    print(f"Output directory: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
