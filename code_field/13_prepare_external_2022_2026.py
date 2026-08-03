from __future__ import annotations

import json

import numpy as np
import pandas as pd

from field_core_experiment_common import REQUIRED_STRENGTH, TARGET
from field_core_validation_common import (
    EXTERNAL_DIR,
    EXTERNAL_PREPARED_FILE,
    EXTERNAL_RAW_FILE,
    read_csv,
    resolve_cast_date_column,
    resolve_unit_system_column,
)


# Run with:
#     python code_field/13_prepare_external_2022_2026.py
#
# Before running, export the current production database with the same final SQL
# query and save the first result set here:
#     data/external_2022_2026/field_core_US_data_current.csv
START_YEAR = 2022
END_YEAR = 2026
EXPECTED_UNIT_SYSTEM_ID = 0
MAX_STRENGTH_PSI = 20_000.0

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

DAY7_CANDIDATES = ["AverageActualStrength7_psi", "AverageActualStrength7"]


def numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(
        series.astype("string").str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def append_reason(reasons: pd.Series, mask: pd.Series, reason: str) -> pd.Series:
    current = reasons.fillna("")
    updated = np.where(
        mask.fillna(False),
        np.where(current.eq(""), reason, current + ";" + reason),
        current,
    )
    return pd.Series(updated, index=reasons.index, dtype="string").replace("", pd.NA)


def main() -> int:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_PREPARED_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("External preparation 13 started: 2022-2026 current production data")
    print(f"Input: {EXTERNAL_RAW_FILE}")
    print(f"Output: {EXTERNAL_PREPARED_FILE}")

    df = read_csv(EXTERNAL_RAW_FILE)
    if "testId" not in df.columns:
        raise KeyError("External data must contain testId.")
    if TARGET not in df.columns or REQUIRED_STRENGTH not in df.columns:
        raise KeyError(
            f"External data must contain {TARGET} and {REQUIRED_STRENGTH}."
        )

    object_columns = df.select_dtypes(include=["object", "string"]).columns
    for column in object_columns:
        df[column] = df[column].replace(r"^\s*$", pd.NA, regex=True)

    cast_date_column = resolve_cast_date_column(df)
    unit_column = resolve_unit_system_column(df)
    df[cast_date_column] = pd.to_datetime(df[cast_date_column], errors="coerce")
    df[TARGET] = numeric(df[TARGET])
    df[REQUIRED_STRENGTH] = numeric(df[REQUIRED_STRENGTH])
    df[unit_column] = numeric(df[unit_column])

    reasons = pd.Series(pd.NA, index=df.index, dtype="string")
    valid_year = df[cast_date_column].dt.year.between(START_YEAR, END_YEAR)
    reasons = append_reason(reasons, ~valid_year, "OutsideExternalYearRange")
    reasons = append_reason(
        reasons,
        df[unit_column].ne(EXPECTED_UNIT_SYSTEM_ID),
        "NonUSUnitSystem",
    )
    reasons = append_reason(
        reasons,
        df[TARGET].isna() | df[TARGET].le(0) | df[TARGET].gt(MAX_STRENGTH_PSI),
        "InvalidActualStrength28",
    )
    reasons = append_reason(
        reasons,
        df[REQUIRED_STRENGTH].isna()
        | df[REQUIRED_STRENGTH].le(0)
        | df[REQUIRED_STRENGTH].gt(MAX_STRENGTH_PSI),
        "InvalidRequiredStrength28",
    )

    duplicate_mask = df["testId"].duplicated(keep=False)
    reasons = append_reason(reasons, duplicate_mask, "DuplicateTestId")

    adjustment_rows: list[dict[str, object]] = []
    for column, (minimum, maximum) in FEATURE_RANGES.items():
        if column not in df.columns:
            continue
        df[column] = numeric(df[column])
        invalid = df[column].notna() & ~df[column].between(minimum, maximum)
        adjustment_rows.append(
            {
                "Column": column,
                "ChangedToMissingCount": int(invalid.sum()),
                "AllowedMinimum": minimum,
                "AllowedMaximum": maximum,
            }
        )
        df.loc[invalid, column] = np.nan

    for column in DAY7_CANDIDATES:
        if column in df.columns:
            df[column] = numeric(df[column])
            invalid = df[column].notna() & (
                df[column].le(0) | df[column].gt(MAX_STRENGTH_PSI)
            )
            adjustment_rows.append(
                {
                    "Column": column,
                    "ChangedToMissingCount": int(invalid.sum()),
                    "AllowedMinimum": 0.0,
                    "AllowedMaximum": MAX_STRENGTH_PSI,
                }
            )
            df.loc[invalid, column] = np.nan

    excluded = df.loc[reasons.notna()].copy()
    excluded["ExternalExclusionReasons"] = reasons.loc[reasons.notna()]
    clean = df.loc[reasons.isna()].copy()
    clean["ExternalValidationYear"] = clean[cast_date_column].dt.year.astype(int)
    clean = clean.sort_values([cast_date_column, "testId"]).reset_index(drop=True)

    clean.to_csv(EXTERNAL_PREPARED_FILE, index=False)
    excluded.to_csv(
        EXTERNAL_PREPARED_FILE.parent / "external_excluded_rows.csv", index=False
    )
    pd.DataFrame(adjustment_rows).to_csv(
        EXTERNAL_PREPARED_FILE.parent / "external_feature_adjustments.csv", index=False
    )

    year_summary = (
        clean.groupby("ExternalValidationYear")
        .agg(
            RowCount=("testId", "size"),
            UniqueTests=("testId", "nunique"),
            ActualStrength28Mean=(TARGET, "mean"),
            RequiredStrength28Mean=(REQUIRED_STRENGTH, "mean"),
        )
        .reset_index()
    )
    year_summary.to_csv(
        EXTERNAL_PREPARED_FILE.parent / "external_year_summary.csv", index=False
    )

    metadata = {
        "input_file": str(EXTERNAL_RAW_FILE),
        "output_file": str(EXTERNAL_PREPARED_FILE),
        "external_start_year": START_YEAR,
        "external_end_year": END_YEAR,
        "expected_unit_system_id": EXPECTED_UNIT_SYSTEM_ID,
        "input_rows": int(len(df)),
        "clean_external_rows": int(len(clean)),
        "excluded_rows": int(len(excluded)),
        "unique_clean_tests": int(clean["testId"].nunique()),
        "earliest_clean_date": str(clean[cast_date_column].min()),
        "latest_clean_date": str(clean[cast_date_column].max()),
    }
    (EXTERNAL_PREPARED_FILE.parent / "external_preparation_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("External preparation 13 completed.")
    print(f"Input rows: {len(df):,}")
    print(f"Clean external rows: {len(clean):,}")
    print(f"Excluded rows: {len(excluded):,}")
    print(year_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
