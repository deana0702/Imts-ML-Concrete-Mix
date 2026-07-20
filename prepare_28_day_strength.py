from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path("data/final_data.csv")
OUTPUT_DIRECTORY = Path("data/prepared_28_day_standard_cure")

TARGET_NOMINAL_AGE = 28

# Assumption confirmed for this preparation:
# 0 = standard cured / not field cured
# 1 = field cured
STANDARD_CURE_VALUE = 0


# ============================================================
# Columns that should remain strings
#
# This prevents values such as plantNumber "002"
# from being changed to numeric value 2.
# ============================================================

STRING_COLUMNS = {
    "OfficeName": "string",
    "projectNo": "string",
    "labNo": "string",
    "SupplierName": "string",
    "adMixture": "string",
    "plantNumber": "string",
    "mixNumber": "string",
    "WaterUnits": "string",
    "ApplicableStrengthType": "string",
    "initialCuringCondition": "string",
    "FinalCure": "string",
    "CloudType": "string",
    "PrecipitationType": "string",
    "WindType": "string",
}


# ============================================================
# Required columns from the SQL extraction
# ============================================================

REQUIRED_COLUMNS = {
    "testId",
    "SpecimenRowId",
    "daysToAge",
    "wasFieldCured",
    "castDate",
    "testedOnDate",
    "SpecifiedBreakAge",
    "RequiredStrength",
    "DesignStrength",
    "ApplicableSpecifiedStrength",
    "ApplicableStrengthType",
    "RequiredStrengthRowCount",
    "DesignStrengthRowCount",
    "calcCompressiveStrength",
}


def detect_delimiter(file_path: Path) -> str:
    """
    Detect whether the file uses comma, tab, semicolon, or pipe.
    """
    with file_path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    ) as file:
        sample = file.read(64_000)

    try:
        dialect = csv.Sniffer().sniff(
            sample,
            delimiters=",\t;|",
        )
        return dialect.delimiter
    except csv.Error:
        return ","


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove BOM, leading/trailing spaces, and line breaks
    from exported SQL column names.
    """
    cleaned = df.copy()

    cleaned.columns = (
        cleaned.columns
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("\r", " ", regex=False)
        .str.replace("\n", " ", regex=False)
        .str.strip()
    )

    return cleaned


def validate_required_columns(df: pd.DataFrame) -> None:
    """
    Stop processing if an expected SQL column is missing.
    """
    missing_columns = sorted(
        REQUIRED_COLUMNS.difference(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "The input file is missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
            + "\n\nAvailable columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in df.columns
            )
        )


def convert_numeric_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> None:
    """
    Convert specified columns to numeric.
    Invalid text is converted to NaN for later review.
    """
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )


def first_non_null(series: pd.Series):
    """
    Return the first non-null value in a grouped column.
    """
    values = series.dropna()

    if values.empty:
        return pd.NA

    return values.iloc[0]


def create_strength_target_28(
    standard_cured_28: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create one 28-day actual-strength Target row per testId.

    Only rows satisfying:
        daysToAge = 28
        wasFieldCured = 0
        calcCompressiveStrength > 0

    are included.
    """
    strength_target = (
        standard_cured_28
        .groupby(
            "testId",
            dropna=False,
        )
        .agg(
            StandardCuredSpecimenCount28=(
                "SpecimenRowId",
                "size",
            ),
            UniqueStandardCuredSpecimenCount28=(
                "SpecimenRowId",
                "nunique",
            ),
            AverageActualStrength28_psi=(
                "calcCompressiveStrength",
                "mean",
            ),
            MinimumActualStrength28_psi=(
                "calcCompressiveStrength",
                "min",
            ),
            MaximumActualStrength28_psi=(
                "calcCompressiveStrength",
                "max",
            ),
            ActualStrengthStdDev28_psi=(
                "calcCompressiveStrength",
                "std",
            ),
            MinimumActualAgeDays=(
                "ActualAgeDays",
                "min",
            ),
            MaximumActualAgeDays=(
                "ActualAgeDays",
                "max",
            ),
            AverageActualAgeDays=(
                "ActualAgeDays",
                "mean",
            ),
        )
        .reset_index()
    )

    strength_target["ActualStrengthRange28_psi"] = (
        strength_target["MaximumActualStrength28_psi"]
        - strength_target["MinimumActualStrength28_psi"]
    )

    strength_target["ActualStrengthCV28_percent"] = np.where(
        strength_target[
            "AverageActualStrength28_psi"
        ].gt(0),
        (
            strength_target[
                "ActualStrengthStdDev28_psi"
            ]
            / strength_target[
                "AverageActualStrength28_psi"
            ]
            * 100
        ),
        np.nan,
    )

    return strength_target


def create_specification_summary(
    standard_cured_28: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create one specification summary per testId.

    These values come from the 28-day Strength specification
    joined by the SQL query.
    """
    specification_summary = (
        standard_cured_28
        .groupby(
            "testId",
            dropna=False,
        )
        .agg(
            SpecifiedBreakAge=(
                "SpecifiedBreakAge",
                first_non_null,
            ),
            RequiredStrength=(
                "RequiredStrength",
                first_non_null,
            ),
            DesignStrength=(
                "DesignStrength",
                first_non_null,
            ),
            ApplicableSpecifiedStrength=(
                "ApplicableSpecifiedStrength",
                first_non_null,
            ),
            ApplicableStrengthType=(
                "ApplicableStrengthType",
                first_non_null,
            ),
            RequiredStrengthRowCount=(
                "RequiredStrengthRowCount",
                "max",
            ),
            DesignStrengthRowCount=(
                "DesignStrengthRowCount",
                "max",
            ),
        )
        .reset_index()
    )

    specification_summary["SpecifiedStrengthMissing"] = (
        specification_summary[
            "ApplicableSpecifiedStrength"
        ].isna()
    )

    specification_summary["SpecifiedBreakAgeIs28"] = (
        specification_summary[
            "SpecifiedBreakAge"
        ].eq(TARGET_NOMINAL_AGE)
    )

    specification_summary["MultipleStrengthRowsFlag"] = (
        specification_summary[
            "RequiredStrengthRowCount"
        ].fillna(0).gt(1)
        |
        specification_summary[
            "DesignStrengthRowCount"
        ].fillna(0).gt(1)
    )

    return specification_summary


def create_test_level_working_data(
    standard_cured_28: pd.DataFrame,
    strength_target_28: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create one row per testId.

    Specification columns are excluded from the left-side
    test-level data because strength_target_28 already contains
    the summarized specification values.
    """

    # Specimen마다 달라지는 컬럼
    specimen_specific_columns = {
        "SpecimenRowId",
        "daysToAge",
        "wasFieldCured",
        "testedOnDate",
        "ActualBreakDate",
        "ScheduledBreakDate",
        "ActualAgeDays",
        "BreakDateDifferenceDays",
        "TestedExactlyOnSchedule",
        "TestedWithinOneDayOfSchedule",
        "ActualCompressiveStrength_psi",
        "widthDiameter",
        "heightLength",
        "calcArea",
        "FractureType",
        "CapType",
        "testLoad",
        "calcCompressiveStrength",
    }

    # strength_target_28에 이미 요약되어 있는 specification 컬럼
    specification_columns = {
        "SpecifiedBreakAge",
        "RequiredStrength",
        "DesignStrength",
        "ApplicableSpecifiedStrength",
        "ApplicableStrengthType",
        "RequiredStrengthRowCount",
        "DesignStrengthRowCount",
    }

    excluded_columns = (
        specimen_specific_columns
        | specification_columns
    )

    # testId별로 한 번만 남길 test-level feature
    test_level_columns = [
        column
        for column in standard_cured_28.columns
        if column not in excluded_columns
    ]

    test_level_features = (
        standard_cured_28
        .sort_values(
            ["testId", "SpecimenRowId"],
            kind="stable",
        )[test_level_columns]
        .drop_duplicates(
            subset=["testId"],
            keep="first",
        )
    )

    working_data = test_level_features.merge(
        strength_target_28,
        on="testId",
        how="inner",
        validate="one_to_one",
    )

    # 안전 검사: merge 후 _x, _y 컬럼이 없어야 함
    unexpected_suffix_columns = [
        column
        for column in working_data.columns
        if column.endswith("_x")
        or column.endswith("_y")
    ]

    if unexpected_suffix_columns:
        raise ValueError(
            "Unexpected duplicate columns after merge:\n"
            + "\n".join(unexpected_suffix_columns)
        )

    return working_data


def save_csv(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    dataframe.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL,
    )


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file was not found:\n"
            f"{INPUT_FILE.resolve()}"
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    delimiter = detect_delimiter(INPUT_FILE)

    delimiter_display = (
        "TAB"
        if delimiter == "\t"
        else repr(delimiter)
    )

    print(f"Detected delimiter: {delimiter_display}")

    # ========================================================
    # 1. Read the SQL-exported file
    # ========================================================

    final_data = pd.read_csv(
        INPUT_FILE,
        sep=delimiter,
        quotechar='"',
        encoding="utf-8-sig",
        dtype=STRING_COLUMNS,
        na_values=[
            "NULL",
            "null",
            "Null",
            "",
        ],
        keep_default_na=True,
        low_memory=False,
        on_bad_lines="error",
    )

    final_data = normalize_column_names(final_data)
    validate_required_columns(final_data)

    print(f"Input rows: {len(final_data):,}")
    print(f"Input columns: {len(final_data.columns):,}")

    # ========================================================
    # 2. Convert required numeric columns
    # ========================================================

    numeric_columns = [
        "testId",
        "SpecimenRowId",
        "daysToAge",
        "wasFieldCured",
        "SpecifiedBreakAge",
        "RequiredStrength",
        "DesignStrength",
        "ApplicableSpecifiedStrength",
        "RequiredStrengthRowCount",
        "DesignStrengthRowCount",
        "calcCompressiveStrength",
    ]

    convert_numeric_columns(
        final_data,
        numeric_columns,
    )

    # ========================================================
    # 3. Parse dates and calculate actual break age
    # ========================================================

    final_data["castDate"] = pd.to_datetime(
        final_data["castDate"],
        errors="coerce",
    )

    final_data["testedOnDate"] = pd.to_datetime(
        final_data["testedOnDate"],
        errors="coerce",
    )

    final_data["ScheduledBreakDate"] = (
        final_data["castDate"].dt.normalize()
        + pd.to_timedelta(
            final_data["daysToAge"],
            unit="D",
        )
    )

    final_data["ActualBreakDate"] = (
        final_data["testedOnDate"].dt.normalize()
    )

    final_data["ActualAgeDays"] = (
        final_data["ActualBreakDate"]
        - final_data["castDate"].dt.normalize()
    ).dt.days

    final_data["BreakDateDifferenceDays"] = (
        final_data["ActualBreakDate"]
        - final_data["ScheduledBreakDate"]
    ).dt.days

    final_data["TestedExactlyOnSchedule"] = (
        final_data["BreakDateDifferenceDays"].eq(0)
    )

    final_data["TestedWithinOneDayOfSchedule"] = (
        final_data["BreakDateDifferenceDays"]
        .between(
            -1,
            1,
            inclusive="both",
        )
    )

    # ========================================================
    # 4. Validate SpecimenRowId uniqueness
    # ========================================================

    null_specimen_ids = final_data.loc[
        final_data["SpecimenRowId"].isna()
    ].copy()

    duplicate_specimen_rows = final_data.loc[
        final_data["SpecimenRowId"]
        .duplicated(keep=False)
    ].copy()

    save_csv(
        null_specimen_ids,
        OUTPUT_DIRECTORY / "qa_null_specimen_ids.csv",
    )

    save_csv(
        duplicate_specimen_rows,
        OUTPUT_DIRECTORY
        / "qa_duplicate_specimen_rows.csv",
    )

    if not duplicate_specimen_rows.empty:
        raise ValueError(
            "Duplicate SpecimenRowId values were found. "
            "Review qa_duplicate_specimen_rows.csv."
        )

    print("SpecimenRowId duplicate check: passed")

    # ========================================================
    # 5. Filter nominal 28-day specimens
    # ========================================================

    specimens_nominal_28 = final_data.loc[
        final_data["daysToAge"].eq(
            TARGET_NOMINAL_AGE
        )
    ].copy()

    print(
        "Nominal 28-day specimens: "
        f"{len(specimens_nominal_28):,}"
    )

    # ========================================================
    # 6. Separate curing types before averaging
    # ========================================================

    standard_cured_28_all = specimens_nominal_28.loc[
        specimens_nominal_28[
            "wasFieldCured"
        ].eq(STANDARD_CURE_VALUE)
    ].copy()

    field_cured_28 = specimens_nominal_28.loc[
        specimens_nominal_28[
            "wasFieldCured"
        ].eq(1)
    ].copy()

    unknown_cure_28 = specimens_nominal_28.loc[
        specimens_nominal_28[
            "wasFieldCured"
        ].isna()
    ].copy()

    other_cure_values_28 = specimens_nominal_28.loc[
        specimens_nominal_28[
            "wasFieldCured"
        ].notna()
        &
        ~specimens_nominal_28[
            "wasFieldCured"
        ].isin([0, 1])
    ].copy()

    print(
        "Standard-cured 28-day specimens "
        "(wasFieldCured = 0): "
        f"{len(standard_cured_28_all):,}"
    )

    print(
        "Excluded field-cured specimens "
        "(wasFieldCured = 1): "
        f"{len(field_cured_28):,}"
    )

    print(
        "Excluded unknown-cure specimens: "
        f"{len(unknown_cure_28):,}"
    )

    # Save excluded curing groups for later review.
    save_csv(
        field_cured_28,
        OUTPUT_DIRECTORY
        / "qa_excluded_field_cured_28.csv",
    )

    save_csv(
        unknown_cure_28,
        OUTPUT_DIRECTORY
        / "qa_excluded_unknown_cure_28.csv",
    )

    save_csv(
        other_cure_values_28,
        OUTPUT_DIRECTORY
        / "qa_unexpected_wasFieldCured_values.csv",
    )

    # ========================================================
    # 7. Filter valid actual-strength results
    # ========================================================

    valid_strength_mask = (
        standard_cured_28_all[
            "calcCompressiveStrength"
        ].notna()
        &
        standard_cured_28_all[
            "calcCompressiveStrength"
        ].gt(0)
    )

    standard_cured_28 = standard_cured_28_all.loc[
        valid_strength_mask
    ].copy()

    invalid_standard_strength_rows = (
        standard_cured_28_all.loc[
            ~valid_strength_mask
        ].copy()
    )

    standard_cured_28[
        "ActualCompressiveStrength_psi"
    ] = standard_cured_28[
        "calcCompressiveStrength"
    ]

    save_csv(
        invalid_standard_strength_rows,
        OUTPUT_DIRECTORY
        / "qa_invalid_standard_cured_28_strength.csv",
    )

    print(
        "Valid standard-cured 28-day specimens: "
        f"{len(standard_cured_28):,}"
    )

    # ========================================================
    # 8. Break-date quality issues
    #
    # Do not remove these rows yet.
    # Save them for review.
    # ========================================================

    break_date_quality_issues = standard_cured_28.loc[
        standard_cured_28[
            "ScheduledBreakDate"
        ].isna()
        |
        standard_cured_28[
            "ActualBreakDate"
        ].isna()
        |
        ~standard_cured_28[
            "TestedWithinOneDayOfSchedule"
        ]
    ].copy()

    save_csv(
        break_date_quality_issues,
        OUTPUT_DIRECTORY
        / "qa_standard_cured_28_break_date_issues.csv",
    )

    # ========================================================
    # 9. Strength-specification quality checks
    # ========================================================

    specified_age_issues = standard_cured_28.loc[
        standard_cured_28[
            "SpecifiedBreakAge"
        ].notna()
        &
        ~standard_cured_28[
            "SpecifiedBreakAge"
        ].eq(TARGET_NOMINAL_AGE)
    ].copy()

    missing_specification_rows = standard_cured_28.loc[
        standard_cured_28[
            "ApplicableSpecifiedStrength"
        ].isna()
    ].copy()

    save_csv(
        specified_age_issues,
        OUTPUT_DIRECTORY
        / "qa_standard_cured_28_specified_age_issues.csv",
    )

    save_csv(
        missing_specification_rows,
        OUTPUT_DIRECTORY
        / "qa_standard_cured_28_missing_specification.csv",
    )

    # ========================================================
    # 10. Check whether the same testId contains mixed
    #     curing types in the original nominal 28-day data
    # ========================================================

    cure_value_count_by_test = (
        specimens_nominal_28
        .groupby("testId")["wasFieldCured"]
        .nunique(dropna=True)
    )

    mixed_curing_test_ids = (
        cure_value_count_by_test.loc[
            cure_value_count_by_test > 1
        ]
        .index
    )

    mixed_curing_rows = specimens_nominal_28.loc[
        specimens_nominal_28[
            "testId"
        ].isin(mixed_curing_test_ids)
    ].sort_values(
        [
            "testId",
            "wasFieldCured",
            "SpecimenRowId",
        ]
    )

    save_csv(
        mixed_curing_rows,
        OUTPUT_DIRECTORY
        / "qa_tests_with_mixed_curing_types.csv",
    )

    print(
        "testId values containing both curing types: "
        f"{len(mixed_curing_test_ids):,}"
    )

    # ========================================================
    # 11. Create averaged Target by testId
    # ========================================================

    strength_target_28 = create_strength_target_28(
        standard_cured_28
    )

    specification_summary = create_specification_summary(
        standard_cured_28
    )

    strength_target_28 = strength_target_28.merge(
        specification_summary,
        on="testId",
        how="left",
        validate="one_to_one",
    )

    # ========================================================
    # 12. Create one-row-per-test working dataset
    # ========================================================

    test_level_working_data = (
        create_test_level_working_data(
            standard_cured_28,
            strength_target_28,
        )
    )

    # ========================================================
    # 13. Round reporting values
    # ========================================================

    columns_to_round = [
        "AverageActualStrength28_psi",
        "MinimumActualStrength28_psi",
        "MaximumActualStrength28_psi",
        "ActualStrengthStdDev28_psi",
        "ActualStrengthRange28_psi",
        "ActualStrengthCV28_percent",
        "MinimumActualAgeDays",
        "MaximumActualAgeDays",
        "AverageActualAgeDays",
    ]

    for dataframe in [
        strength_target_28,
        test_level_working_data,
    ]:
        existing_columns = [
            column
            for column in columns_to_round
            if column in dataframe.columns
        ]

        dataframe[existing_columns] = (
            dataframe[existing_columns].round(2)
        )

    # ========================================================
    # 14. Save primary output files
    # ========================================================

    save_csv(
        standard_cured_28,
        OUTPUT_DIRECTORY
        / "01_standard_cured_28_specimens.csv",
    )

    save_csv(
        strength_target_28,
        OUTPUT_DIRECTORY
        / "02_standard_cured_strength_target_28_by_test.csv",
    )

    save_csv(
        test_level_working_data,
        OUTPUT_DIRECTORY
        / "03_standard_cured_test_level_28_working_data.csv",
    )

    # ========================================================
    # 15. Summary
    # ========================================================

    print()
    print("Preparation completed.")
    print(
        "Valid standard-cured 28-day specimen rows: "
        f"{len(standard_cured_28):,}"
    )
    print(
        "Unique testId Target rows: "
        f"{strength_target_28['testId'].nunique():,}"
    )
    print(
        "One-row-per-test working rows: "
        f"{len(test_level_working_data):,}"
    )
    print(
        "Break-date quality issue rows: "
        f"{len(break_date_quality_issues):,}"
    )
    print(
        "Missing applicable specification rows: "
        f"{len(missing_specification_rows):,}"
    )
    print(
        "Output folder: "
        f"{OUTPUT_DIRECTORY.resolve()}"
    )


if __name__ == "__main__":
    main()