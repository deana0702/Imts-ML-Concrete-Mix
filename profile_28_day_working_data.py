from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


INPUT_FILE = Path(
    "data/prepared_28_day_standard_cure/"
    "03_standard_cured_test_level_28_working_data.csv"
)

OUTPUT_DIRECTORY = Path(
    "data/prepared_28_day_standard_cure/profile"
)

TARGET_COLUMN = "AverageActualStrength28_psi"
ID_COLUMN = "testId"


def normalize_category_value(value: object) -> object:
    """
    Normalize a category only for comparison.

    The original value is not overwritten.

    Examples:
        " Duke City " -> "DUKE CITY"
        "duke   city" -> "DUKE CITY"
    """
    if pd.isna(value):
        return pd.NA

    text = str(value).strip()

    if not text:
        return pd.NA

    text = re.sub(r"\s+", " ", text)
    return text.upper()


def clean_blank_strings_for_profile(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Treat blank strings and SQL-style NULL text as missing
    for profiling purposes.
    """
    cleaned = dataframe.copy()

    text_columns = cleaned.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in text_columns:
        cleaned[column] = (
            cleaned[column]
            .astype("string")
            .str.strip()
            .replace(
                {
                    "": pd.NA,
                    "NULL": pd.NA,
                    "null": pd.NA,
                    "Null": pd.NA,
                }
            )
        )

    return cleaned


def create_field_profile(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    row_count = len(dataframe)
    records: list[dict[str, object]] = []

    for column in dataframe.columns:
        series = dataframe[column]

        non_null_count = int(series.notna().sum())
        null_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))

        record: dict[str, object] = {
            "Column": column,
            "PandasDtype": str(series.dtype),
            "RowCount": row_count,
            "NonNullCount": non_null_count,
            "NullCount": null_count,
            "NullPercent": (
                null_count / row_count * 100
                if row_count
                else np.nan
            ),
            "UniqueCount": unique_count,
            "UniquePercentOfNonNull": (
                unique_count / non_null_count * 100
                if non_null_count
                else np.nan
            ),
            "IsAllNull": non_null_count == 0,
            "IsConstant": unique_count <= 1,
            "IsLikelyIdentifier": (
                non_null_count > 0
                and unique_count == non_null_count
            ),
        }

        if (is_numeric_dtype(series) and not is_bool_dtype(series)):
            numeric_series = pd.to_numeric(
                series,
                errors="coerce",
            )

            record.update(
                {
                    "ZeroCount": int(
                        numeric_series.eq(0).sum()
                    ),
                    "NegativeCount": int(
                        numeric_series.lt(0).sum()
                    ),
                    "Minimum": numeric_series.min(),
                    "Percentile01": numeric_series.quantile(
                        0.01
                    ),
                    "Percentile05": numeric_series.quantile(
                        0.05
                    ),
                    "Mean": numeric_series.mean(),
                    "Median": numeric_series.median(),
                    "Percentile95": numeric_series.quantile(
                        0.95
                    ),
                    "Percentile99": numeric_series.quantile(
                        0.99
                    ),
                    "Maximum": numeric_series.max(),
                    "StandardDeviation": numeric_series.std(),
                    "NormalizedUniqueCount": np.nan,
                    "PotentialFormattingDuplicateCount": (
                        np.nan
                    ),
                }
            )

        else:
            normalized = series.map(
                normalize_category_value
            )

            normalized_unique_count = int(
                normalized.nunique(dropna=True)
            )

            record.update(
                {
                    "ZeroCount": np.nan,
                    "NegativeCount": np.nan,
                    "Minimum": np.nan,
                    "Percentile01": np.nan,
                    "Percentile05": np.nan,
                    "Mean": np.nan,
                    "Median": np.nan,
                    "Percentile95": np.nan,
                    "Percentile99": np.nan,
                    "Maximum": np.nan,
                    "StandardDeviation": np.nan,
                    "NormalizedUniqueCount": (
                        normalized_unique_count
                    ),
                    "PotentialFormattingDuplicateCount": (
                        unique_count
                        - normalized_unique_count
                    ),
                }
            )

        records.append(record)

    profile = pd.DataFrame(records)

    profile["NullPercent"] = profile[
        "NullPercent"
    ].round(2)

    profile["UniquePercentOfNonNull"] = profile[
        "UniquePercentOfNonNull"
    ].round(2)

    return profile.sort_values(
        [
            "NullPercent",
            "Column",
        ],
        ascending=[
            False,
            True,
        ],
    )


def create_numeric_profile(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    records: list[dict[str, object]] = []

    numeric_columns = [
        column
        for column in dataframe.columns
        if (is_numeric_dtype(dataframe[column])
            and not is_bool_dtype(dataframe[column]))
    ]

    for column in numeric_columns:
        series = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        records.append(
            {
                "Column": column,
                "Count": int(series.notna().sum()),
                "NullCount": int(series.isna().sum()),
                "ZeroCount": int(series.eq(0).sum()),
                "NegativeCount": int(series.lt(0).sum()),
                "Minimum": series.min(),
                "P01": series.quantile(0.01),
                "P05": series.quantile(0.05),
                "P25": series.quantile(0.25),
                "Median": series.median(),
                "Mean": series.mean(),
                "P75": series.quantile(0.75),
                "P95": series.quantile(0.95),
                "P99": series.quantile(0.99),
                "Maximum": series.max(),
                "StandardDeviation": series.std(),
            }
        )

    return pd.DataFrame(records)


def create_categorical_top_values(
    dataframe: pd.DataFrame,
    top_n: int = 30,
) -> pd.DataFrame:

    records: list[dict[str, object]] = []

    categorical_columns = dataframe.select_dtypes(
        include=["object", "string", "bool"]
    ).columns

    for column in categorical_columns:
        series = dataframe[column]

        value_counts = series.value_counts(
            dropna=False
        ).head(top_n)

        for rank, (value, count) in enumerate(
            value_counts.items(),
            start=1,
        ):
            normalized_value = (
                normalize_category_value(value)
            )

            records.append(
                {
                    "Column": column,
                    "Rank": rank,
                    "OriginalValue": value,
                    "NormalizedComparisonValue": (
                        normalized_value
                    ),
                    "Count": int(count),
                    "Percent": round(
                        count / len(dataframe) * 100,
                        2,
                    ),
                }
            )

    return pd.DataFrame(records)


def create_formatting_variant_report(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Detect values that become identical after trimming spaces,
    collapsing repeated spaces, and converting to uppercase.

    Example:
        "Broadway 1"
        "broadway 1"
        " Broadway   1 "
    """
    records: list[dict[str, object]] = []

    categorical_columns = dataframe.select_dtypes(
        include=["object", "string", "bool"]
    ).columns

    for column in categorical_columns:
        working = pd.DataFrame(
            {
                "OriginalValue": dataframe[column],
            }
        ).dropna()

        if working.empty:
            continue

        working["NormalizedValue"] = (
            working["OriginalValue"]
            .map(normalize_category_value)
        )

        variant_summary = (
            working
            .groupby(
                "NormalizedValue",
                dropna=False,
            )
            .agg(
                OriginalVariantCount=(
                    "OriginalValue",
                    "nunique",
                ),
                TotalRowCount=(
                    "OriginalValue",
                    "size",
                ),
                OriginalVariants=(
                    "OriginalValue",
                    lambda values: " | ".join(
                        sorted(
                            {
                                str(value)
                                for value in values
                            }
                        )
                    ),
                ),
            )
            .reset_index()
        )

        variant_summary = variant_summary.loc[
            variant_summary[
                "OriginalVariantCount"
            ].gt(1)
        ]

        for row in variant_summary.itertuples(
            index=False
        ):
            records.append(
                {
                    "Column": column,
                    "NormalizedValue": (
                        row.NormalizedValue
                    ),
                    "OriginalVariantCount": (
                        row.OriginalVariantCount
                    ),
                    "TotalRowCount": (
                        row.TotalRowCount
                    ),
                    "OriginalVariants": (
                        row.OriginalVariants
                    ),
                }
            )

    return pd.DataFrame(records)


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

    dataframe = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig",
        na_values=[
            "NULL",
            "null",
            "Null",
            "",
        ],
        keep_default_na=True,
        low_memory=False,
    )

    dataframe = clean_blank_strings_for_profile(
        dataframe
    )

    if ID_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Missing required ID column: {ID_COLUMN}"
        )

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Missing Target column: {TARGET_COLUMN}"
        )

    if dataframe[ID_COLUMN].duplicated().any():
        raise ValueError(
            "testId is not unique in the working dataset."
        )

    if dataframe[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Target contains missing values."
        )

    print(f"Rows: {len(dataframe):,}")
    print(f"Columns: {len(dataframe.columns):,}")
    print("testId uniqueness check: passed")
    print("Target missing-value check: passed")

    field_profile = create_field_profile(
        dataframe
    )

    numeric_profile = create_numeric_profile(
        dataframe
    )

    categorical_top_values = (
        create_categorical_top_values(
            dataframe,
            top_n=30,
        )
    )

    formatting_variants = (
        create_formatting_variant_report(
            dataframe
        )
    )

    field_profile.to_csv(
        OUTPUT_DIRECTORY / "field_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )

    numeric_profile.to_csv(
        OUTPUT_DIRECTORY / "numeric_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )

    categorical_top_values.to_csv(
        OUTPUT_DIRECTORY
        / "categorical_top_values.csv",
        index=False,
        encoding="utf-8-sig",
    )

    formatting_variants.to_csv(
        OUTPUT_DIRECTORY
        / "categorical_formatting_variants.csv",
        index=False,
        encoding="utf-8-sig",
    )

    constant_columns = field_profile.loc[
        field_profile["IsConstant"]
    ]

    high_missing_columns = field_profile.loc[
        field_profile["NullPercent"].ge(50)
    ]

    potential_formatting_duplicates = (
        field_profile.loc[
            field_profile[
                "PotentialFormattingDuplicateCount"
            ]
            .fillna(0)
            .gt(0)
        ]
    )

    print()
    print("Profiling completed.")
    print(
        f"Constant/all-null columns: "
        f"{len(constant_columns):,}"
    )
    print(
        f"Columns with at least 50% missing: "
        f"{len(high_missing_columns):,}"
    )
    print(
        "Columns with possible formatting duplicates: "
        f"{len(potential_formatting_duplicates):,}"
    )
    print(
        f"Output folder: "
        f"{OUTPUT_DIRECTORY.resolve()}"
    )


if __name__ == "__main__":
    main()