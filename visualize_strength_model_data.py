from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_FILE = Path("concrete_28_day_working_data.csv")
OUTPUT_DIRECTORY = Path("strength_data_visualizations")

TARGET_COLUMN = "AverageActualStrength28_psi"

FEATURE_COLUMNS = [
    "CalcCementContent_lbs_yd3",
    "FlyAshContent_lbs_yd3",
    "SandSSD_lbs_yd3",
    "AggregateSSD_lbs_yd3",
    "calcWCRatio",
    "SandMoisture_percent",
    "AggregateMoisture_percent",
]


def safe_filename(value: str) -> str:
    """Convert a column name into a safe filename."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


def save_current_figure(filename: str) -> None:
    """Apply layout, save the current figure, and close it."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIRECTORY / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"Created: {output_path}")


def convert_numeric_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert modeling columns to numeric values safely."""
    result = dataframe.copy()

    numeric_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    if "AverageActualAgeDays" in result.columns:
        numeric_columns.append("AverageActualAgeDays")

    if "RequiredStrength" in result.columns:
        numeric_columns.append("RequiredStrength")

    for column in numeric_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    return result


def prepare_model_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Build a cleaned dataset for the initial batch-only strength model.

    Zero is preserved for:
    - Fly ash
    - Sand moisture
    - Aggregate moisture

    Those may be valid measured values.
    """
    result = dataframe.copy()

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in result.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # The prediction target must exist and be positive.
    result = result[
        result[TARGET_COLUMN].notna()
        & result[TARGET_COLUMN].gt(0)
    ].copy()

    # Keep approximately 28-day strength tests.
    # Change this range after confirming the lab's business rule.
    if "AverageActualAgeDays" in result.columns:
        result = result[
            result["AverageActualAgeDays"].between(25, 35)
        ].copy()

    # These quantities should be positive for ordinary concrete.
    positive_columns = [
        "CalcCementContent_lbs_yd3",
        "SandSSD_lbs_yd3",
        "AggregateSSD_lbs_yd3",
        "calcWCRatio",
    ]

    for column in positive_columns:
        result = result[
            result[column].notna()
            & result[column].gt(0)
        ].copy()

    # Zero can be a valid value for these columns.
    nullable_but_zero_valid_columns = [
        "FlyAshContent_lbs_yd3",
        "SandMoisture_percent",
        "AggregateMoisture_percent",
    ]

    result = result.dropna(
        subset=nullable_but_zero_valid_columns
    ).copy()

    return result


def create_profile(
    dataframe: pd.DataFrame,
    name: str,
) -> None:
    """Save descriptive statistics and missing-value counts."""
    selected_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    profile = dataframe[selected_columns].describe(
        percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    ).transpose()

    profile["missing_count"] = (
        dataframe[selected_columns].isna().sum()
    )
    profile["missing_percent"] = (
        dataframe[selected_columns].isna().mean() * 100
    )
    profile["zero_count"] = (
        dataframe[selected_columns].eq(0).sum()
    )
    profile["negative_count"] = (
        dataframe[selected_columns].lt(0).sum()
    )

    output_path = OUTPUT_DIRECTORY / f"{name}_profile.csv"
    profile.to_csv(output_path)

    print(f"Created: {output_path}")


def plot_target_distribution(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    values = dataframe[TARGET_COLUMN].dropna()

    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=40)

    plt.axvline(
        values.mean(),
        linestyle="--",
        label=f"Mean: {values.mean():,.0f} psi",
    )
    plt.axvline(
        values.median(),
        linestyle=":",
        label=f"Median: {values.median():,.0f} psi",
    )

    plt.xlabel("Actual 28-Day Strength (psi)")
    plt.ylabel("Number of Tests")
    plt.title(
        f"Actual 28-Day Strength Distribution — {dataset_name}"
    )
    plt.legend()

    save_current_figure(
        f"{safe_filename(dataset_name)}_target_distribution.png"
    )


def plot_feature_histograms(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    for column in FEATURE_COLUMNS:
        values = dataframe[column].dropna()

        plt.figure(figsize=(10, 6))
        plt.hist(values, bins=40)

        plt.xlabel(column)
        plt.ylabel("Number of Tests")
        plt.title(f"{column} Distribution — {dataset_name}")

        save_current_figure(
            f"{safe_filename(dataset_name)}_"
            f"{safe_filename(column)}_histogram.png"
        )


def plot_feature_boxplots(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    for column in FEATURE_COLUMNS:
        values = dataframe[column].dropna()

        plt.figure(figsize=(10, 4))
        plt.boxplot(
            values,
            vert=False,
            showfliers=True,
        )

        plt.xlabel(column)
        plt.title(f"{column} Boxplot — {dataset_name}")

        save_current_figure(
            f"{safe_filename(dataset_name)}_"
            f"{safe_filename(column)}_boxplot.png"
        )


def plot_features_against_strength(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    for column in FEATURE_COLUMNS:
        plot_data = dataframe[
            [column, TARGET_COLUMN]
        ].dropna()

        plt.figure(figsize=(10, 6))
        plt.scatter(
            plot_data[column],
            plot_data[TARGET_COLUMN],
            alpha=0.25,
            s=18,
        )

        plt.xlabel(column)
        plt.ylabel("Actual 28-Day Strength (psi)")
        plt.title(
            f"Actual Strength vs. {column} — {dataset_name}"
        )

        save_current_figure(
            f"{safe_filename(dataset_name)}_"
            f"strength_vs_{safe_filename(column)}.png"
        )


def plot_correlation_heatmap(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    correlation_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    correlation = dataframe[correlation_columns].corr(
        method="spearman"
    )

    plt.figure(figsize=(11, 9))

    image = plt.imshow(
        correlation,
        vmin=-1,
        vmax=1,
        aspect="auto",
    )

    plt.colorbar(image, label="Spearman Correlation")

    positions = np.arange(len(correlation_columns))

    plt.xticks(
        positions,
        correlation_columns,
        rotation=45,
        ha="right",
    )
    plt.yticks(
        positions,
        correlation_columns,
    )

    for row_index in range(len(correlation_columns)):
        for column_index in range(len(correlation_columns)):
            value = correlation.iloc[row_index, column_index]

            plt.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    plt.title(
        f"Spearman Correlation Heatmap — {dataset_name}"
    )

    save_current_figure(
        f"{safe_filename(dataset_name)}_correlation_heatmap.png"
    )


def plot_strength_by_wcm_range(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    plot_data = dataframe[
        ["calcWCRatio", TARGET_COLUMN]
    ].dropna().copy()

    wcm_bins = [
        0.00,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.70,
        1.00,
    ]

    plot_data["WCMRange"] = pd.cut(
        plot_data["calcWCRatio"],
        bins=wcm_bins,
        right=False,
    )

    grouped_values = []
    grouped_labels = []

    for category, group in plot_data.groupby(
        "WCMRange",
        observed=True,
    ):
        if len(group) < 5:
            continue

        grouped_values.append(
            group[TARGET_COLUMN].to_numpy()
        )
        grouped_labels.append(str(category))

    if not grouped_values:
        print(
            "Skipped W/CM boxplot because no groups had enough rows."
        )
        return

    plt.figure(figsize=(12, 7))
    plt.boxplot(
        grouped_values,
        tick_labels=grouped_labels,
        showfliers=False,
    )

    plt.xlabel("W/CM Range")
    plt.ylabel("Actual 28-Day Strength (psi)")
    plt.title(
        f"Actual Strength by W/CM Range — {dataset_name}"
    )
    plt.xticks(rotation=45, ha="right")

    save_current_figure(
        f"{safe_filename(dataset_name)}_"
        "strength_by_wcm_range.png"
    )


def plot_total_cementitious_vs_strength(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    plot_data = dataframe[
        [
            "CalcCementContent_lbs_yd3",
            "FlyAshContent_lbs_yd3",
            TARGET_COLUMN,
        ]
    ].dropna().copy()

    plot_data["TotalCementitious_lbs_yd3"] = (
        plot_data["CalcCementContent_lbs_yd3"]
        + plot_data["FlyAshContent_lbs_yd3"]
    )

    plt.figure(figsize=(10, 6))
    plt.scatter(
        plot_data["TotalCementitious_lbs_yd3"],
        plot_data[TARGET_COLUMN],
        alpha=0.25,
        s=18,
    )

    plt.xlabel("Cement + Fly Ash (lb/yd³)")
    plt.ylabel("Actual 28-Day Strength (psi)")
    plt.title(
        "Actual Strength vs. Calculated Total Cementitious "
        f"Content — {dataset_name}"
    )

    save_current_figure(
        f"{safe_filename(dataset_name)}_"
        "strength_vs_total_cementitious.png"
    )


def create_suspicious_value_report(
    dataframe: pd.DataFrame,
) -> None:
    """
    Create flags for review.

    These are investigation thresholds, not final engineering rules.
    """
    report = dataframe.copy()

    report["SuspiciousWCM"] = ~report[
        "calcWCRatio"
    ].between(0.20, 0.90)

    report["SuspiciousCement"] = ~report[
        "CalcCementContent_lbs_yd3"
    ].between(200, 1_200)

    report["SuspiciousSand"] = ~report[
        "SandSSD_lbs_yd3"
    ].between(300, 3_500)

    report["SuspiciousAggregate"] = ~report[
        "AggregateSSD_lbs_yd3"
    ].between(300, 3_500)

    report["SuspiciousSandMoisture"] = ~report[
        "SandMoisture_percent"
    ].between(-10, 20)

    report["SuspiciousAggregateMoisture"] = ~report[
        "AggregateMoisture_percent"
    ].between(-10, 20)

    flag_columns = [
        "SuspiciousWCM",
        "SuspiciousCement",
        "SuspiciousSand",
        "SuspiciousAggregate",
        "SuspiciousSandMoisture",
        "SuspiciousAggregateMoisture",
    ]

    suspicious_rows = report[
        report[flag_columns].any(axis=1)
    ].copy()

    output_path = (
        OUTPUT_DIRECTORY / "suspicious_values_for_review.csv"
    )

    suspicious_rows.to_csv(output_path, index=False)

    print(
        f"Suspicious rows: {len(suspicious_rows):,}"
    )
    print(f"Created: {output_path}")


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Data file was not found: {DATA_FILE.resolve()}"
        )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    raw_dataframe = pd.read_csv(DATA_FILE)
    raw_dataframe = convert_numeric_columns(raw_dataframe)

    cleaned_dataframe = prepare_model_data(raw_dataframe)

    print(f"Raw rows:     {len(raw_dataframe):,}")
    print(f"Cleaned rows: {len(cleaned_dataframe):,}")
    print(
        f"Removed rows: "
        f"{len(raw_dataframe) - len(cleaned_dataframe):,}"
    )

    create_profile(raw_dataframe, "raw")
    create_profile(cleaned_dataframe, "cleaned")

    # Visualize the target before and after cleaning.
    plot_target_distribution(
        raw_dataframe,
        "Raw Data",
    )
    plot_target_distribution(
        cleaned_dataframe,
        "Cleaned Data",
    )

    # Detailed plots for the cleaned model dataset.
    plot_feature_histograms(
        cleaned_dataframe,
        "Cleaned Data",
    )
    plot_feature_boxplots(
        cleaned_dataframe,
        "Cleaned Data",
    )
    plot_features_against_strength(
        cleaned_dataframe,
        "Cleaned Data",
    )
    plot_correlation_heatmap(
        cleaned_dataframe,
        "Cleaned Data",
    )
    plot_strength_by_wcm_range(
        cleaned_dataframe,
        "Cleaned Data",
    )
    plot_total_cementitious_vs_strength(
        cleaned_dataframe,
        "Cleaned Data",
    )

    create_suspicious_value_report(raw_dataframe)

    print(
        "\nVisualization complete. "
        f"Open the folder: {OUTPUT_DIRECTORY.resolve()}"
    )


if __name__ == "__main__":
    main()