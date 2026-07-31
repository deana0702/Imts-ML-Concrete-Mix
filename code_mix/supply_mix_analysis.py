#!/usr/bin/env python3
"""
Concrete Mix Design Review Analytics

Reads an IMTS concrete dataset from Excel or CSV, analyzes valid 28-day
compressive-strength results over a configurable recent period, and saves:

- Test-level analytical data
- Mix-level performance summary
- Mix design review candidates
- Automatically generated candidate messages
- Data-quality summary
- PNG visualizations
- A simple HTML report

Default grouping:
    SupplierName + plantNumber + mixNumber + RequiredStrength

Default candidate rule:
    TestCount >= 30
    MeanMarginPsi >= 1000 psi
    P05MarginPsi >= 300 psi
    FailureRatePct <= 1%

Strength margin:
    AverageActualStrength28_psi - RequiredStrength
"""

from __future__ import annotations

import argparse
import html
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASEPATH = Path("data/prepared_28_day_standard_cure")
INPUT_FILE = BASEPATH / (
    "03_standard_cured_test_level_28_working_data_drop_rows_drop_columns_1.csv"
)
OUTPUT_PATH = BASEPATH / "supply_mix_analysis"

REQUIRED_COLUMNS = {
    "testId",
    "SupplierName",
    "castDate",
    "plantNumber",
    "mixNumber",
    "AverageActualStrength28_psi",
    "AverageActualAgeDays",
    "SpecifiedBreakAge",
    "RequiredStrength",
    "SpecifiedStrengthMissing",
    "SpecifiedBreakAgeIs28",
}

GROUP_COLUMNS = [
    "SupplierName",
    "plantNumber",
    "mixNumber",
    "RequiredStrength",
]


@dataclass(frozen=True)
class CandidateThresholds:
    min_tests: int
    min_mean_margin_psi: float
    min_p05_margin_psi: float
    max_failure_rate_pct: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze IMTS concrete 28-day strength performance and identify "
            "mix-design review candidates."
        )
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Input .xlsx, .xls, or .csv file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mix_review_output"),
        help="Directory for CSV, PNG, TXT, and HTML outputs.",
    )
    parser.add_argument(
        "--sheet-name",
        default=0,
        help=(
            "Excel sheet name or zero-based sheet index. "
            "Ignored for CSV. Default: first sheet."
        ),
    )
    parser.add_argument(
        "--months",
        type=int,
        default=18,
        help="Number of months included in the analysis. Default: 18.",
    )
    parser.add_argument(
        "--as-of-date",
        default=None,
        help=(
            "Analysis end date in YYYY-MM-DD format. "
            "Default: maximum castDate found in the dataset."
        ),
    )
    parser.add_argument(
        "--min-age-days",
        type=float,
        default=27,
        help="Minimum accepted actual break age. Default: 27.",
    )
    parser.add_argument(
        "--max-age-days",
        type=float,
        default=31,
        help="Maximum accepted actual break age. Default: 31.",
    )
    parser.add_argument(
        "--min-tests",
        type=int,
        default=30,
        help="Minimum unique tests for candidate status. Default: 30.",
    )
    parser.add_argument(
        "--min-mean-margin",
        type=float,
        default=1000,
        help="Minimum mean strength margin in psi. Default: 1000.",
    )
    parser.add_argument(
        "--min-p05-margin",
        type=float,
        default=300,
        help="Minimum fifth-percentile margin in psi. Default: 300.",
    )
    parser.add_argument(
        "--max-failure-rate",
        type=float,
        default=1.0,
        help="Maximum below-required rate in percent. Default: 1.0.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Maximum groups shown in ranking charts. Default: 25.",
    )
    parser.add_argument(
        "--trend-limit",
        type=int,
        default=10,
        help="Maximum candidate trend charts to save. Default: 10.",
    )
    parser.add_argument(
        "--no-age-filter",
        action="store_true",
        help="Do not filter by AverageActualAgeDays.",
    )
    return parser.parse_args()


def read_dataset(path: Path, sheet_name: object = 0) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls", ".xlsm"}:
        parsed_sheet = sheet_name
        if isinstance(sheet_name, str) and sheet_name.isdigit():
            parsed_sheet = int(sheet_name)
        return pd.read_excel(path, sheet_name=parsed_sheet)

    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)

    raise ValueError(
        f"Unsupported input type '{suffix}'. Use .xlsx, .xls, .xlsm, or .csv."
    )


def validate_columns(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        formatted = "\n  - ".join(missing)
        raise ValueError(
            "The input dataset is missing required columns:\n  - " + formatted
        )


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    true_values = {"true", "1", "yes", "y", "t"}
    false_values = {"false", "0", "no", "n", "f", ""}

    normalized = series.astype("string").str.strip().str.lower()

    parsed = normalized.map(
        lambda value: (
            True
            if value in true_values
            else False
            if value in false_values or pd.isna(value)
            else np.nan
        )
    )

    return parsed.fillna(False).astype(bool)


def safe_name(value: object, max_length: int = 80) -> str:
    text = str(value)
    text = re.sub(r'[<>:"/\\|?*]+', "_", text)
    text = re.sub(r"\s+", "_", text.strip())
    text = text.strip("._")
    return (text or "unknown")[:max_length]


def format_group_label(row: pd.Series) -> str:
    return (
        f"{row['SupplierName']} | Plant {row['plantNumber']} | "
        f"Mix {row['mixNumber']} | Req {row['RequiredStrength']:,.0f}"
    )


def prepare_analysis_data(
    raw: pd.DataFrame,
    months: int,
    as_of_date: str | None,
    min_age_days: float,
    max_age_days: float,
    use_age_filter: bool,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, dict[str, int]]:
    df = raw.copy()

    original_rows = len(df)

    df["castDate"] = pd.to_datetime(df["castDate"], errors="coerce")

    numeric_columns = [
        "AverageActualStrength28_psi",
        "AverageActualAgeDays",
        "SpecifiedBreakAge",
        "RequiredStrength",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["SpecifiedBreakAgeIs28"] = parse_bool_series(
        df["SpecifiedBreakAgeIs28"]
    )
    df["SpecifiedStrengthMissing"] = parse_bool_series(
        df["SpecifiedStrengthMissing"]
    )

    for column in ["SupplierName", "plantNumber", "mixNumber"]:
        df[column] = df[column].astype("string").str.strip()
        df[column] = df[column].replace(
            {"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA}
        )

    if as_of_date:
        analysis_end = pd.Timestamp(as_of_date).normalize()
    else:
        analysis_end = df["castDate"].max()
        if pd.isna(analysis_end):
            raise ValueError("No valid castDate values were found.")
        analysis_end = analysis_end.normalize()

    analysis_start = analysis_end - pd.DateOffset(months=months)

    valid_28_day_flag = (
        df["SpecifiedBreakAgeIs28"]
        | df["SpecifiedBreakAge"].eq(28)
    )

    mask = (
        df["castDate"].between(analysis_start, analysis_end, inclusive="both")
        & valid_28_day_flag
        & ~df["SpecifiedStrengthMissing"]
        & df["RequiredStrength"].notna()
        & df["AverageActualStrength28_psi"].notna()
        & df["testId"].notna()
        & df["SupplierName"].notna()
        & df["mixNumber"].notna()
        & df["RequiredStrength"].gt(0)
        & df["AverageActualStrength28_psi"].gt(0)
    )

    if use_age_filter:
        mask &= df["AverageActualAgeDays"].between(
            min_age_days,
            max_age_days,
            inclusive="both",
        )

    filtered = df.loc[mask].copy()

    filtered["plantNumber"] = filtered["plantNumber"].fillna("<Unknown>")
    filtered["StrengthMargin_psi"] = (
        filtered["AverageActualStrength28_psi"]
        - filtered["RequiredStrength"]
    )
    filtered["IsBelowRequired"] = (
        filtered["AverageActualStrength28_psi"]
        < filtered["RequiredStrength"]
    )

    # If the source contains repeated rows for the same test/group, collapse
    # them to one analytical test record. Average strength is used safely.
    test_level = (
        filtered.groupby(
            ["testId", *GROUP_COLUMNS],
            dropna=False,
            as_index=False,
        )
        .agg(
            castDate=("castDate", "min"),
            AverageActualStrength28_psi=(
                "AverageActualStrength28_psi",
                "mean",
            ),
            AverageActualAgeDays=("AverageActualAgeDays", "mean"),
            SpecifiedBreakAge=("SpecifiedBreakAge", "first"),
        )
    )

    test_level["StrengthMargin_psi"] = (
        test_level["AverageActualStrength28_psi"]
        - test_level["RequiredStrength"]
    )
    test_level["IsBelowRequired"] = (
        test_level["AverageActualStrength28_psi"]
        < test_level["RequiredStrength"]
    )

    quality = {
        "OriginalRows": original_rows,
        "RowsAfterFiltering": len(filtered),
        "UniqueTestsAfterDeduplication": test_level["testId"].nunique(),
        "DuplicateRowsCollapsed": max(len(filtered) - len(test_level), 0),
        "DistinctMixGroups": int(
            test_level[GROUP_COLUMNS].drop_duplicates().shape[0]
        ),
    }

    return test_level, analysis_start, analysis_end, quality


def percentile_rank(series: pd.Series) -> pd.Series:
    if len(series) <= 1:
        return pd.Series(100.0, index=series.index)
    return series.rank(method="average", pct=True) * 100


def summarize_mix_performance(
    test_level: pd.DataFrame,
    thresholds: CandidateThresholds,
) -> pd.DataFrame:
    if test_level.empty:
        return pd.DataFrame()

    summary = (
        test_level.groupby(GROUP_COLUMNS, dropna=False)
        .agg(
            TestCount=("testId", "nunique"),
            MeanActualStrengthPsi=(
                "AverageActualStrength28_psi",
                "mean",
            ),
            MedianActualStrengthPsi=(
                "AverageActualStrength28_psi",
                "median",
            ),
            MeanMarginPsi=("StrengthMargin_psi", "mean"),
            MedianMarginPsi=("StrengthMargin_psi", "median"),
            StdDevMarginPsi=("StrengthMargin_psi", "std"),
            MinimumMarginPsi=("StrengthMargin_psi", "min"),
            P05MarginPsi=(
                "StrengthMargin_psi",
                lambda values: values.quantile(0.05),
            ),
            P95MarginPsi=(
                "StrengthMargin_psi",
                lambda values: values.quantile(0.95),
            ),
            MaximumMarginPsi=("StrengthMargin_psi", "max"),
            FailureCount=("IsBelowRequired", "sum"),
            FirstCastDate=("castDate", "min"),
            LastCastDate=("castDate", "max"),
        )
        .reset_index()
    )

    summary["StdDevMarginPsi"] = summary["StdDevMarginPsi"].fillna(0)
    summary["FailureRatePct"] = (
        summary["FailureCount"] / summary["TestCount"] * 100
    )

    summary["CoefficientOfVariationPct"] = np.where(
        summary["MeanActualStrengthPsi"].ne(0),
        summary["StdDevMarginPsi"]
        / summary["MeanActualStrengthPsi"]
        * 100,
        np.nan,
    )

    summary["NormalApproxLower5MarginPsi"] = (
        summary["MeanMarginPsi"]
        - 1.645 * summary["StdDevMarginPsi"]
    )

    summary["PeerPercentileWithinRequiredStrength"] = (
        summary.groupby("RequiredStrength", dropna=False)["MeanMarginPsi"]
        .transform(percentile_rank)
    )

    summary["PeerPercentileWithinSupplierAndRequired"] = (
        summary.groupby(
            ["SupplierName", "RequiredStrength"],
            dropna=False,
        )["MeanMarginPsi"]
        .transform(percentile_rank)
    )

    summary["IsReviewCandidate"] = (
        summary["TestCount"].ge(thresholds.min_tests)
        & summary["MeanMarginPsi"].ge(
            thresholds.min_mean_margin_psi
        )
        & summary["P05MarginPsi"].ge(
            thresholds.min_p05_margin_psi
        )
        & summary["FailureRatePct"].le(
            thresholds.max_failure_rate_pct
        )
    )

    summary["Status"] = np.select(
        [
            summary["IsReviewCandidate"],
            summary["TestCount"].lt(thresholds.min_tests),
            summary["FailureRatePct"].gt(
                thresholds.max_failure_rate_pct
            ),
            summary["P05MarginPsi"].lt(
                thresholds.min_p05_margin_psi
            ),
        ],
        [
            "Mix Design Review Candidate",
            "Insufficient Data",
            "Below-Strength Risk",
            "High Variability / Low Tail",
        ],
        default="Not Candidate",
    )

    summary["GroupLabel"] = summary.apply(format_group_label, axis=1)

    round_columns = [
        "MeanActualStrengthPsi",
        "MedianActualStrengthPsi",
        "MeanMarginPsi",
        "MedianMarginPsi",
        "StdDevMarginPsi",
        "MinimumMarginPsi",
        "P05MarginPsi",
        "P95MarginPsi",
        "MaximumMarginPsi",
        "FailureRatePct",
        "CoefficientOfVariationPct",
        "NormalApproxLower5MarginPsi",
        "PeerPercentileWithinRequiredStrength",
        "PeerPercentileWithinSupplierAndRequired",
    ]
    summary[round_columns] = summary[round_columns].round(2)

    return summary.sort_values(
        ["IsReviewCandidate", "MeanMarginPsi", "TestCount"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_candidate_message(row: pd.Series, months: int) -> str:
    failure_phrase = (
        "no below-required results"
        if row["FailureCount"] == 0
        else (
            f"{int(row['FailureCount'])} below-required result(s), "
            f"a {row['FailureRatePct']:.1f}% rate"
        )
    )

    return (
        f"Over the past {months} months, {row['SupplierName']} "
        f"Plant {row['plantNumber']} Mix {row['mixNumber']} "
        f"averaged {row['MeanMarginPsi']:,.0f} psi above its "
        f"{row['RequiredStrength']:,.0f}-psi required 28-day strength "
        f"across {int(row['TestCount'])} valid tests. The lower "
        f"5th-percentile margin was {row['P05MarginPsi']:,.0f} psi, "
        f"with {failure_phrase}. This mix may be a candidate for "
        f"engineering mix-design review."
    )


def save_data_quality_summary(
    output_dir: Path,
    quality: dict[str, int],
    analysis_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
    thresholds: CandidateThresholds,
) -> None:
    rows = [
        ("AnalysisStartDate", analysis_start.date().isoformat()),
        ("AnalysisEndDate", analysis_end.date().isoformat()),
        *quality.items(),
        ("CandidateMinTests", thresholds.min_tests),
        (
            "CandidateMinMeanMarginPsi",
            thresholds.min_mean_margin_psi,
        ),
        (
            "CandidateMinP05MarginPsi",
            thresholds.min_p05_margin_psi,
        ),
        (
            "CandidateMaxFailureRatePct",
            thresholds.max_failure_rate_pct,
        ),
    ]

    pd.DataFrame(rows, columns=["Metric", "Value"]).to_csv(
        output_dir / "data_quality_summary.csv",
        index=False,
    )


def save_ranked_mean_margin_chart(
    summary: pd.DataFrame,
    charts_dir: Path,
    top_n: int,
) -> Path | None:
    eligible = summary.loc[summary["TestCount"] >= 3].copy()
    if eligible.empty:
        return None

    plotted = eligible.nlargest(top_n, "MeanMarginPsi").sort_values(
        "MeanMarginPsi"
    )

    fig_height = max(6, min(18, len(plotted) * 0.42))
    fig, ax = plt.subplots(figsize=(13, fig_height))

    ax.barh(plotted["GroupLabel"], plotted["MeanMarginPsi"])
    ax.axvline(0, linewidth=1)
    ax.set_title(
        f"Top {len(plotted)} Mix Groups by Average 28-Day Strength Margin"
    )
    ax.set_xlabel("Average Actual Strength − Required Strength (psi)")
    ax.set_ylabel("Supplier | Plant | Mix | Required Strength")
    ax.grid(axis="x", alpha=0.25)

    for index, value in enumerate(plotted["MeanMarginPsi"]):
        ax.text(
            value,
            index,
            f" {value:,.0f} psi",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    path = charts_dir / "01_ranked_average_strength_margin.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_percentile_range_chart(
    summary: pd.DataFrame,
    charts_dir: Path,
    top_n: int,
) -> Path | None:
    eligible = summary.loc[summary["TestCount"] >= 3].copy()
    if eligible.empty:
        return None

    plotted = eligible.nlargest(top_n, "MeanMarginPsi").sort_values(
        "MeanMarginPsi"
    )

    lower_error = np.maximum(
        plotted["MeanMarginPsi"] - plotted["P05MarginPsi"],
        0,
    )
    upper_error = np.maximum(
        plotted["P95MarginPsi"] - plotted["MeanMarginPsi"],
        0,
    )

    fig_height = max(6, min(18, len(plotted) * 0.42))
    fig, ax = plt.subplots(figsize=(13, fig_height))

    y_positions = np.arange(len(plotted))
    ax.errorbar(
        plotted["MeanMarginPsi"],
        y_positions,
        xerr=np.vstack([lower_error, upper_error]),
        fmt="o",
        capsize=3,
    )
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plotted["GroupLabel"])
    ax.set_title(
        "Mean Strength Margin with 5th–95th Percentile Range"
    )
    ax.set_xlabel("Strength Margin (psi)")
    ax.set_ylabel("Supplier | Plant | Mix | Required Strength")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    path = charts_dir / "02_mean_margin_percentile_range.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_margin_variability_scatter(
    summary: pd.DataFrame,
    charts_dir: Path,
    thresholds: CandidateThresholds,
) -> Path | None:
    eligible = summary.loc[summary["TestCount"] >= 2].copy()
    if eligible.empty:
        return None

    sizes = 25 + np.sqrt(eligible["TestCount"]) * 18

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(
        eligible["StdDevMarginPsi"],
        eligible["MeanMarginPsi"],
        s=sizes,
        alpha=0.75,
    )

    ax.axhline(thresholds.min_mean_margin_psi, linewidth=1)
    ax.set_title(
        "Average Strength Margin vs. Variability\n"
        "Bubble size represents unique test count"
    )
    ax.set_xlabel("Standard Deviation of Strength Margin (psi)")
    ax.set_ylabel("Average Strength Margin (psi)")
    ax.grid(alpha=0.25)

    labels = eligible.nlargest(
        min(12, len(eligible)),
        "MeanMarginPsi",
    )
    for _, row in labels.iterrows():
        ax.annotate(
            f"{row['SupplierName']} | {row['mixNumber']}",
            (
                row["StdDevMarginPsi"],
                row["MeanMarginPsi"],
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )

    fig.tight_layout()
    path = charts_dir / "03_margin_vs_variability.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_failure_risk_chart(
    summary: pd.DataFrame,
    charts_dir: Path,
    top_n: int,
) -> Path | None:
    eligible = summary.loc[summary["TestCount"] >= 3].copy()
    if eligible.empty:
        return None

    plotted = eligible.sort_values(
        ["FailureRatePct", "TestCount"],
        ascending=[False, False],
    ).head(top_n)
    plotted = plotted.sort_values("FailureRatePct")

    fig_height = max(6, min(18, len(plotted) * 0.42))
    fig, ax = plt.subplots(figsize=(13, fig_height))
    ax.barh(plotted["GroupLabel"], plotted["FailureRatePct"])
    ax.set_title("Mix Groups with Highest Below-Required Strength Rates")
    ax.set_xlabel("Below-Required Test Rate (%)")
    ax.set_ylabel("Supplier | Plant | Mix | Required Strength")
    ax.grid(axis="x", alpha=0.25)

    for index, value in enumerate(plotted["FailureRatePct"]):
        ax.text(
            value,
            index,
            f" {value:.1f}%",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    path = charts_dir / "04_below_required_failure_rate.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_candidate_boxplot(
    test_level: pd.DataFrame,
    summary: pd.DataFrame,
    charts_dir: Path,
    top_n: int,
) -> Path | None:
    selected = summary.loc[summary["IsReviewCandidate"]].head(top_n)
    if selected.empty:
        selected = summary.loc[summary["TestCount"] >= 5].head(top_n)

    if selected.empty:
        return None

    key_frame = selected[GROUP_COLUMNS].copy()
    merged = test_level.merge(
        key_frame,
        on=GROUP_COLUMNS,
        how="inner",
    )

    if merged.empty:
        return None

    selected_labels = selected["GroupLabel"].tolist()
    label_lookup = {
        tuple(row[column] for column in GROUP_COLUMNS): row["GroupLabel"]
        for _, row in selected.iterrows()
    }

    merged["GroupLabel"] = merged.apply(
        lambda row: label_lookup[
            tuple(row[column] for column in GROUP_COLUMNS)
        ],
        axis=1,
    )

    data = [
        merged.loc[
            merged["GroupLabel"].eq(label),
            "StrengthMargin_psi",
        ].dropna().to_numpy()
        for label in selected_labels
    ]

    valid_pairs = [
        (label, values)
        for label, values in zip(selected_labels, data)
        if len(values) > 0
    ]
    if not valid_pairs:
        return None

    labels, values = zip(*valid_pairs)

    fig_height = max(6, min(18, len(labels) * 0.5))
    fig, ax = plt.subplots(figsize=(13, fig_height))
    ax.boxplot(values, vert=False, tick_labels=labels, showfliers=True)
    ax.axvline(0, linewidth=1)
    ax.set_title("Strength Margin Distribution for Candidate / Top Mix Groups")
    ax.set_xlabel("Actual Strength − Required Strength (psi)")
    ax.set_ylabel("Supplier | Plant | Mix | Required Strength")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    path = charts_dir / "05_candidate_strength_margin_boxplot.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_candidate_trends(
    test_level: pd.DataFrame,
    summary: pd.DataFrame,
    trends_dir: Path,
    limit: int,
) -> list[Path]:
    candidates = summary.loc[summary["IsReviewCandidate"]].head(limit)
    saved: list[Path] = []

    for _, candidate in candidates.iterrows():
        mask = pd.Series(True, index=test_level.index)
        for column in GROUP_COLUMNS:
            mask &= test_level[column].eq(candidate[column])

        group_data = (
            test_level.loc[mask]
            .sort_values("castDate")
            .copy()
        )

        if group_data.empty:
            continue

        group_data["RollingMeanMarginPsi"] = (
            group_data["StrengthMargin_psi"]
            .rolling(window=min(5, len(group_data)), min_periods=1)
            .mean()
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(
            group_data["castDate"],
            group_data["StrengthMargin_psi"],
            label="Test margin",
        )
        ax.plot(
            group_data["castDate"],
            group_data["RollingMeanMarginPsi"],
            label="Rolling mean margin",
        )
        ax.axhline(0, linewidth=1, label="Required-strength boundary")
        ax.axhline(
            candidate["MeanMarginPsi"],
            linewidth=1,
            linestyle="--",
            label="Group mean margin",
        )
        ax.set_title(
            f"Strength Margin Trend\n{candidate['GroupLabel']}"
        )
        ax.set_xlabel("Cast Date")
        ax.set_ylabel("Actual Strength − Required Strength (psi)")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()

        filename = (
            f"{safe_name(candidate['SupplierName'])}_"
            f"{safe_name(candidate['plantNumber'])}_"
            f"{safe_name(candidate['mixNumber'])}_"
            f"req_{candidate['RequiredStrength']:.0f}.png"
        )
        path = trends_dir / filename
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    return saved


def create_html_report(
    output_dir: Path,
    summary: pd.DataFrame,
    candidates: pd.DataFrame,
    chart_paths: Iterable[Path],
    trend_paths: Iterable[Path],
    analysis_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
    months: int,
) -> Path:
    report_path = output_dir / "mix_review_report.html"

    summary_columns = [
        "SupplierName",
        "plantNumber",
        "mixNumber",
        "RequiredStrength",
        "TestCount",
        "MeanActualStrengthPsi",
        "MeanMarginPsi",
        "P05MarginPsi",
        "StdDevMarginPsi",
        "FailureRatePct",
        "Status",
    ]

    candidate_table = (
        candidates[summary_columns].head(50).to_html(
            index=False,
            border=0,
            classes="data-table",
        )
        if not candidates.empty
        else "<p>No groups met the configured candidate criteria.</p>"
    )

    top_summary = summary[summary_columns].head(100).to_html(
        index=False,
        border=0,
        classes="data-table",
    )

    chart_html = "\n".join(
        (
            f'<section><h2>{html.escape(path.stem.replace("_", " ").title())}'
            f'</h2><img src="{html.escape(path.relative_to(output_dir).as_posix())}" '
            f'alt="{html.escape(path.stem)}"></section>'
        )
        for path in chart_paths
        if path is not None
    )

    trend_html = "\n".join(
        (
            f'<section><img src="{html.escape(path.relative_to(output_dir).as_posix())}" '
            f'alt="{html.escape(path.stem)}"></section>'
        )
        for path in trend_paths
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>IMTS Concrete Mix Review Analytics</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 32px;
    line-height: 1.45;
    color: #222;
}}
h1, h2 {{ margin-top: 1.4em; }}
.summary-card {{
    display: inline-block;
    padding: 12px 16px;
    margin: 6px;
    border: 1px solid #bbb;
    border-radius: 6px;
    vertical-align: top;
}}
img {{
    max-width: 100%;
    height: auto;
    border: 1px solid #ddd;
}}
.data-table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 13px;
}}
.data-table th, .data-table td {{
    border: 1px solid #ccc;
    padding: 6px 8px;
    text-align: right;
}}
.data-table th:nth-child(-n+3),
.data-table td:nth-child(-n+3),
.data-table th:last-child,
.data-table td:last-child {{
    text-align: left;
}}
.note {{
    background: #f5f5f5;
    padding: 12px;
    border-left: 4px solid #999;
}}
</style>
</head>
<body>
<h1>IMTS Concrete Mix Design Review Analytics</h1>
<p>Analysis period: <strong>{analysis_start.date()}</strong> through
<strong>{analysis_end.date()}</strong> ({months} months).</p>

<div class="summary-card"><strong>Mix groups</strong><br>{len(summary):,}</div>
<div class="summary-card"><strong>Review candidates</strong><br>{len(candidates):,}</div>

<p class="note">
A candidate flag is a statistical screening result, not an instruction to
change a mix design. Engineering review, specification review, trial batches,
and appropriate approvals remain necessary.
</p>

<h2>Mix Design Review Candidates</h2>
{candidate_table}

{chart_html}

<h2>Candidate Trend Charts</h2>
{trend_html if trend_html else "<p>No candidate trend charts were generated.</p>"}

<h2>Top Mix Performance Summary</h2>
{top_summary}
</body>
</html>
"""

    report_path.write_text(document, encoding="utf-8")
    return report_path


def main() -> int:
    args = parse_args()

    if args.months <= 0:
        raise ValueError("--months must be greater than zero.")
    if args.min_tests <= 0:
        raise ValueError("--min-tests must be greater than zero.")
    if args.top_n <= 0:
        raise ValueError("--top-n must be greater than zero.")

    thresholds = CandidateThresholds(
        min_tests=args.min_tests,
        min_mean_margin_psi=args.min_mean_margin,
        min_p05_margin_psi=args.min_p05_margin,
        max_failure_rate_pct=args.max_failure_rate,
    )
    
    output_dir = OUTPUT_PATH
    charts_dir = output_dir / "charts"
    trends_dir = charts_dir / "candidate_trends"

    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    trends_dir.mkdir(parents=True, exist_ok=True)

    raw = read_dataset(INPUT_FILE, args.sheet_name)
    validate_columns(raw)

    test_level, analysis_start, analysis_end, quality = (
        prepare_analysis_data(
            raw=raw,
            months=args.months,
            as_of_date=args.as_of_date,
            min_age_days=args.min_age_days,
            max_age_days=args.max_age_days,
            use_age_filter=not args.no_age_filter,
        )
    )

    if test_level.empty:
        raise ValueError(
            "No valid records remained after filtering. Check date range, "
            "28-day flags, required strength, actual strength, and age limits."
        )

    summary = summarize_mix_performance(test_level, thresholds)
    candidates = summary.loc[summary["IsReviewCandidate"]].copy()

    test_level.to_csv(
        output_dir / "filtered_test_level_data.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "mix_performance_summary.csv",
        index=False,
    )
    candidates.to_csv(
        output_dir / "mix_review_candidates.csv",
        index=False,
    )

    candidate_messages = [
        build_candidate_message(row, args.months)
        for _, row in candidates.iterrows()
    ]
    (output_dir / "candidate_messages.txt").write_text(
        "\n\n".join(candidate_messages)
        if candidate_messages
        else "No groups met the configured candidate criteria.\n",
        encoding="utf-8",
    )

    save_data_quality_summary(
        output_dir=output_dir,
        quality=quality,
        analysis_start=analysis_start,
        analysis_end=analysis_end,
        thresholds=thresholds,
    )

    chart_paths = [
        save_ranked_mean_margin_chart(
            summary,
            charts_dir,
            args.top_n,
        ),
        save_percentile_range_chart(
            summary,
            charts_dir,
            args.top_n,
        ),
        save_margin_variability_scatter(
            summary,
            charts_dir,
            thresholds,
        ),
        save_failure_risk_chart(
            summary,
            charts_dir,
            args.top_n,
        ),
        save_candidate_boxplot(
            test_level,
            summary,
            charts_dir,
            min(args.top_n, 15),
        ),
    ]
    chart_paths = [path for path in chart_paths if path is not None]

    trend_paths = save_candidate_trends(
        test_level=test_level,
        summary=summary,
        trends_dir=trends_dir,
        limit=args.trend_limit,
    )

    report_path = create_html_report(
        output_dir=output_dir,
        summary=summary,
        candidates=candidates,
        chart_paths=chart_paths,
        trend_paths=trend_paths,
        analysis_start=analysis_start,
        analysis_end=analysis_end,
        months=args.months,
    )

    print("Analysis completed successfully.")
    print(f"Input rows: {quality['OriginalRows']:,}")
    print(
        "Valid unique tests: "
        f"{quality['UniqueTestsAfterDeduplication']:,}"
    )
    print(f"Mix groups: {len(summary):,}")
    print(f"Review candidates: {len(candidates):,}")
    print(f"Output directory: {output_dir}")
    print(f"HTML report: {report_path}")

    return 0


if __name__ == "__main__":
    main()