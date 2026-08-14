"""Historical IMTS mix performance evaluation for materials-engineer review.

Workflow
--------
1. Read the prepared batch-strength data created by script 04.
2. Apply office and optional project/supplier/plant/mix filters.
3. Keep projects with enough unique concrete tests.
4. Within each project, aggregate Supplier + Plant + Mix + Required Strength.
5. Keep mix groups with enough unique tests and assign an engineering-review flag.

This is historical screening, not automatic mix approval or mix redesign.
No command-line arguments are used. Edit the settings below, then run:

    python 07_historical_mix_performance_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import mix_config as cfg


# =============================================================================
# EDITABLE SETTINGS
# Use None to keep every value. Use a list to select one or more values.
# Examples: OFFICE_ID_FILTER = [2], PROJECT_ID_FILTER = [7627, 8993]
# =============================================================================
OFFICE_ID_FILTER = 2
PROJECT_ID_FILTER = None
SUPPLIER_FILTER = None
PLANT_FILTER = None
MIX_FILTER = None

# First require enough tests in the project, then enough tests in each
# Project + Supplier + Plant + Mix + Required Strength evaluation group.
MIN_PROJECT_TEST_COUNT = 30
MIN_MIX_GROUP_TEST_COUNT = 30

# Review rules. These are screening thresholds, not engineering approval rules.
OPTIMIZATION_MAX_FAILURE_RATE_PERCENT = 1.0
OPTIMIZATION_MIN_AVG_MARGIN_PSI = 1200.0
OPTIMIZATION_REQUIRE_POSITIVE_P10_MARGIN = True
OPTIMIZATION_MAX_STRENGTH_CV_PERCENT = 15.0

BENCHMARK_MAX_FAILURE_RATE_PERCENT = 1.0
BENCHMARK_MIN_AVG_MARGIN_PSI = 700.0
BENCHMARK_MAX_AVG_MARGIN_PSI = 1500.0
BENCHMARK_MAX_STRENGTH_CV_PERCENT = 15.0

RISK_MIN_FAILURE_RATE_PERCENT = 5.0
RISK_WHEN_P10_MARGIN_BELOW_PSI = 0.0

MAX_ROWS_TO_PRINT = 50

OUTPUT_DIR = cfg.OUTPUT_DIR / "historical_mix_performance"


def normalize_filter_values(values):
    if values is None:
        return None
    if isinstance(values, (str, int, float)):
        return [values]
    return list(values)


def apply_filter(
    df: pd.DataFrame,
    column: str,
    values,
    label: str,
) -> pd.DataFrame:
    selected = normalize_filter_values(values)
    if selected is None:
        return df
    if column not in df.columns:
        raise ValueError(f"Cannot apply {label}: column '{column}' is missing.")

    # String comparison allows numeric IDs entered as either 2 or "2".
    wanted = {str(value).strip().casefold() for value in selected}
    actual = df[column].astype("string").str.strip().str.casefold()
    result = df[actual.isin(wanted)].copy()
    print(f"{label}: {len(df):,} -> {len(result):,} rows")
    return result


def find_optional_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def safe_p10(series: pd.Series) -> float:
    clean = series.dropna()
    return float(clean.quantile(0.10)) if len(clean) else np.nan


def safe_mean(series: pd.Series) -> float:
    clean = series.dropna()
    return float(clean.mean()) if len(clean) else np.nan


def safe_median(series: pd.Series) -> float:
    clean = series.dropna()
    return float(clean.median()) if len(clean) else np.nan


def safe_std(series: pd.Series) -> float:
    clean = series.dropna()
    return float(clean.std(ddof=1)) if len(clean) >= 2 else np.nan


def prepare_analysis_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    required = [
        "testId",
        "officeId",
        "projectId",
        "SupplierName",
        "plantNumber",
        "mixNumber",
        cfg.STRENGTH_TARGET,
        cfg.REQUIRED_STRENGTH_COLUMN,
        "CalcCementContent_lbs_yd3",
        "FlyAshContent_lbs_yd3",
        "TotalCementitiousContent_lbs_yd3",
        "calcWCRatio",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("Prepared data is missing required columns: " + ", ".join(missing))

    numeric_columns = [
        cfg.STRENGTH_TARGET,
        cfg.REQUIRED_STRENGTH_COLUMN,
        "CalcCementContent_lbs_yd3",
        "FlyAshContent_lbs_yd3",
        "TotalCementitiousContent_lbs_yd3",
        "calcWCRatio",
        cfg.SLUMP_TARGET,
        cfg.AIR_TARGET,
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    audit = {"input_rows": int(len(df))}

    df = apply_filter(df, "officeId", OFFICE_ID_FILTER, "Office filter")
    audit["rows_after_office_filter"] = int(len(df))
    df = apply_filter(df, "projectId", PROJECT_ID_FILTER, "Project filter")
    audit["rows_after_project_filter"] = int(len(df))
    df = apply_filter(df, "SupplierName", SUPPLIER_FILTER, "Supplier filter")
    df = apply_filter(df, "plantNumber", PLANT_FILTER, "Plant filter")
    df = apply_filter(df, "mixNumber", MIX_FILTER, "Mix filter")
    audit["rows_after_all_user_filters"] = int(len(df))

    identifiers = ["officeId", "projectId", "SupplierName", "plantNumber", "mixNumber"]
    valid = (
        df[identifiers].notna().all(axis=1)
        & df[cfg.STRENGTH_TARGET].notna()
        & df[cfg.REQUIRED_STRENGTH_COLUMN].notna()
    )
    audit["rows_missing_identifier_strength_or_spec"] = int((~valid).sum())
    df = df[valid].drop_duplicates("testId", keep="first").copy()
    audit["eligible_unique_test_rows"] = int(len(df))

    df["StrengthMargin_psi"] = (
        df[cfg.STRENGTH_TARGET] - df[cfg.REQUIRED_STRENGTH_COLUMN]
    )
    df["StrengthFailure"] = (
        df[cfg.STRENGTH_TARGET] < df[cfg.REQUIRED_STRENGTH_COLUMN]
    ).astype(int)

    return df, audit


def build_project_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["officeId", "projectId"], dropna=False)
        .agg(
            ProjectTestCount=("testId", "nunique"),
            SupplierCount=("SupplierName", "nunique"),
            PlantCount=("plantNumber", "nunique"),
            MixCount=("mixNumber", "nunique"),
            AverageActualStrength28_psi=(cfg.STRENGTH_TARGET, "mean"),
            FailureCount=("StrengthFailure", "sum"),
        )
        .reset_index()
    )
    summary["FailureRate_percent"] = np.where(
        summary["ProjectTestCount"] > 0,
        100.0 * summary["FailureCount"] / summary["ProjectTestCount"],
        np.nan,
    )
    summary["EligibleProject"] = summary["ProjectTestCount"].ge(
        MIN_PROJECT_TEST_COUNT
    )
    return summary


def aggregate_mix_groups(df: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "officeId",
        "projectId",
        "SupplierName",
        "plantNumber",
        "mixNumber",
        cfg.REQUIRED_STRENGTH_COLUMN,
    ]
    optional_identity = []
    project_name = find_optional_column(
        df, ["ProjectName", "projectName", "projectDescription"]
    )
    office_name = find_optional_column(df, ["OfficeName", "officeName"])
    if office_name:
        optional_identity.append(office_name)
    if project_name:
        optional_identity.append(project_name)

    rows: list[dict[str, object]] = []
    for key, group in df.groupby(group_columns, dropna=False, sort=False):
        values = dict(zip(group_columns, key))
        strength = group[cfg.STRENGTH_TARGET]
        margin = group["StrengthMargin_psi"]
        failures = int(group["StrengthFailure"].sum())
        test_count = int(group["testId"].nunique())
        strength_mean = safe_mean(strength)
        strength_std = safe_std(strength)

        row = {
            **values,
            "TestCount": test_count,
            "AverageActualStrength28_psi": strength_mean,
            "MedianActualStrength28_psi": safe_median(strength),
            "StrengthStdDev_psi": strength_std,
            "StrengthCV_percent": (
                100.0 * strength_std / strength_mean
                if pd.notna(strength_std) and strength_mean
                else np.nan
            ),
            "FailureCount": failures,
            "FailureRate_percent": 100.0 * failures / test_count if test_count else np.nan,
            "AverageSpecMargin_psi": safe_mean(margin),
            "MedianSpecMargin_psi": safe_median(margin),
            "P10SpecMargin_psi": safe_p10(margin),
            "MinSpecMargin_psi": float(margin.min()),
            "AverageCementContent_lbs_yd3": safe_mean(
                group["CalcCementContent_lbs_yd3"]
            ),
            "AverageFlyAshContent_lbs_yd3": safe_mean(
                group["FlyAshContent_lbs_yd3"]
            ),
            "AverageTotalCementitious_lbs_yd3": safe_mean(
                group["TotalCementitiousContent_lbs_yd3"]
            ),
            "AverageWCRatio": safe_mean(group["calcWCRatio"]),
        }
        if cfg.SLUMP_TARGET in group.columns:
            row["AverageSlump_in"] = safe_mean(group[cfg.SLUMP_TARGET])
            row["SlumpMeasurementCount"] = int(group[cfg.SLUMP_TARGET].notna().sum())
        if cfg.AIR_TARGET in group.columns:
            row["AverageAir_percent"] = safe_mean(group[cfg.AIR_TARGET])
            row["AirMeasurementCount"] = int(group[cfg.AIR_TARGET].notna().sum())
        for column in optional_identity:
            clean = group[column].dropna()
            row[column] = clean.iloc[0] if len(clean) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def add_comparative_metrics(evaluation: pd.DataFrame) -> pd.DataFrame:
    peer_columns = ["officeId", "projectId", cfg.REQUIRED_STRENGTH_COLUMN]
    evaluation["ProjectSpecMedianCementitious_lbs_yd3"] = evaluation.groupby(
        peer_columns, dropna=False
    )["AverageTotalCementitious_lbs_yd3"].transform("median")
    peer_median = evaluation["ProjectSpecMedianCementitious_lbs_yd3"]
    evaluation["CementitiousVsProjectMedian_percent"] = np.where(
        peer_median.notna() & peer_median.ne(0),
        100.0
        * (evaluation["AverageTotalCementitious_lbs_yd3"] - peer_median)
        / peer_median,
        np.nan,
    )
    return evaluation


def assign_review_flag(row: pd.Series) -> tuple[str, str, int]:
    failure_rate = row["FailureRate_percent"]
    p10_margin = row["P10SpecMargin_psi"]
    avg_margin = row["AverageSpecMargin_psi"]
    strength_cv = row["StrengthCV_percent"]
    cement_vs_median = row["CementitiousVsProjectMedian_percent"]

    if (
        failure_rate >= RISK_MIN_FAILURE_RATE_PERCENT
        or p10_margin < RISK_WHEN_P10_MARGIN_BELOW_PSI
    ):
        return (
            "Quality Risk",
            "Review failures, production consistency, water control, curing, and data quality; do not reduce cementitious content.",
            1,
        )

    positive_p10 = (
        p10_margin > 0 if OPTIMIZATION_REQUIRE_POSITIVE_P10_MARGIN else True
    )
    if (
        failure_rate <= OPTIMIZATION_MAX_FAILURE_RATE_PERCENT
        and avg_margin >= OPTIMIZATION_MIN_AVG_MARGIN_PSI
        and positive_p10
        and strength_cv <= OPTIMIZATION_MAX_STRENGTH_CV_PERCENT
    ):
        comparison = (
            " Cementitious content is above the project/spec peer median."
            if pd.notna(cement_vs_median) and cement_vs_median > 0
            else " Confirm comparable lower-cementitious benchmark mixes."
        )
        return (
            "Potential Optimization Review",
            "Consistent high strength margin with low failure risk; materials engineer should review for trial-batch reduction or SCM substitution."
            + comparison,
            2,
        )

    if (
        failure_rate <= BENCHMARK_MAX_FAILURE_RATE_PERCENT
        and BENCHMARK_MIN_AVG_MARGIN_PSI
        <= avg_margin
        <= BENCHMARK_MAX_AVG_MARGIN_PSI
        and p10_margin > 0
        and strength_cv <= BENCHMARK_MAX_STRENGTH_CV_PERCENT
        and (pd.isna(cement_vs_median) or cement_vs_median <= 0)
    ):
        return (
            "Stable Benchmark",
            "Stable compliance with moderate strength margin and peer-level or lower cementitious content; use as a comparison mix.",
            3,
        )

    return (
        "Monitor",
        "No automatic action; retain for historical comparison and periodic review.",
        4,
    )


def rank_evaluation(evaluation: pd.DataFrame) -> pd.DataFrame:
    flags = evaluation.apply(assign_review_flag, axis=1, result_type="expand")
    flags.columns = ["ReviewFlag", "RecommendedAction", "_Priority"]
    evaluation = pd.concat([evaluation, flags], axis=1)

    # Within each project, review quality risk first, then optimization candidates,
    # benchmarks, and monitored groups. Higher failure/margin/cement content ranks
    # first within its review class to make the ordering transparent.
    evaluation = evaluation.sort_values(
        [
            "officeId",
            "projectId",
            "_Priority",
            "FailureRate_percent",
            "AverageSpecMargin_psi",
            "AverageTotalCementitious_lbs_yd3",
        ],
        ascending=[True, True, True, False, False, False],
    ).reset_index(drop=True)
    evaluation["ProjectReviewRank"] = (
        evaluation.groupby(["officeId", "projectId"]).cumcount() + 1
    )
    evaluation = evaluation.drop(columns=["_Priority"])

    preferred_front = [
        "ProjectReviewRank",
        "ReviewFlag",
        "officeId",
        "projectId",
        "SupplierName",
        "plantNumber",
        "mixNumber",
        cfg.REQUIRED_STRENGTH_COLUMN,
        "TestCount",
        "AverageActualStrength28_psi",
        "StrengthStdDev_psi",
        "StrengthCV_percent",
        "FailureCount",
        "FailureRate_percent",
        "AverageSpecMargin_psi",
        "P10SpecMargin_psi",
        "AverageTotalCementitious_lbs_yd3",
        "CementitiousVsProjectMedian_percent",
        "AverageWCRatio",
        "RecommendedAction",
    ]
    ordered = [column for column in preferred_front if column in evaluation.columns]
    ordered += [column for column in evaluation.columns if column not in ordered]
    return evaluation[ordered]


def round_report_values(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in result.select_dtypes(include=["number"]).columns:
        if column in {
            "officeId",
            "projectId",
            "TestCount",
            "FailureCount",
            "ProjectReviewRank",
            "SlumpMeasurementCount",
            "AirMeasurementCount",
        }:
            continue
        result[column] = result[column].round(3)
    return result


def main() -> None:
    df = pd.read_csv(cfg.PREPARED_DATA_PATH, low_memory=False)
    analysis_rows, audit = prepare_analysis_rows(df)
    if analysis_rows.empty:
        raise ValueError("No eligible rows remain after filters and required-field checks.")

    project_summary = build_project_summary(analysis_rows)
    eligible_projects = project_summary[project_summary["EligibleProject"]][
        ["officeId", "projectId"]
    ]
    eligible_rows = analysis_rows.merge(
        eligible_projects, on=["officeId", "projectId"], how="inner"
    )
    audit["projects_before_minimum_test_filter"] = int(
        project_summary[["officeId", "projectId"]].drop_duplicates().shape[0]
    )
    audit["projects_meeting_minimum_test_count"] = int(len(eligible_projects))
    audit["rows_in_eligible_projects"] = int(len(eligible_rows))

    if eligible_rows.empty:
        raise ValueError(
            "No projects meet MIN_PROJECT_TEST_COUNT. Lower the editable threshold "
            "or change the office/project filters."
        )

    evaluation = aggregate_mix_groups(eligible_rows)
    audit["mix_groups_before_minimum_test_filter"] = int(len(evaluation))
    evaluation = evaluation[
        evaluation["TestCount"].ge(MIN_MIX_GROUP_TEST_COUNT)
    ].copy()
    audit["mix_groups_meeting_minimum_test_count"] = int(len(evaluation))

    if evaluation.empty:
        raise ValueError(
            "Projects passed, but no Project/Supplier/Plant/Mix/RequiredStrength "
            "group meets MIN_MIX_GROUP_TEST_COUNT. Lower that editable threshold."
        )

    evaluation = add_comparative_metrics(evaluation)
    evaluation = rank_evaluation(evaluation)
    evaluation = round_report_values(evaluation)

    review_candidates = evaluation[
        evaluation["ReviewFlag"].isin(
            ["Quality Risk", "Potential Optimization Review", "Stable Benchmark"]
        )
    ].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    project_summary.to_csv(OUTPUT_DIR / "project_eligibility_summary.csv", index=False)
    evaluation.to_csv(OUTPUT_DIR / "project_mix_performance_evaluation.csv", index=False)
    review_candidates.to_csv(OUTPUT_DIR / "materials_engineer_review_candidates.csv", index=False)

    audit.update(
        {
            "minimum_project_test_count": MIN_PROJECT_TEST_COUNT,
            "minimum_mix_group_test_count": MIN_MIX_GROUP_TEST_COUNT,
            "review_flag_counts": {
                str(key): int(value)
                for key, value in evaluation["ReviewFlag"].value_counts().items()
            },
            "important_note": (
                "Flags are historical screening rules. They do not approve mix "
                "changes, cement reduction, carbon claims, or specification compliance."
            ),
        }
    )
    with (OUTPUT_DIR / "historical_mix_analysis_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(audit, handle, indent=2)

    print("\nHistorical mix performance evaluation")
    display_columns = [
        "ProjectReviewRank",
        "ReviewFlag",
        "officeId",
        "projectId",
        "SupplierName",
        "plantNumber",
        "mixNumber",
        cfg.REQUIRED_STRENGTH_COLUMN,
        "TestCount",
        "AverageActualStrength28_psi",
        "FailureRate_percent",
        "AverageSpecMargin_psi",
        "P10SpecMargin_psi",
        "AverageTotalCementitious_lbs_yd3",
        "CementitiousVsProjectMedian_percent",
    ]
    print(evaluation[display_columns].head(MAX_ROWS_TO_PRINT).to_string(index=False))
    print(f"\nEvaluation rows: {len(evaluation):,}")
    print(f"Review candidates: {len(review_candidates):,}")
    print(f"Outputs: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
