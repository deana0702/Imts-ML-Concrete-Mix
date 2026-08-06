from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# Run with:
#     python 02_create_grouped_splits.py
# Root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_BASE = ROOT_DIR / "data/field_core_outputs/field_core_clean/field_core_clean_base.csv"
INPUT_WITH_REQUIRED = ROOT_DIR / "data/field_core_outputs/field_core_clean/field_core_clean_with_required.csv"
OUTPUT_DIR = ROOT_DIR / "data/field_core_outputs/field_core_splits"

TARGET = "AverageActualStrength28_psi"
TEST_SIZE = 0.20
RANDOM_STATE = 42
GROUP_COLUMN_CANDIDATES = ["projectId", "projectNo"]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\nRun 01_clean_field_core_dataset.py first."
        )
    return pd.read_csv(path, low_memory=False)


def resolve_group_column(df: pd.DataFrame) -> str:
    for column in GROUP_COLUMN_CANDIDATES:
        if column in df.columns and df[column].notna().any():
            return column
    raise KeyError(
        f"No grouping column found. Expected one of {GROUP_COLUMN_CANDIDATES}."
    )


def make_groups(df: pd.DataFrame, group_column: str) -> pd.Series:
    groups = df[group_column].astype("string").str.strip()
    missing = groups.isna() | groups.eq("")

    # A missing project gets its own unique group. This prevents all missing
    # projects from being treated as one giant group.
    if "testId" in df.columns:
        fallback = "MISSING_PROJECT_TEST_" + df["testId"].astype("string")
    else:
        fallback = "MISSING_PROJECT_ROW_" + pd.Series(df.index, index=df.index).astype("string")
    return groups.mask(missing, fallback)


def target_summary(df: pd.DataFrame, split_name: str) -> dict[str, object]:
    target = pd.to_numeric(df[TARGET], errors="coerce")
    return {
        "Split": split_name,
        "RowCount": int(len(df)),
        "UniqueTests": int(df["testId"].nunique(dropna=True)) if "testId" in df.columns else len(df),
        "TargetMean": float(target.mean()),
        "TargetMedian": float(target.median()),
        "TargetStd": float(target.std()),
        "TargetMin": float(target.min()),
        "TargetMax": float(target.max()),
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 2 started: create reusable grouped train/test split")
    base = read_csv(INPUT_BASE)
    with_required = read_csv(INPUT_WITH_REQUIRED)
    print(f"with_required shape: {with_required.shape}")
    if "testId" not in base.columns or TARGET not in base.columns:
        raise KeyError("The clean base data must contain testId and the target column.")

    group_column = resolve_group_column(base)
    groups = make_groups(base, group_column)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_positions, test_positions = next(
        splitter.split(base, y=base[TARGET], groups=groups)
    )

    assignments = pd.DataFrame(
        {
            "testId": base["testId"].values,
            "SplitGroup": groups.values,
            "DatasetSplit": "train",
        }
    )
    assignments.loc[test_positions, "DatasetSplit"] = "test"

    base_with_split = base.merge(
        assignments[["testId", "SplitGroup", "DatasetSplit"]],
        on="testId",
        how="left",
        validate="one_to_one",
    )
    required_with_split = with_required.merge(
        assignments[["testId", "SplitGroup", "DatasetSplit"]],
        on="testId",
        how="left",
        validate="one_to_one",
    )

    if required_with_split["DatasetSplit"].isna().any():
        raise RuntimeError("Some required-strength rows did not receive a split assignment.")

    base_train = base_with_split.loc[base_with_split["DatasetSplit"].eq("train")].copy()
    base_test = base_with_split.loc[base_with_split["DatasetSplit"].eq("test")].copy()
    comparison_train = required_with_split.loc[
        required_with_split["DatasetSplit"].eq("train")
    ].copy()
    comparison_test = required_with_split.loc[
        required_with_split["DatasetSplit"].eq("test")
    ].copy()

    train_groups = set(base_train["SplitGroup"].astype(str))
    test_groups = set(base_test["SplitGroup"].astype(str))
    overlap = train_groups.intersection(test_groups)
    if overlap:
        raise RuntimeError(f"Group leakage detected: {len(overlap)} groups overlap.")

    assignments.to_csv(OUTPUT_DIR / "split_assignments.csv", index=False)
    base_train.to_csv(OUTPUT_DIR / "field_only_full_train.csv", index=False)
    base_test.to_csv(OUTPUT_DIR / "field_only_full_test.csv", index=False)
    comparison_train.to_csv(OUTPUT_DIR / "comparison_train.csv", index=False)
    comparison_test.to_csv(OUTPUT_DIR / "comparison_test.csv", index=False)

    summary_rows = [
        target_summary(base_train, "FieldOnlyFull_Train"),
        target_summary(base_test, "FieldOnlyFull_Test"),
        target_summary(comparison_train, "CommonComparison_Train"),
        target_summary(comparison_test, "CommonComparison_Test"),
    ]
    summary = pd.DataFrame(summary_rows)
    summary["UniqueGroups"] = [
        base_train["SplitGroup"].nunique(),
        base_test["SplitGroup"].nunique(),
        comparison_train["SplitGroup"].nunique(),
        comparison_test["SplitGroup"].nunique(),
    ]
    summary.to_csv(OUTPUT_DIR / "split_summary.csv", index=False)

    metadata = {
        "group_column": group_column,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "base_train_rows": len(base_train),
        "base_test_rows": len(base_test),
        "comparison_train_rows": len(comparison_train),
        "comparison_test_rows": len(comparison_test),
        "train_unique_groups": len(train_groups),
        "test_unique_groups": len(test_groups),
        "group_overlap_count": len(overlap),
    }
    (OUTPUT_DIR / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("Step 2 completed.")
    print(f"Grouping column: {group_column}")
    print(f"Common comparison train rows: {len(comparison_train):,}")
    print(f"Common comparison test rows: {len(comparison_test):,}")
    print(f"Overlapping groups: {len(overlap)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
