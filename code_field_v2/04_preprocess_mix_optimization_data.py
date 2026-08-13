"""Create a reliable batch-complete cohort for IMTS Mix Optimization.

Edit mix_config.py, then run: python 04_preprocess_mix_optimization_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import mix_config as cfg


def read_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("RAW_INPUT_PATH must be CSV or Parquet.")


def main() -> None:
    df = read_data(cfg.RAW_INPUT_PATH)
    required = ["testId", cfg.GROUP_COLUMN, cfg.STRENGTH_TARGET] + cfg.CORE_BATCH_FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("SQL extract is missing required batch columns: " + ", ".join(missing))

    input_rows = len(df)
    numeric = list(dict.fromkeys(
        cfg.MODEL_FEATURES
        + [cfg.STRENGTH_TARGET, cfg.SLUMP_TARGET, cfg.AIR_TARGET,
           cfg.REQUIRED_STRENGTH_COLUMN, "ConcreteTestUnitSystem", "IsValidCastDate"]
    ))
    for column in numeric:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    fly_ash_nulls_assumed_zero = 0
    if cfg.ASSUME_NULL_FLY_ASH_IS_ZERO_WHEN_CEMENT_EXISTS:
        assume_zero = (
            df["CalcCementContent_lbs_yd3"].notna()
            & df["FlyAshContent_lbs_yd3"].isna()
        )
        fly_ash_nulls_assumed_zero = int(assume_zero.sum())
        df.loc[assume_zero, "FlyAshContent_lbs_yd3"] = 0.0

    if cfg.FILTER_UNIT_SYSTEM and "ConcreteTestUnitSystem" in df.columns:
        df = df[df["ConcreteTestUnitSystem"].eq(cfg.UNIT_SYSTEM_TO_KEEP)].copy()
    if cfg.REQUIRE_VALID_CAST_DATE and "IsValidCastDate" in df.columns:
        df = df[df["IsValidCastDate"].eq(1)].copy()

    invalid_audit = []
    for column, (low, high) in cfg.VALID_RANGES.items():
        if column not in df.columns:
            continue
        invalid = df[column].notna() & ~df[column].between(low, high)
        invalid_audit.append({"column": column, "invalid_replaced": int(invalid.sum())})
        df.loc[invalid, column] = np.nan

    # Recompute consistent derived recipe features from IMTS fields.
    cement = df["CalcCementContent_lbs_yd3"]
    fly_ash = df["FlyAshContent_lbs_yd3"]
    df["TotalCementitiousContent_lbs_yd3"] = cement + fly_ash
    df["FlyAshFraction"] = fly_ash / df["TotalCementitiousContent_lbs_yd3"].replace(0, np.nan)
    df["TotalAggregateSSD_lbs_yd3"] = df["SandSSD_lbs_yd3"] + df["AggregateSSD_lbs_yd3"]
    df["SandFractionOfAggregate"] = (
        df["SandSSD_lbs_yd3"] / df["TotalAggregateSSD_lbs_yd3"].replace(0, np.nan)
    )

    # Material recipes are not imputed: incomplete recipes are excluded.
    complete_mask = df[cfg.CORE_BATCH_FEATURES].notna().all(axis=1)
    complete_mask &= df[cfg.STRENGTH_TARGET].notna()
    prepared = df.loc[complete_mask].drop_duplicates("testId", keep="first").copy()

    cfg.PREPARED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(cfg.PREPARED_DATA_PATH, index=False)
    pd.DataFrame(invalid_audit).to_csv(
        cfg.PREPARED_DATA_PATH.parent / "mix_invalid_range_audit.csv", index=False
    )
    summary = {
        "input_rows": input_rows,
        "batch_complete_strength_rows": len(prepared),
        "retained_percent": round(100 * len(prepared) / input_rows, 2) if input_rows else None,
        "projects": int(prepared[cfg.GROUP_COLUMN].nunique()),
        "suppliers": int(prepared["SupplierName"].nunique()) if "SupplierName" in prepared else None,
        "plants": int(prepared["plantNumber"].nunique()) if "plantNumber" in prepared else None,
        "mixes": int(prepared["mixNumber"].nunique()) if "mixNumber" in prepared else None,
        "material_recipe_imputed": False,
        "null_fly_ash_rows_assumed_zero": fly_ash_nulls_assumed_zero,
    }
    with (cfg.PREPARED_DATA_PATH.parent / "mix_preprocessing_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)
    print("Mix preprocessing completed.")
    print(json.dumps(summary, indent=2))
    print(f"Prepared data: {cfg.PREPARED_DATA_PATH.resolve()}")


if __name__ == "__main__":
    main()
