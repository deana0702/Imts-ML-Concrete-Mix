"""Rank safe historical IMTS batch designs for a requested specification.

This searches observed recipes; it does not invent extrapolated recipes.
Edit OPT_* settings in mix_config.py, then run this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import mix_config as cfg


def load_model(name: str, required: bool = True):
    path = cfg.MODEL_OUTPUT_DIR / f"best_{name}_model.joblib"
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Run script 05 first. Missing: {path}")
        return None
    return joblib.load(path)


def apply_range(df: pd.DataFrame, column: str, low, high) -> pd.DataFrame:
    if low is not None:
        df = df[df[column].notna() & df[column].ge(low)]
    if high is not None:
        df = df[df[column].notna() & df[column].le(high)]
    return df


def main() -> None:
    df = pd.read_csv(cfg.PREPARED_DATA_PATH, low_memory=False)
    strength_model = load_model("strength28")
    slump_model = load_model("slump", required=False)
    air_model = load_model("air", required=False)
    if strength_model["test_r2"] < cfg.MIN_SURROGATE_R2_FOR_OPTIMIZATION:
        raise ValueError(
            f"Strength model held-out R2={strength_model['test_r2']:.3f} is too low "
            "for optimization. Batch data has not demonstrated enough predictive value."
        )

    if cfg.OPT_SUPPLIER_NAME is not None:
        df = df[df["SupplierName"].astype(str).eq(str(cfg.OPT_SUPPLIER_NAME))]
    if cfg.OPT_PLANT_NUMBER is not None:
        df = df[df["plantNumber"].astype(str).eq(str(cfg.OPT_PLANT_NUMBER))]
    if cfg.OPT_WC_RATIO_MAX is not None:
        df = df[df["calcWCRatio"].le(cfg.OPT_WC_RATIO_MAX)]
    if df.empty:
        raise ValueError("No historical recipes remain after supplier/plant/WC filters.")

    # Remove exact duplicate recipes so repeated testing does not dominate output.
    recipe_columns = strength_model["features"]
    candidate = df.drop_duplicates(recipe_columns).copy()
    x = candidate[recipe_columns].apply(pd.to_numeric, errors="coerce")
    candidate["PredictedStrength28_psi"] = strength_model["pipeline"].predict(x)
    candidate["ConservativeStrength28_psi"] = (
        candidate["PredictedStrength28_psi"] - strength_model["p90_absolute_error"]
    )
    required_strength = cfg.OPT_REQUIRED_STRENGTH_PSI + cfg.OPT_STRENGTH_SAFETY_MARGIN_PSI
    candidate = candidate[candidate["ConservativeStrength28_psi"].ge(required_strength)]

    if slump_model is not None:
        if slump_model["test_r2"] < cfg.MIN_SURROGATE_R2_FOR_OPTIMIZATION and (
            cfg.OPT_SLUMP_MIN_IN is not None or cfg.OPT_SLUMP_MAX_IN is not None
        ):
            raise ValueError(
                f"Slump model R2={slump_model['test_r2']:.3f} is too low to enforce "
                "a slump constraint. Set slump limits to None or improve the data."
            )
        sx = candidate[slump_model["features"]].apply(pd.to_numeric, errors="coerce")
        candidate["PredictedSlump_in"] = slump_model["pipeline"].predict(sx)
        candidate = apply_range(
            candidate, "PredictedSlump_in", cfg.OPT_SLUMP_MIN_IN, cfg.OPT_SLUMP_MAX_IN
        )
    elif cfg.OPT_SLUMP_MIN_IN is not None or cfg.OPT_SLUMP_MAX_IN is not None:
        raise ValueError("Slump constraint requested, but no slump model was trained.")

    if air_model is not None:
        if air_model["test_r2"] < cfg.MIN_SURROGATE_R2_FOR_OPTIMIZATION and (
            cfg.OPT_AIR_MIN_PERCENT is not None or cfg.OPT_AIR_MAX_PERCENT is not None
        ):
            raise ValueError(
                f"Air model R2={air_model['test_r2']:.3f} is too low to enforce "
                "an air constraint. Set air limits to None or improve the data."
            )
        ax = candidate[air_model["features"]].apply(pd.to_numeric, errors="coerce")
        candidate["PredictedAir_percent"] = air_model["pipeline"].predict(ax)
        candidate = apply_range(
            candidate, "PredictedAir_percent", cfg.OPT_AIR_MIN_PERCENT, cfg.OPT_AIR_MAX_PERCENT
        )
    elif cfg.OPT_AIR_MIN_PERCENT is not None or cfg.OPT_AIR_MAX_PERCENT is not None:
        raise ValueError("Air constraint requested, but no air model was trained.")

    if candidate.empty:
        raise ValueError(
            "No historical recipe meets the conservative constraints. Relax the safety "
            "margin/spec filters or collect more batch-complete data; do not extrapolate blindly."
        )
    objective = {
        "cement": "CalcCementContent_lbs_yd3",
        "total_cementitious": "TotalCementitiousContent_lbs_yd3",
    }.get(cfg.OPTIMIZATION_OBJECTIVE)
    if objective is None:
        raise ValueError("OPTIMIZATION_OBJECTIVE must be cement or total_cementitious.")
    candidate = candidate.sort_values(
        [objective, "ConservativeStrength28_psi"], ascending=[True, False]
    ).head(cfg.OPT_TOP_N)

    cfg.OPTIMIZATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = cfg.OPTIMIZATION_OUTPUT_DIR / "recommended_historical_mix_candidates.csv"
    keep = [c for c in cfg.ID_COLUMNS + recipe_columns + [
        "PredictedStrength28_psi", "ConservativeStrength28_psi",
        "PredictedSlump_in", "PredictedAir_percent",
        cfg.STRENGTH_TARGET, cfg.REQUIRED_STRENGTH_COLUMN
    ] if c in candidate.columns]
    candidate[keep].to_csv(output, index=False)
    summary = {
        "candidate_count_returned": len(candidate),
        "objective": objective,
        "required_strength_psi": cfg.OPT_REQUIRED_STRENGTH_PSI,
        "safety_margin_psi": cfg.OPT_STRENGTH_SAFETY_MARGIN_PSI,
        "strength_model_p90_absolute_error_deducted_psi": strength_model["p90_absolute_error"],
        "conservative_strength_is_heuristic_not_confidence_bound": True,
        "search_method": "observed historical IMTS recipes only",
        "true_carbon_optimization_performed": False,
    }
    with (cfg.OPTIMIZATION_OUTPUT_DIR / "optimization_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)
    print("Historical mix candidate search completed.")
    print(json.dumps(summary, indent=2))
    print(f"Candidates: {output.resolve()}")


if __name__ == "__main__":
    main()
