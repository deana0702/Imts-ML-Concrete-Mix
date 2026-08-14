"""Diagnose why script 05 and 05B produce different BatchOnly R2 values.

This script does not change preprocessing. It reads the same prepared CSV from
mix_config.py and runs controlled experiments that separately measure:

1. Complete-case cohort selection effect
2. Re-splitting-after-filtering effect
3. Adding actual slump and air effect

No command-line arguments are used. Edit paths/settings in mix_config.py.
Run:
    python 05c_diagnose_r2_difference.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

import mix_config as cfg


FIELD_FEATURES = ["EffectiveSlump_in", "EffectiveAir_percent"]
OUTPUT_DIR = cfg.MODEL_OUTPUT_DIR / "r2_difference_diagnostic"


def candidate_models() -> dict[str, object]:
    """Exactly match the model settings used by scripts 05 and 05B."""
    return {
        "DummyMedian": DummyRegressor(strategy="median"),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=cfg.RANDOM_STATE,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            max_features=0.8,
            n_jobs=cfg.N_JOBS,
            random_state=cfg.RANDOM_STATE,
        ),
    }


def split_like_05(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce script 05: split all prepared rows before target filtering."""
    groups = df[cfg.GROUP_COLUMN].fillna(df["testId"].astype(str))
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE,
    )
    train_index, test_index = next(splitter.split(df, groups=groups))
    return df.iloc[train_index].copy(), df.iloc[test_index].copy()


def split_like_05b(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce script 05B: split the already filtered complete-case cohort."""
    groups = df[cfg.GROUP_COLUMN].astype(str)
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE,
    )
    train_index, test_index = next(splitter.split(df, groups=groups))
    return df.iloc[train_index].copy(), df.iloc[test_index].copy()


def strength_eligible(df: pd.DataFrame) -> pd.DataFrame:
    return df[df[cfg.STRENGTH_TARGET].notna()].copy()


def field_complete(df: pd.DataFrame) -> pd.DataFrame:
    """Use exactly the complete-row conditions applied in script 05B."""
    mask = (
        df[cfg.STRENGTH_TARGET].notna()
        & df[cfg.GROUP_COLUMN].notna()
        & df[FIELD_FEATURES].notna().all(axis=1)
        & df[cfg.CORE_BATCH_FEATURES].notna().all(axis=1)
    )
    return df[mask].copy()


def target_summary(y: pd.Series) -> dict[str, float]:
    return {
        "target_mean_psi": float(y.mean()),
        "target_std_psi": float(y.std(ddof=0)),
        "target_min_psi": float(y.min()),
        "target_p01_psi": float(y.quantile(0.01)),
        "target_p50_psi": float(y.quantile(0.50)),
        "target_p99_psi": float(y.quantile(0.99)),
        "target_max_psi": float(y.max()),
    }


def train_experiment(
    experiment: str,
    explanation: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
) -> list[dict[str, object]]:
    if len(train_df) < 100 or len(test_df) < 30:
        raise ValueError(
            f"{experiment} has insufficient rows: "
            f"train={len(train_df)}, test={len(test_df)}"
        )

    x_train = train_df[features].apply(pd.to_numeric, errors="coerce")
    x_test = test_df[features].apply(pd.to_numeric, errors="coerce")
    y_train = train_df[cfg.STRENGTH_TARGET]
    y_test = test_df[cfg.STRENGTH_TARGET]
    summary = target_summary(y_test)
    rows: list[dict[str, object]] = []

    for model_name, estimator in candidate_models().items():
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        prediction = pipeline.predict(x_test)
        residual = prediction - y_test.to_numpy()
        absolute_error = np.abs(residual)

        rows.append(
            {
                "experiment": experiment,
                "explanation": explanation,
                "model": model_name,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_projects": int(train_df[cfg.GROUP_COLUMN].nunique()),
                "test_projects": int(test_df[cfg.GROUP_COLUMN].nunique()),
                "feature_count": len(features),
                "features": " | ".join(features),
                "MAE_psi": float(mean_absolute_error(y_test, prediction)),
                "MedianAE_psi": float(median_absolute_error(y_test, prediction)),
                "RMSE_psi": float(mean_squared_error(y_test, prediction) ** 0.5),
                "R2": float(r2_score(y_test, prediction)),
                "MeanBias_psi": float(np.mean(residual)),
                "P90AbsoluteError_psi": float(np.percentile(absolute_error, 90)),
                **summary,
            }
        )
    return rows


def select_result(
    results: pd.DataFrame, experiment: str, model: str
) -> pd.Series:
    match = results[
        (results["experiment"] == experiment) & (results["model"] == model)
    ]
    if len(match) != 1:
        raise RuntimeError(f"Expected one result for {experiment}/{model}.")
    return match.iloc[0]


def build_effect_table(results: pd.DataFrame) -> pd.DataFrame:
    """Calculate controlled R2 changes for each non-dummy model."""
    comparisons = [
        (
            "Complete-case cohort effect under original 05 split",
            "A_05_original_all_strength_rows",
            "B_05_split_complete_rows_batch_only",
        ),
        (
            "Actual slump/air effect under original 05 split",
            "B_05_split_complete_rows_batch_only",
            "C_05_split_complete_rows_plus_slump_air",
        ),
        (
            "Re-splitting effect after complete-case filtering",
            "B_05_split_complete_rows_batch_only",
            "D_05B_resplit_complete_rows_batch_only",
        ),
        (
            "Actual slump/air effect under 05B split",
            "D_05B_resplit_complete_rows_batch_only",
            "E_05B_resplit_complete_rows_plus_slump_air",
        ),
        (
            "Total observed difference: 05 original to 05B BatchOnly",
            "A_05_original_all_strength_rows",
            "D_05B_resplit_complete_rows_batch_only",
        ),
    ]

    rows: list[dict[str, object]] = []
    for model in ["HistGradientBoosting", "ExtraTrees"]:
        for effect, before_name, after_name in comparisons:
            before = select_result(results, before_name, model)
            after = select_result(results, after_name, model)
            rows.append(
                {
                    "model": model,
                    "effect": effect,
                    "before_experiment": before_name,
                    "after_experiment": after_name,
                    "before_rows": int(before["train_rows"] + before["test_rows"]),
                    "after_rows": int(after["train_rows"] + after["test_rows"]),
                    "before_R2": before["R2"],
                    "after_R2": after["R2"],
                    "R2_change": after["R2"] - before["R2"],
                    "before_MAE_psi": before["MAE_psi"],
                    "after_MAE_psi": after["MAE_psi"],
                    "MAE_improvement_psi": before["MAE_psi"] - after["MAE_psi"],
                    "before_test_target_std_psi": before["target_std_psi"],
                    "after_test_target_std_psi": after["target_std_psi"],
                }
            )
    return pd.DataFrame(rows)


def save_project_membership(
    path: Path,
    split_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    frames = []
    for partition, part_df in [("train", train_df), ("test", test_df)]:
        project_counts = (
            part_df.dropna(subset=[cfg.GROUP_COLUMN])
            .groupby(cfg.GROUP_COLUMN, dropna=False)
            .size()
            .rename("row_count")
            .reset_index()
        )
        project_counts.insert(0, "partition", partition)
        project_counts.insert(0, "split_name", split_name)
        frames.append(project_counts)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)


def main() -> None:
    df = pd.read_csv(cfg.PREPARED_DATA_PATH, low_memory=False)
    batch_features = [column for column in cfg.MODEL_FEATURES if column in df.columns]
    required = [
        "testId",
        cfg.GROUP_COLUMN,
        cfg.STRENGTH_TARGET,
        *cfg.CORE_BATCH_FEATURES,
        *FIELD_FEATURES,
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("Prepared data is missing: " + ", ".join(missing))

    # Split exactly as script 05 does, before target/complete-case filtering.
    original_train_all, original_test_all = split_like_05(df)
    original_train_strength = strength_eligible(original_train_all)
    original_test_strength = strength_eligible(original_test_all)

    # Keep the original 05 project assignment, then select complete field rows.
    original_train_complete = field_complete(original_train_all)
    original_test_complete = field_complete(original_test_all)

    # Reproduce 05B: select complete rows first, then create a new project split.
    complete_all = field_complete(df)
    resplit_train_complete, resplit_test_complete = split_like_05b(complete_all)

    experiment_definitions = [
        (
            "A_05_original_all_strength_rows",
            "Exact strength experiment from script 05.",
            original_train_strength,
            original_test_strength,
            batch_features,
        ),
        (
            "B_05_split_complete_rows_batch_only",
            "Keep the original 05 split; restrict both partitions to 05B complete rows.",
            original_train_complete,
            original_test_complete,
            batch_features,
        ),
        (
            "C_05_split_complete_rows_plus_slump_air",
            "Same rows and split as B; add actual slump and air.",
            original_train_complete,
            original_test_complete,
            batch_features + FIELD_FEATURES,
        ),
        (
            "D_05B_resplit_complete_rows_batch_only",
            "Exact 05B cohort/split using BatchOnly features.",
            resplit_train_complete,
            resplit_test_complete,
            batch_features,
        ),
        (
            "E_05B_resplit_complete_rows_plus_slump_air",
            "Same rows and split as D; add actual slump and air.",
            resplit_train_complete,
            resplit_test_complete,
            batch_features + FIELD_FEATURES,
        ),
    ]

    all_results: list[dict[str, object]] = []
    for definition in experiment_definitions:
        all_results.extend(train_experiment(*definition))

    results = pd.DataFrame(all_results)
    effects = build_effect_table(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "r2_diagnostic_all_experiments.csv", index=False)
    effects.to_csv(OUTPUT_DIR / "r2_difference_decomposition.csv", index=False)
    save_project_membership(
        OUTPUT_DIR / "05_original_split_projects.csv",
        "05_original_split",
        original_train_strength,
        original_test_strength,
    )
    save_project_membership(
        OUTPUT_DIR / "05b_resplit_projects.csv",
        "05b_resplit",
        resplit_train_complete,
        resplit_test_complete,
    )

    summary = {
        "prepared_input": str(cfg.PREPARED_DATA_PATH),
        "random_state": cfg.RANDOM_STATE,
        "test_size": cfg.TEST_SIZE,
        "group_column": cfg.GROUP_COLUMN,
        "batch_features": batch_features,
        "field_features": FIELD_FEATURES,
        "interpretation_order": [
            "A to B measures complete-case cohort effect with original split retained.",
            "B to C measures slump/air feature effect under original split.",
            "B to D measures re-splitting effect on the complete-case cohort.",
            "D to E measures slump/air feature effect under the 05B split.",
            "A to D is the total 05-versus-05B BatchOnly difference.",
        ],
    }
    with (OUTPUT_DIR / "diagnostic_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    display_columns = [
        "experiment",
        "model",
        "train_rows",
        "test_rows",
        "test_projects",
        "MAE_psi",
        "RMSE_psi",
        "R2",
        "target_std_psi",
        "target_min_psi",
        "target_max_psi",
    ]
    print("\nR2 diagnostic experiments")
    print(results[display_columns].to_string(index=False))
    print("\nR2 difference decomposition")
    print(effects.to_string(index=False))
    print(f"\nOutputs: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
