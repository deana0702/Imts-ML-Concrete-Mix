"""Project-group cross-validation for IMTS 28-day concrete strength models.

This script compares three controlled feature/cohort definitions:

1. AllStrength_BatchOnly
   Reproduces the script-05 strength cohort as closely as possible.
2. CompleteCase_BatchOnly
   Uses rows with strength, core batch, actual slump, and actual air, but the
   model receives only batch features.
3. CompleteCase_PlusSlumpAir
   Uses exactly the same rows/folds as #2 and adds actual slump and air.

All validation folds are grouped by projectId. No command-line arguments are
used. Edit settings below or paths in mix_config.py, then run:

    python 05d_project_group_cross_validation.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

import mix_config as cfg


# ---------------------------------------------------------------------------
# Editable cross-validation settings. No missing config constants are required.
# ---------------------------------------------------------------------------
CV_FOLDS = 5
FIELD_FEATURES = ["EffectiveSlump_in", "EffectiveAir_percent"]
OUTPUT_DIR = cfg.MODEL_OUTPUT_DIR / "project_group_cross_validation"


def candidate_models() -> dict[str, object]:
    """Match the model settings used in scripts 05 and 05B."""
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
        "Random Forest": Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="median", add_indicator=True),
                        ),
                        (
                            "model",
                            RandomForestRegressor(
                                n_estimators=500,
                                min_samples_leaf=5,
                                max_features=0.8,
                                random_state=42,
                                n_jobs=-1,
                            ),
                        ),
                    ]
                ),
    }


def add_cv_group(df: pd.DataFrame, allow_missing_project: bool) -> pd.DataFrame:
    """Create collision-safe group labels without changing source columns."""
    result = df.copy()
    project = result[cfg.GROUP_COLUMN]
    if allow_missing_project:
        result["_cv_group"] = np.where(
            project.notna(),
            "project::" + project.astype(str),
            "missing_project_test::" + result["testId"].astype(str),
        )
    else:
        result = result[project.notna()].copy()
        result["_cv_group"] = "project::" + result[cfg.GROUP_COLUMN].astype(str)
    return result


def build_cohorts(
    df: pd.DataFrame, batch_features: list[str]
) -> dict[str, tuple[pd.DataFrame, list[str], str]]:
    all_strength = df[df[cfg.STRENGTH_TARGET].notna()].copy()
    all_strength = add_cv_group(all_strength, allow_missing_project=True)

    complete_mask = (
        df[cfg.STRENGTH_TARGET].notna()
        & df[cfg.GROUP_COLUMN].notna()
        & df[FIELD_FEATURES].notna().all(axis=1)
        & df[cfg.CORE_BATCH_FEATURES].notna().all(axis=1)
    )
    complete = add_cv_group(df[complete_mask].copy(), allow_missing_project=False)

    return {
        "AllStrength_BatchOnly": (
            all_strength,
            batch_features,
            "Script-05-like strength cohort; batch features only.",
        ),
        "CompleteCase_BatchOnly": (
            complete,
            batch_features,
            "05B complete-case cohort; batch features only.",
        ),
        "CompleteCase_PlusSlumpAir": (
            complete.copy(),
            batch_features + FIELD_FEATURES,
            "Same rows/folds as CompleteCase_BatchOnly; adds actual slump and air.",
        ),
    }


def calculate_metrics(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    residual = prediction - y_true.to_numpy()
    absolute_error = np.abs(residual)
    return {
        "MAE_psi": float(mean_absolute_error(y_true, prediction)),
        "MedianAE_psi": float(median_absolute_error(y_true, prediction)),
        "RMSE_psi": float(mean_squared_error(y_true, prediction) ** 0.5),
        "R2": float(r2_score(y_true, prediction)),
        "MeanBias_psi": float(np.mean(residual)),
        "P90AbsoluteError_psi": float(np.percentile(absolute_error, 90)),
    }


def target_statistics(y: pd.Series) -> dict[str, float]:
    return {
        "target_mean_psi": float(y.mean()),
        "target_std_psi": float(y.std(ddof=0)),
        "target_min_psi": float(y.min()),
        "target_p01_psi": float(y.quantile(0.01)),
        "target_p50_psi": float(y.quantile(0.50)),
        "target_p99_psi": float(y.quantile(0.99)),
        "target_max_psi": float(y.max()),
    }


def validate_cohort(
    feature_set: str,
    explanation: str,
    df: pd.DataFrame,
    features: list[str],
) -> tuple[list[dict[str, object]], list[pd.DataFrame], list[pd.DataFrame]]:
    group_count = df["_cv_group"].nunique()
    if group_count < CV_FOLDS:
        raise ValueError(
            f"{feature_set} has only {group_count} groups; "
            f"CV_FOLDS={CV_FOLDS} cannot be used."
        )

    splitter = GroupKFold(n_splits=CV_FOLDS)
    groups = df["_cv_group"]
    fold_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    membership_frames: list[pd.DataFrame] = []

    for fold, (train_index, validation_index) in enumerate(
        splitter.split(df, groups=groups), start=1
    ):
        train_df = df.iloc[train_index].copy()
        validation_df = df.iloc[validation_index].copy()

        train_groups = set(train_df["_cv_group"])
        validation_groups = set(validation_df["_cv_group"])
        overlap = train_groups.intersection(validation_groups)
        if overlap:
            raise RuntimeError(
                f"Group leakage in {feature_set}, fold {fold}: {len(overlap)} groups."
            )

        x_train = train_df[features].apply(pd.to_numeric, errors="coerce")
        x_validation = validation_df[features].apply(pd.to_numeric, errors="coerce")
        y_train = train_df[cfg.STRENGTH_TARGET]
        y_validation = validation_df[cfg.STRENGTH_TARGET]

        fold_membership = (
            validation_df.groupby("_cv_group", dropna=False)
            .size()
            .rename("validation_rows")
            .reset_index()
        )
        fold_membership.insert(0, "fold", fold)
        fold_membership.insert(0, "feature_set", feature_set)
        membership_frames.append(fold_membership)

        for model_name, estimator in candidate_models().items():
            pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("model", estimator),
                ]
            )
            pipeline.fit(x_train, y_train)
            prediction = pipeline.predict(x_validation)

            fold_rows.append(
                {
                    "feature_set": feature_set,
                    "explanation": explanation,
                    "model": model_name,
                    "fold": fold,
                    "train_rows": len(train_df),
                    "validation_rows": len(validation_df),
                    "train_groups": len(train_groups),
                    "validation_groups": len(validation_groups),
                    "group_overlap": 0,
                    "feature_count": len(features),
                    **calculate_metrics(y_validation, prediction),
                    **target_statistics(y_validation),
                }
            )

            prediction_frames.append(
                pd.DataFrame(
                    {
                        "feature_set": feature_set,
                        "model": model_name,
                        "fold": fold,
                        "testId": validation_df["testId"].to_numpy(),
                        "projectId": validation_df[cfg.GROUP_COLUMN].to_numpy(),
                        "cv_group": validation_df["_cv_group"].to_numpy(),
                        "actual_strength_28_psi": y_validation.to_numpy(),
                        "predicted_strength_28_psi": prediction,
                        "residual_psi": prediction - y_validation.to_numpy(),
                        "absolute_error_psi": np.abs(
                            prediction - y_validation.to_numpy()
                        ),
                    }
                )
            )

    return fold_rows, prediction_frames, membership_frames


def summarize_folds(fold_results: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "MAE_psi",
        "MedianAE_psi",
        "RMSE_psi",
        "R2",
        "MeanBias_psi",
        "P90AbsoluteError_psi",
    ]
    rows: list[dict[str, object]] = []

    for (feature_set, model), group in fold_results.groupby(
        ["feature_set", "model"], sort=False
    ):
        row: dict[str, object] = {
            "feature_set": feature_set,
            "model": model,
            "folds": len(group),
            "total_rows_in_cohort": int(group["validation_rows"].sum()),
            "total_validation_groups": int(group["validation_groups"].sum()),
        }
        for metric in metric_columns:
            row[f"Mean_{metric}"] = float(group[metric].mean())
            row[f"Std_{metric}"] = float(group[metric].std(ddof=1))
            row[f"Min_{metric}"] = float(group[metric].min())
            row[f"Max_{metric}"] = float(group[metric].max())
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["feature_set", "Mean_MAE_psi"])


def build_controlled_improvement(summary: pd.DataFrame) -> pd.DataFrame:
    """Compare BatchOnly versus Slump/Air on identical complete rows/folds."""
    rows: list[dict[str, object]] = []
    for model in summary["model"].unique():
        batch = summary[
            (summary["feature_set"] == "CompleteCase_BatchOnly")
            & (summary["model"] == model)
        ].iloc[0]
        field = summary[
            (summary["feature_set"] == "CompleteCase_PlusSlumpAir")
            & (summary["model"] == model)
        ].iloc[0]
        rows.append(
            {
                "model": model,
                "BatchOnly_Mean_R2": batch["Mean_R2"],
                "PlusSlumpAir_Mean_R2": field["Mean_R2"],
                "Mean_R2_change": field["Mean_R2"] - batch["Mean_R2"],
                "BatchOnly_Mean_MAE_psi": batch["Mean_MAE_psi"],
                "PlusSlumpAir_Mean_MAE_psi": field["Mean_MAE_psi"],
                "Mean_MAE_improvement_psi": (
                    batch["Mean_MAE_psi"] - field["Mean_MAE_psi"]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("PlusSlumpAir_Mean_MAE_psi")


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
        raise ValueError("Prepared data is missing required columns: " + ", ".join(missing))

    cohorts = build_cohorts(df, batch_features)
    all_fold_rows: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []
    all_memberships: list[pd.DataFrame] = []

    for feature_set, (cohort, features, explanation) in cohorts.items():
        print(
            f"Running {CV_FOLDS}-fold project CV: {feature_set} "
            f"({len(cohort):,} rows, {cohort['_cv_group'].nunique():,} groups)"
        )
        fold_rows, predictions, memberships = validate_cohort(
            feature_set, explanation, cohort, features
        )
        all_fold_rows.extend(fold_rows)
        all_predictions.extend(predictions)
        all_memberships.extend(memberships)

    fold_results = pd.DataFrame(all_fold_rows)
    summary = summarize_folds(fold_results)
    improvement = build_controlled_improvement(summary)
    predictions = pd.concat(all_predictions, ignore_index=True)
    memberships = pd.concat(all_memberships, ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(OUTPUT_DIR / "cv_fold_results.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "cv_model_summary.csv", index=False)
    improvement.to_csv(OUTPUT_DIR / "cv_slump_air_improvement.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "cv_out_of_fold_predictions.csv", index=False)
    memberships.to_csv(OUTPUT_DIR / "cv_fold_project_membership.csv", index=False)

    run_summary = {
        "prepared_input": str(cfg.PREPARED_DATA_PATH),
        "cv_method": "GroupKFold",
        "cv_folds": CV_FOLDS,
        "group_column": cfg.GROUP_COLUMN,
        "random_state_note": (
            "GroupKFold is deterministic and does not use RANDOM_STATE. "
            "Model estimators still use cfg.RANDOM_STATE."
        ),
        "batch_features": batch_features,
        "field_features": FIELD_FEATURES,
        "cohorts": {
            name: {
                "rows": len(cohort),
                "groups": int(cohort["_cv_group"].nunique()),
                "features": features,
                "explanation": explanation,
            }
            for name, (cohort, features, explanation) in cohorts.items()
        },
    }
    with (OUTPUT_DIR / "cv_run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2)

    display_columns = [
        "feature_set",
        "model",
        "folds",
        "total_rows_in_cohort",
        "Mean_MAE_psi",
        "Std_MAE_psi",
        "Mean_RMSE_psi",
        "Mean_R2",
        "Std_R2",
        "Min_R2",
        "Max_R2",
    ]
    print("\nCross-validation summary")
    print(summary[display_columns].to_string(index=False))
    print("\nControlled actual slump/air improvement")
    print(improvement.to_string(index=False))
    print(f"\nOutputs: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
