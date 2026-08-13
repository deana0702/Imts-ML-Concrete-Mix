"""Train and compare 28-day concrete failure-risk classifiers.

FailureFlag28=1 is the positive class. Edit config.py, then run:
    python 03_train_classification_models.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

import config


REQUIRED_CONFIG_NAMES = [
    "CLASSIFICATION_INPUT_PATH",
    "CLASSIFICATION_OUTPUT_DIR",
    "TEST_SIZE",
    "VALIDATION_SIZE_WITHIN_TRAIN",
    "RANDOM_STATE",
    "N_JOBS",
    "TRAIN_FEATURE_SET_NAMES",
    "MIN_FAILURE_RECALL",
    "FEATURE_SETS",
    "GROUP_COLUMN",
    "CLASSIFICATION_TARGET",
]


def validate_config() -> None:
    missing = [name for name in REQUIRED_CONFIG_NAMES if not hasattr(config, name)]
    if missing:
        raise RuntimeError(
            "Your config.py is older than this classification script. Missing: "
            + ", ".join(missing)
            + ". Replace config.py with the version delivered with this script."
        )
    for name in ["TEST_SIZE", "VALIDATION_SIZE_WITHIN_TRAIN", "MIN_FAILURE_RECALL"]:
        if not 0 < getattr(config, name) < 1:
            raise ValueError(f"{name} must be between 0 and 1.")
    unknown = [n for n in config.TRAIN_FEATURE_SET_NAMES if n not in config.FEATURE_SETS]
    if unknown:
        raise ValueError("Unknown TRAIN_FEATURE_SET_NAMES: " + ", ".join(unknown))


def read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists() and path.suffix.lower() == ".csv":
        parquet_path = path.with_suffix(".parquet")
        if parquet_path.exists():
            path = parquet_path
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("CLASSIFICATION_INPUT_PATH must be a CSV or Parquet file.")


def group_split(
    df: pd.DataFrame, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = df[config.GROUP_COLUMN].fillna(df["testId"].astype(str))
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    left_index, right_index = next(splitter.split(df, groups=groups))
    return df.iloc[left_index].copy(), df.iloc[right_index].copy()


def available_features(df: pd.DataFrame, feature_set_name: str) -> list[str]:
    features = [c for c in config.FEATURE_SETS[feature_set_name] if c in df.columns]
    if not features:
        raise ValueError(f"No usable columns found for {feature_set_name}.")
    return features


def model_candidates() -> dict[str, Pipeline]:
    return {
        "LogisticRegression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=config.RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.06,
                        max_iter=300,
                        max_leaf_nodes=31,
                        l2_regularization=1.0,
                        random_state=config.RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=3,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=config.N_JOBS,
                        random_state=config.RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def choose_threshold(y_true: pd.Series, probability: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    candidates = pd.DataFrame(
        {"threshold": thresholds, "precision": precision[:-1], "recall": recall[:-1]}
    )
    eligible = candidates[candidates["recall"] >= config.MIN_FAILURE_RECALL]
    if eligible.empty:
        return float(candidates.sort_values("recall", ascending=False).iloc[0]["threshold"])
    return float(eligible.sort_values(["precision", "threshold"], ascending=False).iloc[0]["threshold"])


def evaluate(y_true: pd.Series, probability: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "ROC_AUC": roc_auc_score(y_true, probability),
        "PR_AUC": average_precision_score(y_true, probability),
        "failure_recall": recall_score(y_true, predicted, zero_division=0),
        "failure_precision": precision_score(y_true, predicted, zero_division=0),
        "failure_f1": f1_score(y_true, predicted, zero_division=0),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def main() -> None:
    validate_config()
    output_dir = config.CLASSIFICATION_OUTPUT_DIR
    model_dir = output_dir / "saved_models"
    prediction_dir = output_dir / "test_predictions"
    model_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    df = read_dataset(config.CLASSIFICATION_INPUT_PATH)
    required = [config.CLASSIFICATION_TARGET, config.GROUP_COLUMN, "testId"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    df = df[df[config.CLASSIFICATION_TARGET].isin([0, 1])].copy()
    train_validation_df, test_df = group_split(df, config.TEST_SIZE, config.RANDOM_STATE)
    train_df, validation_df = group_split(
        train_validation_df,
        config.VALIDATION_SIZE_WITHIN_TRAIN,
        config.RANDOM_STATE + 1,
    )

    result_rows: list[dict[str, object]] = []
    for feature_set_name in config.TRAIN_FEATURE_SET_NAMES:
        features = available_features(df, feature_set_name)
        feature_train_df = train_df
        feature_validation_df = validation_df
        feature_test_df = test_df
        if feature_set_name.startswith("Day7"):
            feature_train_df = train_df[
                train_df["AverageActualStrength7_psi"].notna()
            ].copy()
            feature_validation_df = validation_df[
                validation_df["AverageActualStrength7_psi"].notna()
            ].copy()
            feature_test_df = test_df[
                test_df["AverageActualStrength7_psi"].notna()
            ].copy()
        x_train = feature_train_df[features].apply(pd.to_numeric, errors="coerce")
        x_validation = feature_validation_df[features].apply(pd.to_numeric, errors="coerce")
        x_test = feature_test_df[features].apply(pd.to_numeric, errors="coerce")
        y_train = feature_train_df[config.CLASSIFICATION_TARGET].astype(int)
        y_validation = feature_validation_df[config.CLASSIFICATION_TARGET].astype(int)
        y_test = feature_test_df[config.CLASSIFICATION_TARGET].astype(int)
        for split_name, target in {
            "train": y_train,
            "validation": y_validation,
            "test": y_test,
        }.items():
            if target.nunique() < 2:
                raise ValueError(
                    f"{feature_set_name} {split_name} split does not contain both "
                    "pass and failure rows. Change RANDOM_STATE or split sizes."
                )

        for model_name, pipeline in model_candidates().items():
            fit_parameters = {}
            if model_name == "HistGradientBoosting":
                fit_parameters["model__sample_weight"] = compute_sample_weight(
                    class_weight="balanced", y=y_train
                )
            pipeline.fit(x_train, y_train, **fit_parameters)
            validation_probability = pipeline.predict_proba(x_validation)[:, 1]
            threshold = choose_threshold(y_validation, validation_probability)
            test_probability = pipeline.predict_proba(x_test)[:, 1]
            row = {
                "feature_set": feature_set_name,
                "model": model_name,
                "train_rows": len(feature_train_df),
                "validation_rows": len(feature_validation_df),
                "test_rows": len(feature_test_df),
                "test_failure_rate": float(y_test.mean()),
                "feature_count": len(features),
                **evaluate(y_test, test_probability, threshold),
            }
            result_rows.append(row)

            artifact = {
                "pipeline": pipeline,
                "features": features,
                "feature_set": feature_set_name,
                "target": config.CLASSIFICATION_TARGET,
                "failure_probability_threshold": threshold,
                "positive_class": 1,
            }
            joblib.dump(artifact, model_dir / f"{feature_set_name}__{model_name}.joblib")
            predicted = (test_probability >= threshold).astype(int)
            pd.DataFrame(
                {
                    "testId": feature_test_df["testId"].to_numpy(),
                    "projectId": feature_test_df[config.GROUP_COLUMN].to_numpy(),
                    "actual_failure": y_test.to_numpy(),
                    "failure_probability": test_probability,
                    "predicted_failure": predicted,
                    "threshold": threshold,
                }
            ).to_csv(
                prediction_dir / f"{feature_set_name}__{model_name}.csv", index=False
            )

    results = pd.DataFrame(result_rows).sort_values(
        ["failure_recall", "failure_precision", "PR_AUC"], ascending=False
    )
    results.to_csv(output_dir / "classification_model_comparison.csv", index=False)
    summary = {
        "input_rows": len(df),
        "train_rows": len(train_df),
        "validation_rows": len(validation_df),
        "test_rows": len(test_df),
        "overall_failure_rate": float(df[config.CLASSIFICATION_TARGET].mean()),
        "threshold_selection_rule": f"maximize precision with validation recall >= {config.MIN_FAILURE_RECALL}",
        "best_model_by_recall_then_precision": results.iloc[0].to_dict(),
    }
    with (output_dir / "classification_run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("Classification training completed. FailureFlag28=1 is positive.")
    print(results.to_string(index=False))
    print(f"Outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
