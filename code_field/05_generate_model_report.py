from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Run with:
#     python 05_generate_model_report.py
ROOT_DIR = Path(__file__).resolve().parent.parent
HOLDOUT_METRICS_FILE = (
    ROOT_DIR / "data/field_core_outputs/field_core_models" / "holdout_model_metrics.csv"
)
HOLDOUT_PREDICTIONS_FILE = (
    ROOT_DIR / "data/field_core_outputs/field_core_models" / "holdout_predictions.csv"
)
CV_SUMMARY_FILE = ROOT_DIR / "data/field_core_outputs/field_core_cv" / "grouped_cv_summary.csv"
SPLIT_SUMMARY_FILE = ROOT_DIR / "data/field_core_outputs/field_core_splits" / "split_summary.csv"
CLEANING_SUMMARY_FILE = (
    ROOT_DIR / "data/field_core_outputs/field_core_clean" / "cleaning_run_summary.json"
)
OUTPUT_DIR = ROOT_DIR / "data/field_core_outputs/field_core_report"


def require_file(path: Path, prior_step: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run {prior_step} first.")


def save_bar_plot(
    frame: pd.DataFrame,
    value_column: str,
    title: str,
    x_label: str,
    output_path: Path,
) -> None:
    labels = frame["FeatureSet"].astype(str) + " / " + frame["Model"].astype(str)
    values = frame[value_column]

    plt.figure(figsize=(10, max(5, len(frame) * 0.55)))
    plt.barh(labels, values)
    plt.xlabel(x_label)
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    require_file(HOLDOUT_METRICS_FILE, "03_train_holdout_models.py")
    require_file(HOLDOUT_PREDICTIONS_FILE, "03_train_holdout_models.py")
    require_file(CV_SUMMARY_FILE, "04_run_grouped_cross_validation.py")
    require_file(SPLIT_SUMMARY_FILE, "02_create_grouped_splits.py")

    holdout = pd.read_csv(HOLDOUT_METRICS_FILE).sort_values("MAE")
    predictions = pd.read_csv(HOLDOUT_PREDICTIONS_FILE, low_memory=False)
    cv = pd.read_csv(CV_SUMMARY_FILE).sort_values("MAE_mean")
    split = pd.read_csv(SPLIT_SUMMARY_FILE)

    save_bar_plot(
        holdout,
        "MAE",
        "Holdout MAE by Model and Feature Set",
        "MAE (psi; lower is better)",
        OUTPUT_DIR / "holdout_mae.png",
    )
    save_bar_plot(
        holdout,
        "RMSE",
        "Holdout RMSE by Model and Feature Set",
        "RMSE (psi; lower is better)",
        OUTPUT_DIR / "holdout_rmse.png",
    )

    cv_labels = cv["FeatureSet"].astype(str) + " / " + cv["Model"].astype(str)
    plt.figure(figsize=(10, max(5, len(cv) * 0.6)))
    plt.errorbar(
        cv["MAE_mean"],
        cv_labels,
        xerr=cv["MAE_std"].fillna(0),
        fmt="o",
        capsize=4,
    )
    plt.xlabel("Grouped CV MAE, mean ± standard deviation (psi)")
    plt.title("Grouped Cross-Validation Stability")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "grouped_cv_mae.png", dpi=160)
    plt.close()

    best = holdout.iloc[0]
    best_predictions = predictions.loc[
        predictions["FeatureSet"].eq(best["FeatureSet"])
        & predictions["Model"].eq(best["Model"])
    ].copy()

    plt.figure(figsize=(7, 7))
    plt.scatter(
        best_predictions["ActualStrength28_psi"],
        best_predictions["PredictedStrength28_psi"],
        alpha=0.25,
        s=12,
    )
    minimum = min(
        best_predictions["ActualStrength28_psi"].min(),
        best_predictions["PredictedStrength28_psi"].min(),
    )
    maximum = max(
        best_predictions["ActualStrength28_psi"].max(),
        best_predictions["PredictedStrength28_psi"].max(),
    )
    plt.plot([minimum, maximum], [minimum, maximum], linestyle="--")
    plt.xlabel("Actual 28-day strength (psi)")
    plt.ylabel("Predicted 28-day strength (psi)")
    plt.title(f"Best Holdout Model: {best['FeatureSet']} / {best['Model']}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "best_predicted_vs_actual.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.hist(best_predictions["ResidualPsi"].dropna(), bins=60)
    plt.xlabel("Residual = predicted - actual (psi)")
    plt.ylabel("Test count")
    plt.title("Best Model Residual Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "best_residual_distribution.png", dpi=160)
    plt.close()

    field_only_best = holdout.loc[holdout["FeatureSet"].eq("FieldOnly")].iloc[0]
    required_best = holdout.loc[
        holdout["FeatureSet"].eq("FieldPlusRequired")
    ].iloc[0]
    mae_improvement = field_only_best["MAE"] - required_best["MAE"]
    mae_improvement_percent = (
        mae_improvement / field_only_best["MAE"] * 100.0
        if field_only_best["MAE"] != 0
        else np.nan
    )

    best_cv = cv.iloc[0]
    report = f"""# IMTS Field Core 28-Day Strength Model Report

## Best grouped holdout result

- Feature set: **{best['FeatureSet']}**
- Model: **{best['Model']}**
- MAE: **{best['MAE']:,.1f} psi**
- Median absolute error: **{best['MedianAE']:,.1f} psi**
- RMSE: **{best['RMSE']:,.1f} psi**
- R²: **{best['R2']:.3f}**
- Mean bias: **{best['MeanBias']:,.1f} psi**
- Predictions within ±300 psi: **{best['Within300PsiPercent']:.1f}%**
- Predictions within ±500 psi: **{best['Within500PsiPercent']:.1f}%**

## Value of required strength

Best Field Only model: **{field_only_best['Model']}**, MAE **{field_only_best['MAE']:,.1f} psi**.  
Best Field + Required Strength model: **{required_best['Model']}**, MAE **{required_best['MAE']:,.1f} psi**.

Adding applicable 28-day required/design strength changed MAE by **{mae_improvement:,.1f} psi** (**{mae_improvement_percent:.1f}% improvement** when positive).

## Grouped cross-validation

- Best feature set: **{best_cv['FeatureSet']}**
- Best model: **{best_cv['Model']}**
- Mean CV MAE: **{best_cv['MAE_mean']:,.1f} psi**
- CV MAE standard deviation: **{best_cv['MAE_std']:,.1f} psi**
- Mean CV RMSE: **{best_cv['RMSE_mean']:,.1f} psi**
- Mean CV R²: **{best_cv['R2_mean']:.3f}**

## Interpretation

- The grouped holdout answers how the model performs on projects that were not used for training.
- The grouped cross-validation result is the stronger estimate of stability because it repeats that test across multiple project groups.
- Compare every trained model with `DummyMean`. A complex model is useful only when it improves materially over the dummy baseline.
- Field Only and Field + Required Strength were evaluated on the same rows and the same project split, so the comparison is fair.
- This is a feasibility model, not an engineering acceptance or rejection decision.

## Figures

- `holdout_mae.png`
- `holdout_rmse.png`
- `grouped_cv_mae.png`
- `best_predicted_vs_actual.png`
- `best_residual_distribution.png`
"""
    (OUTPUT_DIR / "field_core_model_report.md").write_text(report, encoding="utf-8")

    holdout.to_csv(OUTPUT_DIR / "holdout_metrics_sorted.csv", index=False)
    cv.to_csv(OUTPUT_DIR / "cv_summary_sorted.csv", index=False)
    split.to_csv(OUTPUT_DIR / "split_summary_copy.csv", index=False)

    print("Step 5 completed.")
    print(f"Report: {OUTPUT_DIR / 'field_core_model_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
