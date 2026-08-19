import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load file
ROOT_DIR = Path(__file__).parent.parent
# if the plots directory does not exist, create it

INPUT_FILE_PATH = ROOT_DIR / "data" / "field_core_outputs" / "regression_vs_classifier_threshold_sweep_four_features"
OUTPUT_PATH = INPUT_FILE_PATH / "plots"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(f"{INPUT_FILE_PATH}/threshold_sweep_all.csv")

# Example: Plot DirectClassifier models only
df_plot = df[df["Method"] == "DirectClassifier"]

plt.figure(figsize=(8, 6))

for feature_set in df_plot["FeatureSet"].unique():
    subset = df_plot[df_plot["FeatureSet"] == feature_set]

    plt.plot(
        subset["Recall"],
        subset["Precision"],
        marker="o",
        linewidth=2,
        markersize=4,
        label=feature_set
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision vs Recall")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"{INPUT_FILE_PATH}/plots/precision_vs_recall.png")
plt.show()