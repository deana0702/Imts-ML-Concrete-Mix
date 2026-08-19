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

models = [
    "Day7_FieldPlusRequired",
    "Full_ContextPlusDay7"
]

plt.figure(figsize=(8,6))

for model in models:
    subset = df[
        (df["Method"] == "DirectClassifier") &
        (df["FeatureSet"] == model)
    ]

    plt.plot(
        subset["Recall"],
        subset["Precision"],
        marker="o",
        linewidth=2,
        label=model
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/precision_vs_recall_day7.png")
plt.show()