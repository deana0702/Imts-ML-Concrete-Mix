import matplotlib.pyplot as plt 
import pandas as pd
from pathlib import Path

BASEPATH = Path("data/prepared_28_day_standard_cure")
DATA_FILE = BASEPATH / "results_baseline_models" / "strength_prediction_test_results.csv"
OUTPUT_PATH = BASEPATH / "results_baseline_models"


prediction_results = pd.read_csv(DATA_FILE)

plt.figure(figsize=(8, 8)) 

plt.scatter( 

    prediction_results["ActualStrength28_psi"], 

    prediction_results["PredictedStrength28_psi"], 

    alpha=0.3, 

) 

  

minimum = min( 

    prediction_results["ActualStrength28_psi"].min(), 

    prediction_results["PredictedStrength28_psi"].min(), 

) 

maximum = max( 

    prediction_results["ActualStrength28_psi"].max(), 

    prediction_results["PredictedStrength28_psi"].max(), 

) 

  

plt.plot([minimum, maximum], [minimum, maximum]) 

  

plt.xlabel("Actual 28-Day Strength (psi)") 

plt.ylabel("Predicted 28-Day Strength (psi)") 

plt.title("Actual vs. Predicted Strength — Random Forest") 

plt.tight_layout() 

#plt.savefig(OUTPUT_PATH / "actual_vs_predicted_strength.png")
plt.show()

prediction_results["Residual_psi"] = ( 

    prediction_results["ActualStrength28_psi"] 

    - prediction_results["PredictedStrength28_psi"] 

) 

  

plt.figure(figsize=(10, 6)) 

plt.hist( 

    prediction_results["Residual_psi"], 

    bins=40, 

) 

plt.axvline(0, linestyle="--") 

plt.xlabel("Residual: Actual - Predicted (psi)") 

plt.ylabel("Number of Tests") 

plt.title("Random Forest Residual Distribution") 

plt.tight_layout() 

plt.show() 



prediction_results["StrengthRange"] = pd.cut( 

    prediction_results["ActualStrength28_psi"], 

    bins=[0, 3000, 4000, 5000, 6000, 7000, 9000, float("inf")], 

) 

  

error_by_range = ( 

    prediction_results 

    .groupby("StrengthRange", observed=True) 

    .agg( 

        TestCount=("AbsoluteError_psi", "size"), 

        MAE=("AbsoluteError_psi", "mean"), 

        MedianError=("AbsoluteError_psi", "median"), 

        ActualMean=("ActualStrength28_psi", "mean"), 

        PredictedMean=("PredictedStrength28_psi", "mean"), 

    ) 

) 

  

print(error_by_range) 


largest_errors = prediction_results.sort_values( 

    "AbsoluteError_psi", 

    ascending=False, 

).head(50) 

  

largest_errors.to_csv( 

    "largest_strength_prediction_errors.csv", 

    index=False, 

) 

  

print( 

    largest_errors[ 

        [ 

            "ActualStrength28_psi", 

            "PredictedStrength28_psi", 

            "AbsoluteError_psi", 

        ] 

    ].head(20) 

) 

