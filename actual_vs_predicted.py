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

plt.savefig(OUTPUT_PATH / "actual_vs_predicted_strength.png")