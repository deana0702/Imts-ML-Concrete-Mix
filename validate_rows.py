



import pandas as pd

df = pd.read_csv(
    "data\\prepared_28_day_standard_cure\\"
    "03_standard_cured_test_level_28_working_data.csv",
    low_memory=False,
)
print(f"Shape: {df.shape}")
# remove the rows if SpecifiedStrengthMissing is True,
df = df[df["SpecifiedStrengthMissing"] != True]
# Remove rows where CalcCementContent_lbs_yd3 is missing or 0
df = df[(df["CalcCementContent_lbs_yd3"].notna()) & (df["CalcCementContent_lbs_yd3"] != 0)]
# Remove rows where SandMoisture_percent is missing
df = df[(df["SandMoisture_percent"].notna())]
# Remove rows where AggregateSSD_lbs_yd3 is missing or 0
df = df[(df["AggregateSSD_lbs_yd3"].notna()) & (df["AggregateSSD_lbs_yd3"] != 0)]
# Remove rows where SandSSD_lbs_yd3 is missing or 0
df = df[(df["SandSSD_lbs_yd3"].notna()) & (df["SandSSD_lbs_yd3"] != 0)]
#  Remove rows where AverageActualAgeDays is missing or 0
df = df[(df["AverageActualAgeDays"].notna()) & (df["AverageActualAgeDays"] != 0)]
# Remove rows where calcWCRatio is missing or 0
df = df[(df["calcWCRatio"].notna()) & (df["calcWCRatio"] != 0)]
# Remove rows where FlyAshContent_lbs_yd3 is missing
df = df[(df["FlyAshContent_lbs_yd3"].notna())]


print(f"After dropping rows: {df.shape}")

df.to_csv(
    "data\\prepared_28_day_standard_cure\\"
    "03_standard_cured_test_level_28_working_data_drop_rows.csv",
    index=False,
)