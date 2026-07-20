



import pandas as pd

df = pd.read_csv(
    "data\\prepared_28_day_standard_cure\\"
    "03_standard_cured_test_level_28_working_data.csv",
    low_memory=False,
)
print(f"Shape: {df.shape}")
# remove the rows if SpecifiedStrengthMissing is True,
df = df[df["SpecifiedStrengthMissing"] != True]

print(f"After dropping rows: {df.shape}")

df.to_csv(
    "data\\prepared_28_day_standard_cure\\"
    "03_standard_cured_test_level_28_working_data_drop_rows.csv",
    index=False,
)