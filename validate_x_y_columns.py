import pandas as pd

df = pd.read_csv(
    "data/prepared_28_day_standard_cure/"
    "03_standard_cured_test_level_28_working_data.csv",
    low_memory=False,
)

suffix_columns = [
    column
    for column in df.columns
    if column.endswith("_x")
    or column.endswith("_y")
]

print(suffix_columns)