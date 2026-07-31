import pandas as pd

df = pd.read_csv(
    "data\\prepared_28_day_standard_cure\\"
    "03_standard_cured_test_level_28_working_data_drop_rows.csv",
    low_memory=False,
)
print(f"Initial shape: {df.shape}")
suffix_columns = [
    column
    for column in df.columns
    if column.endswith("_x")
    or column.endswith("_y")
]
print(suffix_columns)

# remove the columns from df : ApplicableStrengthType, ApplicableStrengthType, DesignStrengthRowCount, RequiredStrengthRowCount
# ActualStrengthCV28_percent, ActualStrengthStdDev28_psi, batchTime, sampleTime, MaximumActualAgeDays, MinimumActualAgeDays,
# MaximumActualStrength28_psi, MinimumActualStrength28_psi, MultipleStrengthRowsFlag, OfficeName, castDate, labNo, projectNo,
# SandBatchWeight_lbs, SandSSDWeight_lbs, SandWaterWeight_lbs, SandComponentCount, AggregateBatchWeight_lbs,
# AggregateSSDWeight_lbs, AggregateWaterWeight_lbs, AggregateComponentCount, CementQuantity_lbs, FlyAshQuantity_lbs,
# waterQuantity, WaterUnits, CalcYield_yd3, TotalCementitiousContent_lbs_yd3, FlyAshFraction,
# StandardCuredSpecimenCount28, UniqueStandardCuredSpecimenCount28
remove_columns = [
    "DesignStrength",
    "ApplicableSpecifiedStrength",
    "ApplicableStrengthType",
    "DesignStrengthRowCount",
    "RequiredStrengthRowCount",
    "ActualStrengthCV28_percent",
    "ActualStrengthStdDev28_psi",
    "MaximumActualAgeDays",
    "MinimumActualAgeDays",
    "MaximumActualStrength28_psi",
    "MinimumActualStrength28_psi",
    "MultipleStrengthRowsFlag",
    "OfficeName",
    "projectNo",
    "SandBatchWeight_lbs",
    "SandSSDWeight_lbs",
    "SandWaterWeight_lbs",
    "SandComponentCount",
    "AggregateBatchWeight_lbs",
    "AggregateSSDWeight_lbs",
    "AggregateWaterWeight_lbs",
    "AggregateComponentCount",
    "CementQuantity_lbs",
    "FlyAshQuantity_lbs",
    "waterQuantity",
    "WaterUnits",
    "CalcYield_yd3",
    "FlyAshFraction",
    "StandardCuredSpecimenCount28",
    "UniqueStandardCuredSpecimenCount28"
]

df = df.drop(columns=remove_columns)
print(f"After dropping columns: {df.shape}")
df.to_csv(
    "data\\prepared_28_day_standard_cure\\"
    "03_standard_cured_test_level_28_working_data_drop_rows_drop_columns_1.csv",
    index=False,
)

# extract average actual age is only 28 days

# 