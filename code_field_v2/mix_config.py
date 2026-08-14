"""Editable settings for IMTS historical-data Mix Optimization.

No command-line arguments are used. Edit this file and run scripts 04, 05, 06.
"""

from pathlib import Path

CONFIG_VERSION = "2026-08-13-imts-mix-optimization-v1"

# Input: CSV/Parquet produced by the IMTS SQL extract containing batch columns.
ROOT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = ROOT_DIR / "data/field_core_outputs_v2/mix_optimization"
RAW_INPUT_PATH = ROOT_DIR / "data/concrete_us_data_v2.csv"
PREPARED_DATA_PATH = OUTPUT_DIR / "preprocessed/mix_training_data.csv"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "models"
OPTIMIZATION_OUTPUT_DIR = OUTPUT_DIR / "candidates"

# IMTS unit/filter settings.
FILTER_UNIT_SYSTEM = True
UNIT_SYSTEM_TO_KEEP = 0
REQUIRE_VALID_CAST_DATE = True
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_JOBS = -1

# IMTS often leaves flyAshQuantity blank when a mix uses no fly ash. Keep True
# only if this interpretation is confirmed for the offices being analyzed.
ASSUME_NULL_FLY_ASH_IS_ZERO_WHEN_CEMENT_EXISTS = True

# A usable row must have every core batch feature below. This intentionally
# favors reliable complete rows rather than imputing the material recipe.
CORE_BATCH_FEATURES = [
    "CalcCementContent_lbs_yd3",
    "FlyAshContent_lbs_yd3",
    "calcWCRatio",
    "SandSSD_lbs_yd3",
    "AggregateSSD_lbs_yd3",
]

OPTIONAL_BATCH_FEATURES = [
    "SandMoisture_percent",
    "AggregateMoisture_percent",
    "WaterAdded_gal_per_yd3",
    "LoadBatchVolume_yd3",
    "CalcYield_yd3",
]

DERIVED_BATCH_FEATURES = [
    "TotalCementitiousContent_lbs_yd3",
    "FlyAshFraction",
    "TotalAggregateSSD_lbs_yd3",
    "SandFractionOfAggregate",
]

MODEL_FEATURES = CORE_BATCH_FEATURES + OPTIONAL_BATCH_FEATURES + DERIVED_BATCH_FEATURES

STRENGTH_TARGET = "AverageActualStrength28_psi"
SLUMP_TARGET = "EffectiveSlump_in"
AIR_TARGET = "EffectiveAir_percent"
REQUIRED_STRENGTH_COLUMN = "ApplicableSpecifiedStrength28"
GROUP_COLUMN = "projectId"
ID_COLUMNS = [
    "testId", "projectId", "officeId", "SupplierName", "plantNumber", "mixNumber"
]

# Conservative data validity ranges. Invalid values become missing.
VALID_RANGES = {
    "CalcCementContent_lbs_yd3": (100.0, 1500.0),
    "FlyAshContent_lbs_yd3": (0.0, 800.0),
    "TotalCementitiousContent_lbs_yd3": (100.0, 1800.0),
    "FlyAshFraction": (0.0, 0.90),
    "calcWCRatio": (0.15, 1.00),
    "SandSSD_lbs_yd3": (100.0, 5000.0),
    "AggregateSSD_lbs_yd3": (100.0, 6000.0),
    "SandMoisture_percent": (-10.0, 30.0),
    "AggregateMoisture_percent": (-10.0, 30.0),
    "WaterAdded_gal_per_yd3": (0.0, 100.0),
    "LoadBatchVolume_yd3": (0.1, 30.0),
    "CalcYield_yd3": (0.1, 30.0),
    "AverageActualStrength28_psi": (100.0, 30000.0),
    "EffectiveSlump_in": (0.0, 20.0),
    "EffectiveAir_percent": (0.0, 20.0),
    "ApplicableSpecifiedStrength28": (500.0, 20000.0),
}

# Optimization request. Edit before running script 06.
OPT_REQUIRED_STRENGTH_PSI = 4000.0
OPT_STRENGTH_SAFETY_MARGIN_PSI = 0.0
OPT_SLUMP_MIN_IN = None       # Example: 3.0
OPT_SLUMP_MAX_IN = None       # Example: 5.0
OPT_AIR_MIN_PERCENT = None    # Example: 5.0
OPT_AIR_MAX_PERCENT = None    # Example: 7.0
OPT_WC_RATIO_MAX = None       # Example: 0.45
OPT_SUPPLIER_NAME = None      # Exact value or None
OPT_PLANT_NUMBER = None       # Exact value or None
OPT_TOP_N = 25

# Objective uses IMTS data only. True carbon optimization is unavailable
# without approved material-specific emission factors.
OPTIMIZATION_OBJECTIVE = "cement"  # "cement" or "total_cementitious"

# Do not use a surrogate constraint when its held-out R2 is below this value.
MIN_SURROGATE_R2_FOR_OPTIMIZATION = 0.10
