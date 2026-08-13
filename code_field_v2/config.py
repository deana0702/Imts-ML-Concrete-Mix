"""Configuration for the IMTS Concrete Predictive Quality Analysis pipeline.

This file intentionally contains no model training code.  It centralizes the
column names and feature sets shared by the preprocessing, regression, and
classification scripts.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# File locations
# ---------------------------------------------------------------------------
DEFAULT_INPUT_PATH = Path("data/concrete_us_data_v2.csv")
DEFAULT_OUTPUT_DIR = Path("data/field_core_outputs_v2/preprocessed")


# ---------------------------------------------------------------------------
# Dataset identity and targets
# ---------------------------------------------------------------------------
ID_COLUMNS = ["officeId", "projectId", "SampleId", "testId"]
AUDIT_ID_COLUMNS = ["OfficeName", "projectNo", "labNo", "CastDate"]
GROUP_COLUMN = "projectId"
DATE_COLUMN = "CastDate"
UNIT_SYSTEM_COLUMN = "ConcreteTestUnitSystem"
US_UNIT_SYSTEM_ID = 0

REGRESSION_TARGET = "AverageActualStrength28_psi"
CLASSIFICATION_TARGET = "FailureFlag28"


# ---------------------------------------------------------------------------
# Day-0 features: known at or near placement time
# ---------------------------------------------------------------------------
DAY0_REQUIRED_FEATURES = [
    "ApplicableSpecifiedStrength28",
]

DAY0_FIELD_NUMERIC_FEATURES = [
    "EffectiveSlump_in",
    "EffectiveSpread_in",
    "EffectiveAir_percent",
    "EffectiveUnitWeight_lb_ft3",
    "EffectiveConcreteTemp_F",
    "AmbientTemp_F",
]

DAY0_OPERATION_NUMERIC_FEATURES = [
    "WaterAdded_gal_per_yd3",
    "BatchToSampleMinutes",
    "BatchToCastMinutes",
]

DAY0_OPERATION_BINARY_FEATURES = [
    "HasWaterAdded",
    "HasAnyAfterSPMeasurement",
]

DAY0_MISSING_INDICATORS = [
    "EffectiveSlumpMissing",
    "EffectiveSpreadMissing",
    "EffectiveAirMissing",
    "EffectiveUnitWeightMissing",
    "EffectiveConcreteTempMissing",
    "WaterAddedPerVolumeMissing",
    "InitialCuringConditionMissing",
]


# ---------------------------------------------------------------------------
# Specification and compliance features: known at Day 0
# ---------------------------------------------------------------------------
SPECIFICATION_LIMIT_FEATURES = [
    "uwSlump_specMin",
    "uwSlump_specMax",
    "uwSpread_specMin",
    "uwSpread_specMax",
    "uwAir_specMin",
    "uwAir_specMax",
    "uwWeight_specMin",
    "uwWeight_specMax",
    "uwConcreteTemp_specMin",
    "uwConcreteTemp_specMax",
]

COMPLIANCE_FLAG_FEATURES = [
    "SlumpOutOfSpecFlag",
    "SpreadOutOfSpecFlag",
    "AirOutOfSpecFlag",
    "UnitWeightOutOfSpecFlag",
    "ConcreteTempOutOfSpecFlag",
    "HasAnyFieldMeasurementWithSpec",
    "HasAnyFieldMeasurementOutOfSpec",
]

DERIVED_COMPLIANCE_FEATURES = [
    "SlumpBelowMinAmount",
    "SlumpAboveMaxAmount",
    "SpreadBelowMinAmount",
    "SpreadAboveMaxAmount",
    "AirBelowMinAmount",
    "AirAboveMaxAmount",
    "UnitWeightBelowMinAmount",
    "UnitWeightAboveMaxAmount",
    "ConcreteTempBelowMinAmount",
    "ConcreteTempAboveMaxAmount",
    "FieldOutOfSpecCount",
]


# ---------------------------------------------------------------------------
# Day-7 additions: never use in the Day-0 model
# ---------------------------------------------------------------------------
DAY7_FEATURES = [
    "AverageActualStrength7_psi",
    "MinimumActualStrength7_psi",
    "MaximumActualStrength7_psi",
    "ActualStrengthRange7_psi",
    "ActualStrength7SpecimenCount",
]

DAY7_DERIVED_FEATURES = [
    "Strength7ToSpecifiedStrength28Ratio",
    "Strength7Margin_psi",
]


# ---------------------------------------------------------------------------
# Optional context. These are preserved for later leakage-controlled encoding.
# They are not included in the initial numeric Field-Core feature sets.
# ---------------------------------------------------------------------------
CONTEXT_COLUMNS = ["SupplierName", "plantNumber", "mixNumber"]


# ---------------------------------------------------------------------------
# Model-ready feature sets
# ---------------------------------------------------------------------------
DAY0_FIELD_CORE_FEATURES = (
    DAY0_REQUIRED_FEATURES
    + DAY0_FIELD_NUMERIC_FEATURES
    + DAY0_OPERATION_NUMERIC_FEATURES
    + DAY0_OPERATION_BINARY_FEATURES
    + DAY0_MISSING_INDICATORS
)

DAY0_WITH_SPEC_FEATURES = DAY0_FIELD_CORE_FEATURES + SPECIFICATION_LIMIT_FEATURES

DAY0_WITH_COMPLIANCE_FEATURES = (
    DAY0_WITH_SPEC_FEATURES
    + COMPLIANCE_FLAG_FEATURES
    + DERIVED_COMPLIANCE_FEATURES
)

DAY7_WITH_COMPLIANCE_FEATURES = (
    DAY0_WITH_COMPLIANCE_FEATURES
    + DAY7_FEATURES
    + DAY7_DERIVED_FEATURES
)

FEATURE_SETS = {
    "Day0_FieldCore": DAY0_FIELD_CORE_FEATURES,
    "Day0_FieldPlusSpec": DAY0_WITH_SPEC_FEATURES,
    "Day0_FieldPlusCompliance": DAY0_WITH_COMPLIANCE_FEATURES,
    "Day7_FieldPlusCompliance": DAY7_WITH_COMPLIANCE_FEATURES,
}


# ---------------------------------------------------------------------------
# Columns that must never be used as predictive inputs
# ---------------------------------------------------------------------------
LEAKAGE_COLUMNS = [
    "AverageActualStrength28_psi",
    "MinimumActualStrength28_psi",
    "MaximumActualStrength28_psi",
    "ActualStrengthRange28_psi",
    "StrengthMargin28_psi",
    "FailureFlag28",
    "FieldOutOfSpecAndStrengthFailure28",
    "BelowSpecifiedStrength28SpecimenCount",
    "AtOrAboveSpecifiedStrength28SpecimenCount",
    "UnevaluableStrength28SpecimenCount",
    "HasAnyBelowSpecifiedStrength28Specimen",
    "BelowSpecifiedStrength28SpecimenPercent",
    "ActualStrength7To28Ratio",
]


# ---------------------------------------------------------------------------
# Conservative validity rules. Out-of-range values become missing, not rows
# deleted. Review and adjust these boundaries with a concrete domain expert.
# ---------------------------------------------------------------------------
VALID_RANGES = {
    "ApplicableSpecifiedStrength28": (500.0, 20_000.0),
    "AverageActualStrength7_psi": (100.0, 30_000.0),
    "AverageActualStrength28_psi": (100.0, 30_000.0),
    "EffectiveSlump_in": (0.0, 20.0),
    "EffectiveSpread_in": (0.0, 60.0),
    "EffectiveAir_percent": (0.0, 20.0),
    "EffectiveUnitWeight_lb_ft3": (50.0, 200.0),
    "EffectiveConcreteTemp_F": (20.0, 130.0),
    "AmbientTemp_F": (-40.0, 140.0),
    "WaterAdded_gal_per_yd3": (0.0, 100.0),
    "BatchToSampleMinutes": (0.0, 1_440.0),
    "BatchToCastMinutes": (0.0, 1_440.0),
}


BASE_REQUIRED_COLUMNS = [
    "testId",
    "projectId",
    "ApplicableSpecifiedStrength28",
    "AverageActualStrength28_psi",
]

