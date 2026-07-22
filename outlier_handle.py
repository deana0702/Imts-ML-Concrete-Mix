import numpy as np

df["WCMQualityFlag"] = np.select(
    [
        df["calcWCRatio"].isna(),
        df["calcWCRatio"].lt(0.20),
        df["calcWCRatio"].between(0.20, 0.2615, inclusive="left"),
        df["calcWCRatio"].between(0.2615, 0.59, inclusive="both"),
        df["calcWCRatio"].between(0.59, 0.90, inclusive="right"),
        df["calcWCRatio"].gt(0.90),
    ],
    [
        "MISSING",
        "SUSPICIOUS_LOW",
        "LOW_REVIEW",
        "CORE_RANGE",
        "HIGH_REVIEW",
        "SUSPICIOUS_HIGH",
    ],
    default="UNKNOWN",
)