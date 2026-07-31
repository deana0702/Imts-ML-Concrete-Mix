/* ============================================================================
   IMTS Concrete Field-Core Predictive 28-Day Strength Dataset

   Purpose:
       Build a one-row-per-concrete-test extract for the first predictive model:
       field measurements available at/near placement time -> 28-day strength.

   Main model target:
       AverageActualStrength28_psi

   Recommended first-model features:
       - ApplicableSpecifiedStrength28
       - EffectiveSlump_in
       - EffectiveAir_percent
       - EffectiveUnitWeight_lb_ft3
       - EffectiveConcreteTemp_F
       - AmbientTemp_F
       - WaterAdded_gal_per_yd3
       - HasWaterAdded
       - BatchToSampleMinutes
       - BatchToCastMinutes
       - HasAnyAfterSPMeasurement

   Important design decisions:
       1. One row per concrete test.
       2. Standard-cured specimens only for 7-day and 28-day strength summaries.
       3. After-SP measurement is preferred when valid; otherwise Actual is used.
       4. N/A values are converted to NULL; they are never converted to zero.
       5. Water added is normalized by the current load/batch volume, not cumulative volume.
       6. Initial curing free text is retained as Raw and Clean text for later mapping.
       7. Final cure is retained for audit but should NOT be used in the placement-time model.
       8. The concrete-test-level unitSystem is used as the effective unit system.
          This is safer than inferring units again from office/project settings because
          the test record is expected to contain the resolved unit after any override.
       9. This query defaults to one unit system only. In the supplied US sample,
          c.unitSystem = 0. Verify that ID before running against production.
      10. hoursToAge is intentionally not referenced for old-database compatibility.
      11. Offices listed in @ExcludedOfficeIds are omitted from both the model extract
          and the coverage summary.
   ============================================================================ */

SET NOCOUNT ON;

DECLARE @RequiredStrengthTypeId int = 30010;
DECLARE @DesignStrengthTypeId   int = 30011;

/*
    Office filter:
      NULL = all offices within the selected concrete-test unit system.
*/
DECLARE @OfficeId int = NULL;

/*
    Offices to exclude from this extract.

    These offices are excluded even when @OfficeId is NULL. If @OfficeId is set
    to one of these IDs, no rows will be returned for that office.

    Add or remove OfficeId values here as needed.
*/
DECLARE @ExcludedOfficeIds TABLE
(
    OfficeId int NOT NULL PRIMARY KEY
);

INSERT INTO @ExcludedOfficeIds (OfficeId)
VALUES
    (1),
    (26),
    (64);

/*
    IMPORTANT: Verify the actual ID in IMTS.
    The provided US raw-data sample used c.unitSystem = 0.
*/
DECLARE @TargetConcreteUnitSystem int = 0;

/* Optional user-selected cast-date window. @EndCastDate is inclusive. */
DECLARE @StartCastDate date = NULL;
DECLARE @EndCastDate   date = NULL;

/*
    Data-quality boundary for obviously invalid dates.
    These boundaries do not change the stored data; they only control model output.
*/
DECLARE @MinimumValidCastDate date = '2000-01-01';
DECLARE @MaximumValidCastDate date = CAST(GETDATE() AS date);
DECLARE @IncludeInvalidCastDates bit = 0;

/* Model-output filters. Coverage summary still evaluates all rows in #FinalData. */
DECLARE @Require28DayActual bit = 1;
DECLARE @RequireAnyFieldCoreMeasurement bit = 1;

IF OBJECT_ID('tempdb..#FinalData') IS NOT NULL
BEGIN
    DROP TABLE #FinalData;
END;

;WITH NormalizedBatchRows AS
(
    SELECT
        b.id AS BatchRowId,
        b.concreteTestId,

        CASE
            WHEN UPPER(LTRIM(RTRIM(b.name))) IN
            (
                'MISC SAND 1',
                'MISC SAND 2'
            )
                THEN 'SAND'

            WHEN UPPER(LTRIM(RTRIM(b.name))) IN
            (
                'MISC AGGREGATE 1',
                'MISC AGGREGATE 2',
                'MISC AGGREGATE 3'
            )
                THEN 'AGGREGATE'

            ELSE 'OTHER'
        END AS MaterialGroup,

        CAST(b.weight AS decimal(18, 4)) AS BatchWeight_lbs,
        CAST(b.moisture AS decimal(18, 6)) AS Moisture_percent,

        /*
            IMTS calculation:

            Batch Weight = SSD Weight + Water Weight
            Moisture %   = Water Weight / SSD Weight * 100

            Therefore:
            SSD Weight = Batch Weight / (1 + Moisture / 100)
        */
        CASE
            WHEN b.moisture IS NULL
              OR b.moisture <= -100
                THEN NULL

            ELSE
                CAST(b.weight AS decimal(18, 4))
                /
                NULLIF
                (
                    1.0 + CAST(b.moisture AS decimal(18, 6)) / 100.0,
                    0
                )
        END AS SSDWeight_lbs,

        /* Water Weight = Batch Weight - SSD Weight */
        CASE
            WHEN b.moisture IS NULL
              OR b.moisture <= -100
                THEN NULL

            ELSE
                CAST(b.weight AS decimal(18, 4))
                -
                (
                    CAST(b.weight AS decimal(18, 4))
                    /
                    NULLIF
                    (
                        1.0 + CAST(b.moisture AS decimal(18, 6)) / 100.0,
                        0
                    )
                )
        END AS WaterWeight_lbs,

        CASE
            WHEN b.moisture IS NULL
              OR b.moisture <= -100
                THEN 1
            ELSE 0
        END AS HasInvalidMoisture

    FROM dbo.FieldConcreteBatchRows AS b

    /* Zero-weight rows mean the material slot was not used. */
    WHERE b.weight > 0
),

BatchData AS
(
    SELECT
        nbr.concreteTestId,

        /* =========================================================
           Sand 1 + Sand 2
           ========================================================= */
        SUM
        (
            CASE
                WHEN nbr.MaterialGroup = 'SAND'
                    THEN nbr.BatchWeight_lbs
                ELSE 0
            END
        ) AS SandBatchWeight_lbs,

        SUM
        (
            CASE
                WHEN nbr.MaterialGroup = 'SAND'
                    THEN nbr.SSDWeight_lbs
                ELSE 0
            END
        ) AS SandSSDWeight_lbs,

        SUM
        (
            CASE
                WHEN nbr.MaterialGroup = 'SAND'
                    THEN nbr.WaterWeight_lbs
                ELSE 0
            END
        ) AS SandWaterWeight_lbs,

        /*
            Combined sand moisture:
            Total Water Weight / Total SSD Weight * 100

            If any used sand row has missing/invalid moisture,
            return NULL rather than calculating a partial value.
        */
        CASE
            WHEN MAX
            (
                CASE
                    WHEN nbr.MaterialGroup = 'SAND'
                        THEN nbr.HasInvalidMoisture
                    ELSE 0
                END
            ) = 1
                THEN NULL

            ELSE
                SUM
                (
                    CASE
                        WHEN nbr.MaterialGroup = 'SAND'
                            THEN nbr.WaterWeight_lbs
                        ELSE 0
                    END
                )
                /
                NULLIF
                (
                    SUM
                    (
                        CASE
                            WHEN nbr.MaterialGroup = 'SAND'
                                THEN nbr.SSDWeight_lbs
                            ELSE 0
                        END
                    ),
                    0
                )
                * 100.0
        END AS SandMoisture_percent,

        SUM
        (
            CASE
                WHEN nbr.MaterialGroup = 'SAND'
                    THEN 1
                ELSE 0
            END
        ) AS SandComponentCount,

        /* =========================================================
           Aggregate 1 + Aggregate 2 + Aggregate 3
           ========================================================= */
        SUM
        (
            CASE
                WHEN nbr.MaterialGroup = 'AGGREGATE'
                    THEN nbr.BatchWeight_lbs
                ELSE 0
            END
        ) AS AggregateBatchWeight_lbs,

        SUM
        (
            CASE
                WHEN nbr.MaterialGroup = 'AGGREGATE'
                    THEN nbr.SSDWeight_lbs
                ELSE 0
            END
        ) AS AggregateSSDWeight_lbs,

        SUM
        (
            CASE
                WHEN nbr.MaterialGroup = 'AGGREGATE'
                    THEN nbr.WaterWeight_lbs
                ELSE 0
            END
        ) AS AggregateWaterWeight_lbs,

        /* Combined aggregate moisture */
        CASE
            WHEN MAX
            (
                CASE
                    WHEN nbr.MaterialGroup = 'AGGREGATE'
                        THEN nbr.HasInvalidMoisture
                    ELSE 0
                END
            ) = 1
                THEN NULL

            ELSE
                SUM
                (
                    CASE
                        WHEN nbr.MaterialGroup = 'AGGREGATE'
                            THEN nbr.WaterWeight_lbs
                        ELSE 0
                    END
                )
                /
                NULLIF
                (
                    SUM
                    (
                        CASE
                            WHEN nbr.MaterialGroup = 'AGGREGATE'
                                THEN nbr.SSDWeight_lbs
                            ELSE 0
                        END
                    ),
                    0
                )
                * 100.0
        END AS AggregateMoisture_percent,

        SUM
        (
            CASE
                WHEN nbr.MaterialGroup = 'AGGREGATE'
                    THEN 1
                ELSE 0
            END
        ) AS AggregateComponentCount

    FROM NormalizedBatchRows AS nbr

    WHERE nbr.MaterialGroup IN
    (
        'SAND',
        'AGGREGATE'
    )

    GROUP BY
        nbr.concreteTestId
),

StrengthByAge AS
(
    SELECT
        cs.concreteTestId,
        cs.days AS BreakAge,

        /*
            Required and Design are separated.
            If duplicates exist for the same test, age and type,
            MAX returns the larger/conservative value.
        */
        MAX
        (
            CASE
                WHEN cs.strengthType = @RequiredStrengthTypeId
                    THEN CAST(cs.strength AS decimal(18, 2))
            END
        ) AS RequiredStrength,

        MAX
        (
            CASE
                WHEN cs.strengthType = @DesignStrengthTypeId
                    THEN CAST(cs.strength AS decimal(18, 2))
            END
        ) AS DesignStrength,

        SUM
        (
            CASE
                WHEN cs.strengthType = @RequiredStrengthTypeId
                    THEN 1
                ELSE 0
            END
        ) AS RequiredStrengthRowCount,

        SUM
        (
            CASE
                WHEN cs.strengthType = @DesignStrengthTypeId
                    THEN 1
                ELSE 0
            END
        ) AS DesignStrengthRowCount

    FROM dbo.FieldConcreteStrengthRows AS cs

    WHERE cs.days IS NOT NULL
      AND cs.strength IS NOT NULL
      AND cs.strengthType IN
      (
          @RequiredStrengthTypeId,
          @DesignStrengthTypeId
      )

    GROUP BY
        cs.concreteTestId,
        cs.days
),

StrengthTarget28 AS
(
    SELECT
        sba.concreteTestId,
        sba.BreakAge AS SpecifiedBreakAge,
        sba.RequiredStrength AS RequiredStrength28,
        sba.DesignStrength AS DesignStrength28,

        /* Prefer Required when both Required and Design exist. */
        COALESCE
        (
            sba.RequiredStrength,
            sba.DesignStrength
        ) AS ApplicableSpecifiedStrength28,

        CASE
            WHEN sba.RequiredStrength IS NOT NULL
                THEN 'Required'

            WHEN sba.DesignStrength IS NOT NULL
                THEN 'Design'

            ELSE NULL
        END AS ApplicableStrengthType28,

        sba.RequiredStrengthRowCount AS RequiredStrength28RowCount,
        sba.DesignStrengthRowCount AS DesignStrength28RowCount

    FROM StrengthByAge AS sba

    WHERE sba.BreakAge = 28
),

SpecimenStrengthSummary AS
(
    SELECT
        r.concreteTestId,

        /* =========================================================
           Standard-cured 7-day actual strength
           ========================================================= */
        AVG
        (
            CASE
                WHEN r.daysToAge = 7
                 AND r.wasFieldCured = 0
                    THEN CAST
                    (
                        COALESCE
                        (
                            r.calcCompressiveStrengthUnrounded,
                            r.calcCompressiveStrength
                        )
                        AS decimal(18, 2)
                    )
            END
        ) AS AverageActualStrength7_psi,

        MIN
        (
            CASE
                WHEN r.daysToAge = 7
                 AND r.wasFieldCured = 0
                    THEN COALESCE
                    (
                        r.calcCompressiveStrengthUnrounded,
                        r.calcCompressiveStrength
                    )
            END
        ) AS MinimumActualStrength7_psi,

        MAX
        (
            CASE
                WHEN r.daysToAge = 7
                 AND r.wasFieldCured = 0
                    THEN COALESCE
                    (
                        r.calcCompressiveStrengthUnrounded,
                        r.calcCompressiveStrength
                    )
            END
        ) AS MaximumActualStrength7_psi,

        SUM
        (
            CASE
                WHEN r.daysToAge = 7
                 AND r.wasFieldCured = 0
                 AND COALESCE
                 (
                     r.calcCompressiveStrengthUnrounded,
                     r.calcCompressiveStrength
                 ) IS NOT NULL
                    THEN 1
                ELSE 0
            END
        ) AS ActualStrength7SpecimenCount,


        /* =========================================================
           Standard-cured 28-day actual strength
           ========================================================= */
        AVG
        (
            CASE
                WHEN r.daysToAge = 28
                 AND r.wasFieldCured = 0
                    THEN CAST
                    (
                        COALESCE
                        (
                            r.calcCompressiveStrengthUnrounded,
                            r.calcCompressiveStrength
                        )
                        AS decimal(18, 2)
                    )
            END
        ) AS AverageActualStrength28_psi,

        MIN
        (
            CASE
                WHEN r.daysToAge = 28
                 AND r.wasFieldCured = 0
                    THEN COALESCE
                    (
                        r.calcCompressiveStrengthUnrounded,
                        r.calcCompressiveStrength
                    )
            END
        ) AS MinimumActualStrength28_psi,

        MAX
        (
            CASE
                WHEN r.daysToAge = 28
                 AND r.wasFieldCured = 0
                    THEN COALESCE
                    (
                        r.calcCompressiveStrengthUnrounded,
                        r.calcCompressiveStrength
                    )
            END
        ) AS MaximumActualStrength28_psi,

        SUM
        (
            CASE
                WHEN r.daysToAge = 28
                 AND r.wasFieldCured = 0
                 AND COALESCE
                 (
                     r.calcCompressiveStrengthUnrounded,
                     r.calcCompressiveStrength
                 ) IS NOT NULL
                    THEN 1
                ELSE 0
            END
        ) AS ActualStrength28SpecimenCount,


        /* Audit counts */
        SUM
        (
            CASE
                WHEN r.wasFieldCured = 1
                 AND r.daysToAge = 7
                 AND COALESCE
                 (
                     r.calcCompressiveStrengthUnrounded,
                     r.calcCompressiveStrength
                 ) IS NOT NULL
                    THEN 1
                ELSE 0
            END
        ) AS FieldCuredStrength7SpecimenCount,

        SUM
        (
            CASE
                WHEN r.wasFieldCured = 1
                 AND r.daysToAge = 28
                 AND COALESCE
                 (
                     r.calcCompressiveStrengthUnrounded,
                     r.calcCompressiveStrength
                 ) IS NOT NULL
                    THEN 1
                ELSE 0
            END
        ) AS FieldCuredStrength28SpecimenCount

    FROM dbo.FieldConcreteTestRows AS r

    GROUP BY
        r.concreteTestId
)


SELECT
    /* =============================================================
       Office, project, sample and test identifiers
       ============================================================= */
    o.officeId,
    o.name AS OfficeName,

    p.projectId,
    p.projectNo,

    s.id AS SampleId,
    s.labNo,

    t.testId,
    c.id AS ConcreteTestDataId,
    c.testSubTypeId,

    c.supplierId,
    sp.name AS SupplierName,

    /*
       Use the unit stored on the concrete test as the resolved/effective unit.
       This query filters to one unit system so the unit-specific aliases below
       remain valid for the first US-only model.
    */
    c.unitSystem AS ConcreteTestUnitSystem,
    c.officeRegion,

    /* =============================================================
       Placement, supplier, plant, truck and mix context
       Context fields are retained for audit and later models.
       Supplier/plant/mix should not be used in the first Field-Core baseline.
       ============================================================= */
    c.castDate,

    CASE
        WHEN c.castDate IS NOT NULL
         AND c.castDate >= @MinimumValidCastDate
         AND c.castDate < DATEADD(DAY, 1, @MaximumValidCastDate)
            THEN 1
        ELSE 0
    END AS IsValidCastDate,

    c.placementType,
    c.sampledFrom,
    c.plantNumber,
    c.mixNumber,
    c.truckNumber,
    c.truckLoadNum,
    c.ofTotalTruckLoads,
    c.ticketNumber,

    c.batchSize AS LoadBatchVolumeRaw,
    c.[LoadBatchVolumneNA] AS LoadBatchVolumeNA,
    c.loadVolumeUnits,
    c.cumulativeLoadVolume,

    c.revolution AS MixerDrumRevolutions,
    c.truckDischarge AS FinishOfTruckDischarge,

    /* =============================================================
       Batch, sample and cast times
       ============================================================= */
    c.batchTime,
    c.sampleTime,
    c.finishTime AS castTime,

    CASE
        WHEN c.batchTime IS NULL
          OR c.sampleTime IS NULL
            THEN NULL

        WHEN DATEDIFF(MINUTE, c.batchTime, c.sampleTime) < 0
            THEN DATEDIFF(MINUTE, c.batchTime, c.sampleTime) + 1440

        ELSE DATEDIFF(MINUTE, c.batchTime, c.sampleTime)
    END AS BatchToSampleMinutes,

    CASE
        WHEN c.batchTime IS NULL
          OR c.finishTime IS NULL
            THEN NULL

        WHEN DATEDIFF(MINUTE, c.batchTime, c.finishTime) < 0
            THEN DATEDIFF(MINUTE, c.batchTime, c.finishTime) + 1440

        ELSE DATEDIFF(MINUTE, c.batchTime, c.finishTime)
    END AS BatchToCastMinutes,

    /* =============================================================
       Environmental and curing fields
       ============================================================= */
    l2.value AS CloudType,
    l3.value AS PrecipitationType,
    l4.value AS WindType,

    c.hiTemp AS InitialCuringHighTemp_F,
    c.lowTemp AS InitialCuringLowTemp_F,

    CASE
        WHEN ISNULL(c.ambientTempNA, 0) = 1
            THEN NULL
        ELSE CAST(c.ambientTemp AS decimal(18, 4))
    END AS AmbientTemp_F,

    c.ambientTemp AS AmbientTempRaw,
    c.ambientTempNA,

    NULLIF(LTRIM(RTRIM(c.initialCuringCondition)), '')
        AS InitialCuringConditionRaw,

    LOWER(NULLIF(LTRIM(RTRIM(c.initialCuringCondition)), ''))
        AS InitialCuringConditionClean,

    CASE
        WHEN NULLIF(LTRIM(RTRIM(c.initialCuringCondition)), '') IS NULL
            THEN 1
        ELSE 0
    END AS InitialCuringConditionMissing,

    /* Audit only for the placement-time model; avoid future-information leakage. */
    l5.value AS FinalCure_AuditOnly,

    /* =============================================================
       Water added and load-volume normalization

       Current-load denominator:
           WaterAdded_gal_per_yd3 = WaterAdded_gallons / LoadBatchVolume_yd3

       cumulativeLoadVolume is retained only for audit and is NOT the denominator.
       ============================================================= */
    c.waterAdded AS WaterAddedRaw,
    c.waterAddedNA,

    wm.WaterAdded_gallons,
    wm.LoadBatchVolume_yd3,
    wm.WaterAdded_gal_per_yd3,
    wm.WaterAdded_lb_per_yd3,
    wm.HasWaterAdded,

    CASE
        WHEN wm.WaterAdded_gal_per_yd3 IS NULL
            THEN 1
        ELSE 0
    END AS WaterAddedPerVolumeMissing,

    c.adMixture AS AdmixtureText_AuditOnly,

    /* =============================================================
       Model-ready effective fresh-concrete measurements

       Rule:
           valid After-SP value -> use After-SP
           otherwise valid Actual value -> use Actual
           otherwise NULL
       ============================================================= */
    fm.EffectiveSlump_in,
    fm.SlumpMeasurementSource,
    CASE WHEN fm.EffectiveSlump_in IS NULL THEN 1 ELSE 0 END
        AS EffectiveSlumpMissing,

    fm.EffectiveSpread_in,
    fm.SpreadMeasurementSource,
    CASE WHEN fm.EffectiveSpread_in IS NULL THEN 1 ELSE 0 END
        AS EffectiveSpreadMissing,

    fm.EffectiveAir_percent,
    fm.AirMeasurementSource,
    CASE WHEN fm.EffectiveAir_percent IS NULL THEN 1 ELSE 0 END
        AS EffectiveAirMissing,

    fm.EffectiveUnitWeight_lb_ft3,
    fm.UnitWeightMeasurementSource,
    CASE WHEN fm.EffectiveUnitWeight_lb_ft3 IS NULL THEN 1 ELSE 0 END
        AS EffectiveUnitWeightMissing,

    fm.EffectiveConcreteTemp_F,
    fm.ConcreteTempMeasurementSource,
    CASE WHEN fm.EffectiveConcreteTemp_F IS NULL THEN 1 ELSE 0 END
        AS EffectiveConcreteTempMissing,

    fm.HasAfterSPSlump,
    fm.HasAfterSPSpread,
    fm.HasAfterSPAir,
    fm.HasAfterSPUnitWeight,
    fm.HasAfterSPConcreteTemp,

    CASE
        WHEN fm.HasAfterSPSlump = 1
          OR fm.HasAfterSPSpread = 1
          OR fm.HasAfterSPAir = 1
          OR fm.HasAfterSPUnitWeight = 1
          OR fm.HasAfterSPConcreteTemp = 1
            THEN 1
        ELSE 0
    END AS HasAnyAfterSPMeasurement,

    /* =============================================================
       Raw fresh-concrete values and N/A flags retained for audit
       ============================================================= */
    c.uwSlump_actual,
    c.uwSlump_actualNA,
    c.uwSlump_afterSP,
    c.uwSlump_afterSPNA,
    c.uwSlump_specMin,
    c.uwSlump_specMinNA,
    c.uwSlump_specMax,
    c.uwSlump_specMaxNA,

    c.uwSpread_actual,
    c.uwSpread_actualNA,
    c.uwSpread_afterSP,
    c.uwSpread_afterSPNA,
    c.uwSpread_specMin,
    c.uwSpread_specMinNA,
    c.uwSpread_specMax,
    c.uwSpread_specMaxNA,

    c.uwAir_actual,
    c.uwAir_actualNA,
    c.uwAir_afterSP,
    c.uwAir_afterSPNA,
    c.uwAir_specMin,
    c.uwAir_specMinNA,
    c.uwAir_specMax,
    c.uwAir_specMaxNA,

    c.uwWeight_actual,
    c.uwWeight_actualNA,
    c.uwWeight_afterSP,
    c.uwWeight_afterSPNA,
    c.uwWeight_specMin,
    c.uwWeight_specMinNA,
    c.uwWeight_specMax,
    c.uwWeight_specMaxNA,

    c.uwConcreteTemp_actual,
    c.uwConcreteTemp_actualNA,
    c.uwConcreteTemp_afterSP,
    c.uwConcreteTemp_afterSPNA,
    c.uwConcreteTemp_specMin,
    c.uwConcreteTemp_specMinNA,
    c.uwConcreteTemp_specMax,
    c.uwConcreteTemp_specMaxNA,

    c.wasAggregateCorrectionDone,
    c.resAirContentASTMsForUS,

    /* =============================================================
       28-day specified strength
       ============================================================= */
    st28.SpecifiedBreakAge,
    st28.RequiredStrength28,
    st28.DesignStrength28,
    st28.ApplicableSpecifiedStrength28,
    st28.ApplicableStrengthType28,
    st28.RequiredStrength28RowCount,
    st28.DesignStrength28RowCount,

    CASE
        WHEN st28.ApplicableSpecifiedStrength28 IS NULL
            THEN 1
        ELSE 0
    END AS SpecifiedStrengthMissing,

    /* =============================================================
       Standard-cured 7-day actual strength summary
       Retained for the later 7-day updated model, not the first Field-Core model.
       ============================================================= */
    ss.AverageActualStrength7_psi,
    ss.MinimumActualStrength7_psi,
    ss.MaximumActualStrength7_psi,

    ss.MaximumActualStrength7_psi
        - ss.MinimumActualStrength7_psi
        AS ActualStrengthRange7_psi,

    ss.ActualStrength7SpecimenCount,

    /* =============================================================
       Standard-cured 28-day target
       ============================================================= */
    ss.AverageActualStrength28_psi,
    ss.MinimumActualStrength28_psi,
    ss.MaximumActualStrength28_psi,

    ss.MaximumActualStrength28_psi
        - ss.MinimumActualStrength28_psi
        AS ActualStrengthRange28_psi,

    ss.ActualStrength28SpecimenCount,
    ss.FieldCuredStrength7SpecimenCount,
    ss.FieldCuredStrength28SpecimenCount,

    /* =============================================================
       Predictive-quality targets
       ============================================================= */
    ss.AverageActualStrength28_psi
        - st28.ApplicableSpecifiedStrength28
        AS StrengthMargin28_psi,

    CASE
        WHEN ss.AverageActualStrength28_psi IS NULL
          OR st28.ApplicableSpecifiedStrength28 IS NULL
            THEN NULL

        WHEN ss.AverageActualStrength28_psi
             < st28.ApplicableSpecifiedStrength28
            THEN 1

        ELSE 0
    END AS FailureFlag28,

    CASE
        WHEN ss.AverageActualStrength7_psi IS NULL
          OR ss.AverageActualStrength28_psi IS NULL
          OR ss.AverageActualStrength28_psi = 0
            THEN NULL

        ELSE
            ss.AverageActualStrength7_psi
            / NULLIF(ss.AverageActualStrength28_psi, 0)
    END AS ActualStrength7To28Ratio,

    /* =============================================================
       Batch data retained for coverage reporting and later Mix Optimization.
       Do not use these columns in the first Field-Core model.
       ============================================================= */
    bd.SandBatchWeight_lbs,
    bd.SandSSDWeight_lbs,
    bd.SandWaterWeight_lbs,
    bd.SandMoisture_percent,
    bd.SandComponentCount,

    bd.SandSSDWeight_lbs
        / NULLIF(c.calcYield, 0)
        AS SandSSD_lbs_yd3,

    bd.AggregateBatchWeight_lbs,
    bd.AggregateSSDWeight_lbs,
    bd.AggregateWaterWeight_lbs,
    bd.AggregateMoisture_percent,
    bd.AggregateComponentCount,

    bd.AggregateSSDWeight_lbs
        / NULLIF(c.calcYield, 0)
        AS AggregateSSD_lbs_yd3,

    c.cementQuantity AS CementQuantity_lbs,
    c.flyAshQuantity AS FlyAshQuantity_lbs,
    c.waterQuantity,
    l.value AS WaterUnits,

    c.calcYield AS CalcYield_yd3,
    c.calcWCRatio,
    c.calcCementContent AS CalcCementContent_lbs_yd3,

    c.flyAshQuantity
        / NULLIF(c.calcYield, 0)
        AS FlyAshContent_lbs_yd3,

    (
        c.cementQuantity
        + ISNULL(c.flyAshQuantity, 0)
    )
        / NULLIF(c.calcYield, 0)
        AS TotalCementitiousContent_lbs_yd3,

    CASE
        WHEN c.cementQuantity
             + ISNULL(c.flyAshQuantity, 0) = 0
            THEN NULL

        ELSE
            ISNULL(c.flyAshQuantity, 0)
            /
            NULLIF
            (
                c.cementQuantity
                + ISNULL(c.flyAshQuantity, 0),
                0
            )
    END AS FlyAshFraction,

    c.isNoMoistureGiven,

    /* =============================================================
       Availability and model-candidate flags
       ============================================================= */
    CASE
        WHEN bd.concreteTestId IS NOT NULL
            THEN 1
        ELSE 0
    END AS HasSandOrAggregateBatchRows,

    CASE
        WHEN c.cementQuantity IS NOT NULL
          OR c.flyAshQuantity IS NOT NULL
          OR c.waterQuantity IS NOT NULL
          OR c.calcWCRatio IS NOT NULL
          OR bd.concreteTestId IS NOT NULL
            THEN 1
        ELSE 0
    END AS HasAnyBatchData,

    CASE
        WHEN fm.EffectiveSlump_in IS NOT NULL
          OR fm.EffectiveSpread_in IS NOT NULL
          OR fm.EffectiveAir_percent IS NOT NULL
          OR fm.EffectiveUnitWeight_lb_ft3 IS NOT NULL
          OR fm.EffectiveConcreteTemp_F IS NOT NULL
            THEN 1
        ELSE 0
    END AS HasAnyEffectiveFreshConcreteMeasurement,

    CASE
        WHEN ss.AverageActualStrength28_psi IS NOT NULL
         AND
         (
             fm.EffectiveSlump_in IS NOT NULL
             OR fm.EffectiveSpread_in IS NOT NULL
             OR fm.EffectiveAir_percent IS NOT NULL
             OR fm.EffectiveUnitWeight_lb_ft3 IS NOT NULL
             OR fm.EffectiveConcreteTemp_F IS NOT NULL
         )
         AND c.castDate IS NOT NULL
         AND c.castDate >= @MinimumValidCastDate
         AND c.castDate < DATEADD(DAY, 1, @MaximumValidCastDate)
            THEN 1
        ELSE 0
    END AS IsFieldCoreStrengthCandidate,

    CASE
        WHEN ss.AverageActualStrength28_psi IS NOT NULL
         AND st28.ApplicableSpecifiedStrength28 IS NOT NULL
         AND
         (
             fm.EffectiveSlump_in IS NOT NULL
             OR fm.EffectiveSpread_in IS NOT NULL
             OR fm.EffectiveAir_percent IS NOT NULL
             OR fm.EffectiveUnitWeight_lb_ft3 IS NOT NULL
             OR fm.EffectiveConcreteTemp_F IS NOT NULL
         )
         AND c.castDate IS NOT NULL
         AND c.castDate >= @MinimumValidCastDate
         AND c.castDate < DATEADD(DAY, 1, @MaximumValidCastDate)
            THEN 1
        ELSE 0
    END AS IsFieldCoreWithRequiredStrengthCandidate

INTO #FinalData

FROM dbo.Offices AS o

INNER JOIN dbo.Projects AS p
    ON p.officeId = o.officeId

INNER JOIN dbo.Samples AS s
    ON s.projectId = p.projectId

INNER JOIN dbo.Tests AS t
    ON t.sampleId = s.id

INNER JOIN dbo.FieldConcreteTestDatumBases AS c
    ON c.testbaseId = t.testId

/* Tests without batch rows must remain in the Field-Core extract. */
LEFT JOIN BatchData AS bd
    ON bd.concreteTestId = c.id

LEFT JOIN StrengthTarget28 AS st28
    ON st28.concreteTestId = c.id

LEFT JOIN SpecimenStrengthSummary AS ss
    ON ss.concreteTestId = c.id

LEFT JOIN dbo.Suppliers AS sp
    ON sp.id = c.supplierId

LEFT JOIN dbo.LocalizedResourceNames AS l
    ON l.resourceId = c.resWaterUnits
   AND l.localeId = 0

LEFT JOIN dbo.LocalizedResourceNames AS l2
    ON l2.resourceId = c.resCloudType
   AND l2.localeId = 0

LEFT JOIN dbo.LocalizedResourceNames AS l3
    ON l3.resourceId = c.resPrecipitationType
   AND l3.localeId = 0

LEFT JOIN dbo.LocalizedResourceNames AS l4
    ON l4.resourceId = c.resWindType
   AND l4.localeId = 0

LEFT JOIN dbo.LocalizedResourceNames AS l5
    ON l5.resourceId = c.resFinalCure
   AND l5.localeId = 0

/* -------------------------------------------------------------
   Effective field measurements: valid After-SP first, Actual second.
   ------------------------------------------------------------- */
CROSS APPLY
(
    SELECT
        CASE
            WHEN ISNULL(c.uwSlump_afterSPNA, 0) = 0
             AND c.uwSlump_afterSP IS NOT NULL
                THEN CAST(c.uwSlump_afterSP AS decimal(18, 4))
            WHEN ISNULL(c.uwSlump_actualNA, 0) = 0
             AND c.uwSlump_actual IS NOT NULL
                THEN CAST(c.uwSlump_actual AS decimal(18, 4))
            ELSE NULL
        END AS EffectiveSlump_in,

        CASE
            WHEN ISNULL(c.uwSlump_afterSPNA, 0) = 0
             AND c.uwSlump_afterSP IS NOT NULL
                THEN 'AfterSP'
            WHEN ISNULL(c.uwSlump_actualNA, 0) = 0
             AND c.uwSlump_actual IS NOT NULL
                THEN 'Actual'
            ELSE NULL
        END AS SlumpMeasurementSource,

        CASE
            WHEN ISNULL(c.uwSpread_afterSPNA, 0) = 0
             AND c.uwSpread_afterSP IS NOT NULL
                THEN CAST(c.uwSpread_afterSP AS decimal(18, 4))
            WHEN ISNULL(c.uwSpread_actualNA, 0) = 0
             AND c.uwSpread_actual IS NOT NULL
                THEN CAST(c.uwSpread_actual AS decimal(18, 4))
            ELSE NULL
        END AS EffectiveSpread_in,

        CASE
            WHEN ISNULL(c.uwSpread_afterSPNA, 0) = 0
             AND c.uwSpread_afterSP IS NOT NULL
                THEN 'AfterSP'
            WHEN ISNULL(c.uwSpread_actualNA, 0) = 0
             AND c.uwSpread_actual IS NOT NULL
                THEN 'Actual'
            ELSE NULL
        END AS SpreadMeasurementSource,

        CASE
            WHEN ISNULL(c.uwAir_afterSPNA, 0) = 0
             AND c.uwAir_afterSP IS NOT NULL
                THEN CAST(c.uwAir_afterSP AS decimal(18, 4))
            WHEN ISNULL(c.uwAir_actualNA, 0) = 0
             AND c.uwAir_actual IS NOT NULL
                THEN CAST(c.uwAir_actual AS decimal(18, 4))
            ELSE NULL
        END AS EffectiveAir_percent,

        CASE
            WHEN ISNULL(c.uwAir_afterSPNA, 0) = 0
             AND c.uwAir_afterSP IS NOT NULL
                THEN 'AfterSP'
            WHEN ISNULL(c.uwAir_actualNA, 0) = 0
             AND c.uwAir_actual IS NOT NULL
                THEN 'Actual'
            ELSE NULL
        END AS AirMeasurementSource,

        CASE
            WHEN ISNULL(c.uwWeight_afterSPNA, 0) = 0
             AND c.uwWeight_afterSP IS NOT NULL
                THEN CAST(c.uwWeight_afterSP AS decimal(18, 4))
            WHEN ISNULL(c.uwWeight_actualNA, 0) = 0
             AND c.uwWeight_actual IS NOT NULL
                THEN CAST(c.uwWeight_actual AS decimal(18, 4))
            ELSE NULL
        END AS EffectiveUnitWeight_lb_ft3,

        CASE
            WHEN ISNULL(c.uwWeight_afterSPNA, 0) = 0
             AND c.uwWeight_afterSP IS NOT NULL
                THEN 'AfterSP'
            WHEN ISNULL(c.uwWeight_actualNA, 0) = 0
             AND c.uwWeight_actual IS NOT NULL
                THEN 'Actual'
            ELSE NULL
        END AS UnitWeightMeasurementSource,

        CASE
            WHEN ISNULL(c.uwConcreteTemp_afterSPNA, 0) = 0
             AND c.uwConcreteTemp_afterSP IS NOT NULL
                THEN CAST(c.uwConcreteTemp_afterSP AS decimal(18, 4))
            WHEN ISNULL(c.uwConcreteTemp_actualNA, 0) = 0
             AND c.uwConcreteTemp_actual IS NOT NULL
                THEN CAST(c.uwConcreteTemp_actual AS decimal(18, 4))
            ELSE NULL
        END AS EffectiveConcreteTemp_F,

        CASE
            WHEN ISNULL(c.uwConcreteTemp_afterSPNA, 0) = 0
             AND c.uwConcreteTemp_afterSP IS NOT NULL
                THEN 'AfterSP'
            WHEN ISNULL(c.uwConcreteTemp_actualNA, 0) = 0
             AND c.uwConcreteTemp_actual IS NOT NULL
                THEN 'Actual'
            ELSE NULL
        END AS ConcreteTempMeasurementSource,

        CASE
            WHEN ISNULL(c.uwSlump_afterSPNA, 0) = 0
             AND c.uwSlump_afterSP IS NOT NULL
                THEN 1 ELSE 0
        END AS HasAfterSPSlump,

        CASE
            WHEN ISNULL(c.uwSpread_afterSPNA, 0) = 0
             AND c.uwSpread_afterSP IS NOT NULL
                THEN 1 ELSE 0
        END AS HasAfterSPSpread,

        CASE
            WHEN ISNULL(c.uwAir_afterSPNA, 0) = 0
             AND c.uwAir_afterSP IS NOT NULL
                THEN 1 ELSE 0
        END AS HasAfterSPAir,

        CASE
            WHEN ISNULL(c.uwWeight_afterSPNA, 0) = 0
             AND c.uwWeight_afterSP IS NOT NULL
                THEN 1 ELSE 0
        END AS HasAfterSPUnitWeight,

        CASE
            WHEN ISNULL(c.uwConcreteTemp_afterSPNA, 0) = 0
             AND c.uwConcreteTemp_afterSP IS NOT NULL
                THEN 1 ELSE 0
        END AS HasAfterSPConcreteTemp
) AS fm

/* -------------------------------------------------------------
   Water normalization. batchSize is the current load/batch volume.
   The first model is filtered to the US unit system, so aliases use
   gallons and cubic yards. Verify the unit-system ID before execution.
   ------------------------------------------------------------- */
CROSS APPLY
(
    SELECT
        CASE
            WHEN ISNULL(c.waterAddedNA, 0) = 1
                THEN NULL
            ELSE CAST(c.waterAdded AS decimal(18, 6))
        END AS WaterAdded_gallons,

        CASE
            WHEN ISNULL(c.[LoadBatchVolumneNA], 0) = 1
                THEN NULL
            ELSE CAST(c.batchSize AS decimal(18, 6))
        END AS LoadBatchVolume_yd3
) AS wr

CROSS APPLY
(
    SELECT
        wr.WaterAdded_gallons,
        wr.LoadBatchVolume_yd3,

        CASE
            WHEN wr.WaterAdded_gallons IS NULL
              OR wr.LoadBatchVolume_yd3 IS NULL
              OR wr.LoadBatchVolume_yd3 <= 0
                THEN NULL
            ELSE
                wr.WaterAdded_gallons
                / NULLIF(wr.LoadBatchVolume_yd3, 0)
        END AS WaterAdded_gal_per_yd3,

        CASE
            WHEN wr.WaterAdded_gallons IS NULL
              OR wr.LoadBatchVolume_yd3 IS NULL
              OR wr.LoadBatchVolume_yd3 <= 0
                THEN NULL
            ELSE
                wr.WaterAdded_gallons * 8.34
                / NULLIF(wr.LoadBatchVolume_yd3, 0)
        END AS WaterAdded_lb_per_yd3,

        CASE
            WHEN wr.WaterAdded_gallons IS NULL
                THEN NULL
            WHEN wr.WaterAdded_gallons > 0
                THEN 1
            ELSE 0
        END AS HasWaterAdded
) AS wm

WHERE
    c.unitSystem = @TargetConcreteUnitSystem
    AND
    (
        @OfficeId IS NULL
        OR p.officeId = @OfficeId
    )
    AND NOT EXISTS
    (
        SELECT 1
        FROM @ExcludedOfficeIds AS excludedOffice
        WHERE excludedOffice.OfficeId = p.officeId
    )
    AND
    (
        @StartCastDate IS NULL
        OR c.castDate >= @StartCastDate
    )
    AND
    (
        @EndCastDate IS NULL
        OR c.castDate < DATEADD(DAY, 1, @EndCastDate)
    );

/* ============================================================================
   Result set 1: Field-Core model extract

   Default behavior:
       - selected concrete-test unit system only
       - valid cast dates only
       - standard-cured 28-day target required
       - at least one effective field measurement required

   Keep the audit/raw columns in the CSV, but the first model should use only
   the recommended model features listed in the header.
   ============================================================================ */
SELECT
    *
FROM #FinalData
WHERE
    (
        @IncludeInvalidCastDates = 1
        OR IsValidCastDate = 1
    )
    AND
    (
        @Require28DayActual = 0
        OR AverageActualStrength28_psi IS NOT NULL
    )
    AND
    (
        @RequireAnyFieldCoreMeasurement = 0
        OR HasAnyEffectiveFreshConcreteMeasurement = 1
    )
ORDER BY
    CastDate,
    testId;

/* ============================================================================
   Result set 2: extraction, quality and feature-coverage summary
   ============================================================================ */
SELECT
    COUNT_BIG(*) AS ConcreteTestCount,
    COUNT(DISTINCT officeId) AS OfficeCount,
    COUNT(DISTINCT projectId) AS ProjectCount,
    COUNT(DISTINCT SampleId) AS SampleCount,
    COUNT(DISTINCT SupplierName) AS SupplierCount,

    MIN(ConcreteTestUnitSystem) AS ConcreteTestUnitSystem,

    SUM(CASE WHEN IsValidCastDate = 1 THEN 1 ELSE 0 END)
        AS TestsWithValidCastDate,

    SUM(CASE WHEN IsValidCastDate = 0 THEN 1 ELSE 0 END)
        AS TestsWithInvalidOrMissingCastDate,

    SUM(CASE WHEN RequiredStrength28 IS NOT NULL THEN 1 ELSE 0 END)
        AS TestsWithRequiredStrength28,

    SUM(CASE WHEN ApplicableSpecifiedStrength28 IS NOT NULL THEN 1 ELSE 0 END)
        AS TestsWithApplicableSpecifiedStrength28,

    SUM(CASE WHEN AverageActualStrength7_psi IS NOT NULL THEN 1 ELSE 0 END)
        AS TestsWithActualStrength7,

    SUM(CASE WHEN AverageActualStrength28_psi IS NOT NULL THEN 1 ELSE 0 END)
        AS TestsWithActualStrength28,

    SUM(CASE WHEN FailureFlag28 = 1 THEN 1 ELSE 0 END)
        AS FailedStrengthTestCount28,

    SUM(CASE WHEN EffectiveSlump_in IS NOT NULL THEN 1 ELSE 0 END)
        AS TestsWithEffectiveSlump,

    SUM(CASE WHEN EffectiveSpread_in IS NOT NULL THEN 1 ELSE 0 END)
        AS TestsWithEffectiveSpread,

    SUM(CASE WHEN EffectiveAir_percent IS NOT NULL THEN 1 ELSE 0 END)
        AS TestsWithEffectiveAir,

    SUM(CASE WHEN EffectiveUnitWeight_lb_ft3 IS NOT NULL THEN 1 ELSE 0 END)
        AS TestsWithEffectiveUnitWeight,

    SUM(CASE WHEN EffectiveConcreteTemp_F IS NOT NULL THEN 1 ELSE 0 END)
        AS TestsWithEffectiveConcreteTemperature,

    SUM(CASE WHEN AmbientTemp_F IS NOT NULL THEN 1 ELSE 0 END)
        AS TestsWithAmbientTemperature,

    SUM(CASE WHEN WaterAdded_gal_per_yd3 IS NOT NULL THEN 1 ELSE 0 END)
        AS TestsWithNormalizedWaterAdded,

    SUM(CASE WHEN HasAnyAfterSPMeasurement = 1 THEN 1 ELSE 0 END)
        AS TestsWithAnyAfterSPMeasurement,

    SUM(CASE WHEN InitialCuringConditionMissing = 0 THEN 1 ELSE 0 END)
        AS TestsWithInitialCuringConditionText,

    SUM(CASE WHEN HasAnyEffectiveFreshConcreteMeasurement = 1 THEN 1 ELSE 0 END)
        AS TestsWithAnyEffectiveFreshMeasurement,

    SUM(CASE WHEN IsFieldCoreStrengthCandidate = 1 THEN 1 ELSE 0 END)
        AS FieldCoreStrengthCandidateCount,

    SUM(CASE WHEN IsFieldCoreWithRequiredStrengthCandidate = 1 THEN 1 ELSE 0 END)
        AS FieldCoreWithRequiredStrengthCandidateCount,

    SUM(CASE WHEN HasAnyBatchData = 1 THEN 1 ELSE 0 END)
        AS TestsWithAnyBatchData,

    CAST
    (
        100.0
        * SUM(CASE WHEN AverageActualStrength28_psi IS NOT NULL THEN 1 ELSE 0 END)
        / NULLIF(COUNT_BIG(*), 0)
        AS decimal(10, 2)
    ) AS ActualStrength28CoveragePercent,

    CAST
    (
        100.0
        * SUM(CASE WHEN HasAnyEffectiveFreshConcreteMeasurement = 1 THEN 1 ELSE 0 END)
        / NULLIF(COUNT_BIG(*), 0)
        AS decimal(10, 2)
    ) AS EffectiveFreshMeasurementCoveragePercent,

    CAST
    (
        100.0
        * SUM(CASE WHEN WaterAdded_gal_per_yd3 IS NOT NULL THEN 1 ELSE 0 END)
        / NULLIF(COUNT_BIG(*), 0)
        AS decimal(10, 2)
    ) AS NormalizedWaterAddedCoveragePercent,

    CAST
    (
        100.0
        * SUM(CASE WHEN IsFieldCoreStrengthCandidate = 1 THEN 1 ELSE 0 END)
        / NULLIF(COUNT_BIG(*), 0)
        AS decimal(10, 2)
    ) AS FieldCoreStrengthCandidatePercent,

    CAST
    (
        100.0
        * SUM(CASE WHEN HasAnyBatchData = 1 THEN 1 ELSE 0 END)
        / NULLIF(COUNT_BIG(*), 0)
        AS decimal(10, 2)
    ) AS BatchDataCoveragePercent,

    MIN(CASE WHEN IsValidCastDate = 1 THEN CastDate END)
        AS EarliestValidCastDate,

    MAX(CASE WHEN IsValidCastDate = 1 THEN CastDate END)
        AS LatestValidCastDate,

    MIN(CastDate) AS RawEarliestCastDate,
    MAX(CastDate) AS RawLatestCastDate

FROM #FinalData;
