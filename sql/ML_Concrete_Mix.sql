DECLARE @RequiredStrengthTypeId int = 30010;
DECLARE @DesignStrengthTypeId   int = 30011;

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
                    1.0
                    + CAST(b.moisture AS decimal(18, 6)) / 100.0,
                    0
                )
        END AS SSDWeight_lbs,

        /*
            Water Weight = Batch Weight - SSD Weight
        */
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
                        1.0
                        + CAST(b.moisture AS decimal(18, 6)) / 100.0,
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

    -- Zero-weight rows mean the material slot was not used.
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

        /*
            Combined aggregate moisture:

            Total Water Weight / Total SSD Weight * 100
        */
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

            If duplicate rows exist for the same:
            concreteTestId + days + strengthType,
            MAX returns the conservative/larger value.
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

        /*
            These counts help detect duplicate strength rows.
        */
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
)

--SELECT TOP (1000)
SELECT 

    /* =============================================================
       Office, project, sample and test
       ============================================================= */

    o.name AS OfficeName,
    p.projectNo,
    s.labNo,
    t.testId,
    sp.name AS SupplierName,

    c.castDate,
    c.testSubTypeId,

    /* =============================================================
       Batch, sample and cast times
       ============================================================= */

    c.batchTime,
    c.sampleTime,
    c.finishTime AS castTime,

    /* =============================================================
       Environmental and curing fields
       ============================================================= */

    l2.value AS CloudType,
    l3.value AS PrecipitationType,
    l4.value AS WindType,

    c.initialCuringCondition,
    l5.value AS FinalCure,

    /* =============================================================
       Required concrete test fields
       ============================================================= */

    c.adMixture,
    c.plantNumber,
    c.mixNumber,

    c.uwSlump_actual,
    c.uwSlump_specMin,
    c.uwSlump_specMax,

    c.uwAir_actual,
    c.uwAir_specMin,
    c.uwAir_specMax,

    c.uwConcreteTemp_actual,
    c.uwConcreteTemp_specMin,
    c.uwConcreteTemp_specMax,

    /* =============================================================
       Strength requirements for this specimen's break age
       ============================================================= */

    sb.BreakAge AS SpecifiedBreakAge,

    sb.RequiredStrength,
    sb.DesignStrength,

    /*
        Prefer Required Strength when both Required and Design
        exist for the same break age.
    */
    COALESCE
    (
        sb.RequiredStrength,
        sb.DesignStrength
    ) AS ApplicableSpecifiedStrength,

    CASE
        WHEN sb.RequiredStrength IS NOT NULL
            THEN 'Required'

        WHEN sb.DesignStrength IS NOT NULL
            THEN 'Design'

        ELSE NULL
    END AS ApplicableStrengthType,

    sb.RequiredStrengthRowCount,
    sb.DesignStrengthRowCount,

    /* =============================================================
       Aggregated sand information
       ============================================================= */

    bd.SandBatchWeight_lbs,
    bd.SandSSDWeight_lbs,
    bd.SandWaterWeight_lbs,
    bd.SandMoisture_percent,
    bd.SandComponentCount,

    bd.SandSSDWeight_lbs
        / NULLIF(c.calcYield, 0)
        AS SandSSD_lbs_yd3,

    /* =============================================================
       Aggregated aggregate information
       ============================================================= */

    bd.AggregateBatchWeight_lbs,
    bd.AggregateSSDWeight_lbs,
    bd.AggregateWaterWeight_lbs,
    bd.AggregateMoisture_percent,
    bd.AggregateComponentCount,

    bd.AggregateSSDWeight_lbs
        / NULLIF(c.calcYield, 0)
        AS AggregateSSD_lbs_yd3,

    /* =============================================================
       Original batch input and calculated results
       ============================================================= */

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
        WHEN
            c.cementQuantity
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

    /* =============================================================
       Specimen row
       One concrete test may return multiple rows here.
       That is expected.
       ============================================================= */

    r.id AS SpecimenRowId,
    r.daysToAge,
    r.wasFieldCured,
    r.testedOnDate,

    r.widthDiameter,
    r.heightLength,
    r.calcArea,

    r.resFractureType AS FractureType,
    r.resCapType AS CapType,

    r.testLoad,
    r.calcCompressiveStrength
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

/*
    BatchData has one row per concreteTestId.
    Therefore, batch rows no longer multiply specimen rows.
*/
INNER JOIN BatchData AS bd
    ON bd.concreteTestId = c.id

/*
    FieldConcreteTestRows remains one row per specimen.
*/
INNER JOIN dbo.FieldConcreteTestRows AS r
    ON r.concreteTestId = c.id

/*
    StrengthByAge has one row per:
        concreteTestId + break age

    Therefore the strength row matching the specimen age is attached.
*/
LEFT JOIN StrengthByAge AS sb
    ON sb.concreteTestId = c.id
   AND sb.BreakAge = r.daysToAge

LEFT JOIN dbo.Suppliers AS sp
    ON sp.id = c.supplierId

LEFT JOIN dbo.LocalizedResourceNames AS l
    ON l.resourceId = c.resWaterUnits and l.localeId = 0

LEFT JOIN dbo.LocalizedResourceNames AS l2
    ON l2.resourceId = c.resCloudType and l2.localeId = 0

LEFT JOIN dbo.LocalizedResourceNames AS l3
    ON l3.resourceId = c.resPrecipitationType and l3.localeId = 0

LEFT JOIN dbo.LocalizedResourceNames AS l4
    ON l4.resourceId = c.resWindType and l4.localeId = 0

LEFT JOIN dbo.LocalizedResourceNames AS l5
    ON l5.resourceId = c.resFinalCure and l5.localeId = 0

WHERE p.officeId = 2
  AND r.calcCompressiveStrength IS NOT NULL

ORDER BY
    c.castDate,
    t.testId,
    r.daysToAge,
    r.testedOnDate,
    r.id;

SELECT
    COUNT_BIG(*) AS SpecimenCount,
    COUNT(DISTINCT SpecimenRowId) AS UniqueSpecimenCount,
    COUNT(DISTINCT testId) AS ConcreteTestCount,
    COUNT(DISTINCT labNo) AS SampleCount,
    COUNT(DISTINCT projectNo) AS ProjectCount,
    COUNT(DISTINCT SupplierName) AS SupplierCount,
    MIN(CastDate) AS EarliestCastDate,
    MAX(CastDate) AS LatestCastDate
FROM #FinalData;

select  * from #FinalData;

select *  into OUTFILE 'path/to/file.csv'
FIELDS TERMINATED BY ',' 
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n';
from #FinalData

SELECT
    daysToAge as DaysToAge,
    COUNT(*) AS SpecimenCount,
    COUNT(DISTINCT testId) AS ConcreteTestCount,
    AVG(CAST(CalcCompressiveStrength AS decimal(18, 2)))
        AS AverageStrength
FROM #FinalData
GROUP BY DaysToAge
ORDER BY DaysToAge;


DROP TABLE IF EXISTS #StrengthTarget28;

SELECT
    TestId,

    COUNT(*) AS SpecimenCount28,

    AVG
    (
        CAST(CalcCompressiveStrength AS decimal(18, 2))
    ) AS AverageStrength28,

    MIN
    (
        CAST(CalcCompressiveStrength AS decimal(18, 2))
    ) AS MinimumStrength28,

    MAX
    (
        CAST(CalcCompressiveStrength AS decimal(18, 2))
    ) AS MaximumStrength28,

    STDEV
    (
        CAST(CalcCompressiveStrength AS decimal(18, 2))
    ) AS StrengthStandardDeviation28

INTO #StrengthTarget28

FROM #FinalData

WHERE DaysToAge = 28
  AND CalcCompressiveStrength IS NOT NULL

GROUP BY TestId;

select * from #StrengthTarget28