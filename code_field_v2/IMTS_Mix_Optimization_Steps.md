# IMTS Mix Optimization — Maximum Feasible Scope with Current Data

## What this pipeline can do

1. Select tests that contain complete cement, fly ash, W/C, sand, aggregate,
   and 28-day strength data.
   When configured, a blank fly-ash value with a valid cement value is treated
   as zero; confirm that office-level data-entry convention before relying on it.
2. Train batch-to-28-day-strength models without using 7-day strength or actual
   28-day result as features.
3. Optionally train batch-to-slump and batch-to-air surrogate models when those
   field measurements are present.
4. Compare against a Dummy median baseline, then search only historical IMTS
   recipes when the held-out strength model demonstrates predictive value.
5. Retain candidates whose heuristic conservative strength (prediction minus
   held-out P90 absolute error) meets the requested strength.
6. Rank feasible candidates by Portland-cement content or total cementitious
   content.

## Fields used

Core recipe: cement content, fly-ash content, W/C ratio, sand SSD, aggregate
SSD. Optional recipe/production fields: moisture, water added, load volume, and
yield. Targets: actual 28-day strength, actual slump, and actual air. Supplier,
plant, and mix identifiers are retained for audit/filtering rather than treated
as material quantities.

## What this pipeline does not claim

- It does not perform true carbon optimization because IMTS does not contain
  approved supplier/material-specific emission factors.
- It does not perform cost optimization because material prices are absent.
- It does not distinguish cement chemistry/type or all admixture dosages.
- It does not invent arbitrary recipes outside the historical IMTS data range.
- A recommended row is a review candidate, not an approved mix design.

## Run order

1. Edit paths and optimization constraints in `mix_config.py`.
2. Run `python 04_preprocess_mix_optimization_data.py`.
3. Review `mix_preprocessing_summary.json`. If the retained cohort is very
   small, restrict analysis to an office such as Albuquerque with better batch
   coverage.
4. Run `python 05_train_mix_surrogate_models.py`.
5. Review `mix_surrogate_model_comparison.csv`. Do not rely on optimization if
   the strength model does not materially beat the included Dummy baseline on
   the held-out project-grouped test set.
6. Set required strength/specification values in `mix_config.py`.
7. Run `python 06_find_mix_optimization_candidates.py`.
8. Have a concrete/materials engineer review the returned historical recipes.
