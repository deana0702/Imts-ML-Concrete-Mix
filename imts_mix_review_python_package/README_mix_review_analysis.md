# IMTS Concrete Mix Review Analytics

## What the script does

`concrete_mix_review_analysis.py` reads an Excel or CSV dataset and saves:

- `filtered_test_level_data.csv`
- `mix_performance_summary.csv`
- `mix_review_candidates.csv`
- `candidate_messages.txt`
- `data_quality_summary.csv`
- `mix_review_report.html`
- PNG charts under `charts/`
- Candidate time-trend charts under `charts/candidate_trends/`

The analysis uses:

- Recent-period filtering, default 18 months
- Valid 28-day results
- Strength margin = actual 28-day strength − required strength
- Grouping by supplier, plant, mix number, and required strength
- Mean, median, standard deviation, 5th and 95th percentiles
- Below-required rate
- Configurable review-candidate thresholds

## Install

```bash
pip install -r requirements_mix_review.txt
```

## Run with Excel

```bash
python concrete_mix_review_analysis.py Atlanta-Concrete-Raw.xlsx \
  --output-dir mix_review_output
```

Windows PowerShell can also use one line:

```powershell
python .\concrete_mix_review_analysis.py .\Atlanta-Concrete-Raw.xlsx --output-dir .\mix_review_output
```

## Run with CSV

```bash
python concrete_mix_review_analysis.py Atlanta-Concrete-Raw.csv \
  --output-dir mix_review_output
```

## Use a specific Excel sheet

```bash
python concrete_mix_review_analysis.py Atlanta-Concrete-Raw.xlsx \
  --sheet-name "Sheet1" \
  --output-dir mix_review_output
```

## Change candidate thresholds

```bash
python concrete_mix_review_analysis.py Atlanta-Concrete-Raw.xlsx \
  --output-dir mix_review_output \
  --min-tests 20 \
  --min-mean-margin 900 \
  --min-p05-margin 250 \
  --max-failure-rate 1.0
```

## Use a fixed analysis end date

By default, the script uses the maximum `castDate` in the file. This is useful
for historical datasets. To use a fixed date:

```bash
python concrete_mix_review_analysis.py Atlanta-Concrete-Raw.xlsx \
  --as-of-date 2026-07-22 \
  --output-dir mix_review_output
```

## Notes

- Candidate status is a screening flag, not an instruction to reduce cement.
- Engineering review and trial validation are still required.
- If a true mix revision identifier exists, add it to `GROUP_COLUMNS`.
- Supplier and plant identifiers are preferable to display names when available.
