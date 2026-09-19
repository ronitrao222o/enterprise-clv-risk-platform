<!-- GENERATED_RESULTS_START -->
Measured results from the seeded 5,000-customer run (regenerate with `python -m src.train`).

Test: 2024-12-31 → 2025-12-31; **3,100 active accounts**, **472 churn events**, Cox Harrell C-index **0.606**.

| Model | 12m MAE (USD) | 12m RMSE (USD) | 12m Brier |
|---|---:|---:|---:|
| baseline | 16,250.75 | 31,070.48 | 0.1308 |
| cox_raw | 16,158.42 | 29,813.01 | 0.1289 |
| cox_calibrated | 14,890.68 | 29,644.61 | 0.1266 |

Validation-fitted hazard multiplier: **0.845**. The calibration procedure was fixed before test evaluation; all three variants are reported, including regressions.

These are synthetic-data results, not evidence of real-world performance. The 24-month risk/CLV output is **not** validated by the 12-month test window.
<!-- GENERATED_RESULTS_END -->
