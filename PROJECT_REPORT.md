# Enterprise CLV & Customer Survival Analytics Platform

## Project report

**Prepared:** 23 September 2026  
**Repository:** [ronitrao222o/enterprise-clv-risk-platform](https://github.com/ronitrao222o/enterprise-clv-risk-platform)  
**Evidence:** The seeded local run saved in `outputs/metrics.json`, `outputs/test_financial.csv`, `outputs/test_brier.csv`, and `outputs/test_calibration.csv`.  
**Data classification:** Fully synthetic. No real customer data or confidential company information was used.

## Executive summary

This project is an end-to-end customer analytics system for a simulated enterprise storage company. It generates five years of account and installed-array history, predicts time to customer churn, converts survival probabilities into discounted recurring contribution, compares the forecast with a fixed financial baseline, and exposes results through a warehouse table and Streamlit dashboard.

On the locked 2025 test period, the calibrated Cox model lowered 12-month customer-level CLV mean absolute error from **$16,250.75** to **$14,890.68** (**8.37% lower**) and improved the 12-month churn Brier score from **0.1308** to **0.1266**. The test concordance index was **0.606**, indicating modest ranking ability. The model did not win in every segment: the fixed baseline had lower CLV error for Enterprise and Mid-market customers, while the calibrated model improved substantially for Strategic customers. These results show a useful synthetic-data demonstration, not validated business performance on real accounts.

## Business problem and scope

The simulated company earns recurring subscription and support revenue from customers who operate storage arrays. An account can continue paying after replacing an array, so treating every hardware retirement as churn would confuse asset lifecycle with loss of the customer relationship. A single binary churn flag would also hide *when* revenue is at risk. This project estimates a survival curve for each active account and uses its probability of remaining active in each future month to calculate expected contribution.

The financial target is **finite-horizon recurring contribution**. It excludes sunk hardware purchases, future hardware sales, customer acquisition cost, and intervention cost. “Revenue at risk” in the dashboard is current ARR multiplied by 12-month churn probability; it is an exposure measure rather than an observed or expected revenue loss.

## Data and system design

The generator uses seed 42 and covers January 2021 through December 2025. The completed default run produced:

| Source table | Rows | Grain |
|---|---:|---|
| `customers` | 5,000 | One enterprise account |
| `arrays` | 18,241 | One installed asset spell |
| `monthly_account_metrics` | 214,616 | One active account-month |
| `events` | 27,998 | One renewal, churn, replacement, end-of-service, or expansion event |

The simulated risk process links declining utilization, support activity, critical incidents, renewal timing, and weak expansion to churn. It creates a realistic *relationship* among signals without claiming to reproduce an actual vendor's economics. An account that churns has no later monthly metrics. Replacement arrays receive new asset IDs, and scheduled end-of-service dates remain known from installation.

The pipeline runs in three commands:

```bash
python -m src.generate_data
python -m src.train
python -m src.score --backend duckdb
```

`src.features` builds month-end account snapshots; `src.train` fits and evaluates the model; `src.score` writes risk scores, survival curves, account explanations, and dashboard inputs. The score contract includes 3-, 6-, 12-, and 24-month churn risk, predicted CLV, a leading positive risk contribution, a model version, and a scoring timestamp. DuckDB works without cloud credentials; an optional Snowflake adapter publishes the same tables when configured. The four dashboard views cover portfolio overview, risk exploration, customer detail, and model validation.

## Modeling and leakage controls

The rule baseline was saved before fitting a statistical model. Its assumptions are a 90% annual renewal probability, 0.1% monthly background churn, 4% annual expansion, 72% recurring contribution margin, and 10% annual discount rate. Renewal probability drops at each account's next contract anniversary. Both baseline and model forecasts use the same margin, expansion, discounting, and month-end cash-flow convention.

The main statistical model is an L2-penalized Cox proportional hazards model fitted with `lifelines`; a Kaplan–Meier curve describes the training landmark cohort. Numeric features are scaled and categorical values encoded using training information. Features include account tenure, time to renewal, prior renewals, hardware age and capacity, scheduled end-of-service proximity, utilization trend, recent support activity, ARR growth, and expansion history. Account-level explanations are exact contributions to the Cox log relative hazard, rather than causal claims.

Training uses one assigned December 2021 or December 2022 landmark per eligible customer, with outcomes known through December 2023. A December 2023 snapshot and outcomes through December 2024 fit **one** cumulative-hazard calibration factor. The December 2024 snapshot and outcomes through December 2025 form the untouched 12-month test. Every feature table is filtered to information available on or before its scoring date; the generator's latent distress variable and future events are not predictors. Tests verify that changing or removing future rows does not alter historical features.

Calibration transforms survival as `S_calibrated(t) = S_raw(t)^alpha`. The factor fitted on validation is **0.845**. This keeps each curve non-increasing and retains the model's risk ranking. The 24-month score uses this same factor, but only the first 12 months have out-of-time validation.

For competing risks, customer churn ends the **account** relationship. Array replacement and normal end-of-service are nonterminal account events. At the **array** level, churn, replacement, and end-of-service are distinct possible first terminal events. A discrete Aalen–Johansen calculation produces descriptive cumulative-incidence curves for those asset outcomes. It does not reclassify replacements as customer churn.

## Test results

The test includes **3,100 active accounts** at 31 December 2024; **472** churned within the following 12 months (**15.23%** observed churn). Harrell's C-index for Cox risk ranking was **0.606**.

| Forecast | CLV MAE per account | CLV RMSE per account | 12-month churn Brier |
|---|---:|---:|---:|
| Fixed financial baseline | $16,250.75 | $31,070.48 | 0.1308 |
| Raw Cox survival | $16,158.42 | $29,813.01 | 0.1289 |
| Calibrated Cox survival | **$14,890.68** | **$29,644.61** | **0.1266** |

The calibrated model reduced MAE by **8.37%**, RMSE by **4.59%**, and Brier score by **3.20%** relative to the baseline. Observed discounted recurring contribution across the test cohort was **$367.12 million**. The calibrated forecast total was **$365.92 million**, an aggregate shortfall of about **$1.20 million**. Aggregate closeness does not imply low error for an individual account; the per-account MAE remains material.

Segment results show where that improvement comes from:

| Customer segment | Accounts | Baseline MAE | Calibrated Cox MAE | Lower-error method |
|---|---:|---:|---:|---|
| Enterprise | 1,184 | $17,690 | $18,592 | Baseline |
| Mid-market | 1,301 | $6,866 | $7,404 | Baseline |
| Strategic | 615 | $33,333 | $23,601 | Calibrated Cox |

Calibration improved the overall Brier score, but it is **not perfect**. The calibrated model's average 12-month risk was **17.39%**, above the **15.23%** observed churn rate. Probability-bucket results and Wilson intervals are saved in `outputs/test_calibration.csv`. The 3- and 6-month Brier scores are also available in `outputs/test_brier.csv`; the report emphasizes 12 months because that is the complete financial backtest horizon.

The Brier calculation explicitly rejects an evaluation group censored before the requested horizon. The historical validation and test landmarks have complete administrative follow-up to 12 months except for observed churn, so ordinary horizon-specific Brier scoring is appropriate here. A real dataset with earlier loss to follow-up would require censoring-adjusted evaluation.

## Validation and engineering quality

The local end-to-end run regenerated the data, trained and calibrated the model, scored **2,628** accounts active at 31 December 2025, and published tables to DuckDB. The delivered score version is `cox-v1-44f6d95411`. All **29 pytest tests** pass; Ruff and dependency checks pass. Tests cover source keys and exposure, reproducibility, future-data leakage, CLV arithmetic, event classification, calibration, survival scoring, explanation additivity, score schema, warehouse queries, and dashboard page rendering. The GitHub workflow repeats generation, training, scoring, lint, and tests on Python 3.11.

The run is reproducible from `config.json`; model and source fingerprints prevent scoring regenerated data with an older model bundle. Generated datasets, local database files, and fitted model binaries are intentionally excluded from Git, while the README and small measured summaries remain in the repository.

## Limitations and next steps

The data is simulated, so no reported improvement should be presented as a real-world retention or revenue lift. The Cox model assumes proportional hazards and holds account features fixed after the scoring date. Test C-index is modest, subgroup errors vary, and the 24-month output is beyond the 12-month out-of-time evaluation window. One temporal holdout does not establish stability through market or product changes. The current baseline and model also share a fixed expansion assumption instead of forecasting expansion separately.

For a production study, the next steps are to validate the data definitions and event availability timestamps on real accounts; test proportional-hazards assumptions; run multiple rolling-origin backtests with customer-level uncertainty intervals; evaluate 24-month calibration once enough follow-up exists; model expansion and margins separately; and assess whether acting on the scores produces incremental value. Snowflake publishing is implemented but has not been tested against a configured cloud account.

## Reproducing this report

From the repository root, install dependencies from `requirements.txt`, then run the three pipeline commands above and `pytest -q`. `make install`, `make pipeline`, and `make check` provide the same workflow on macOS or Linux. The README documents the full architecture, assumptions, warehouse setup, and dashboard. Detailed measured outputs remain in `outputs/` after a run; the summary tables above were computed from those saved files, not estimated manually.
