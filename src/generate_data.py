"""Generate related, longitudinal synthetic enterprise accounts; no real customer data."""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
from scipy.special import expit

from src.config import DATA, load_config, setup


def generate(
    n_customers: int = 5000, seed: int = 42, months: int = 60, start_date: str = "2021-01-31"
) -> dict[str, pd.DataFrame]:
    """Simulate monthly cash-flow exposure and distinct account/hardware events.

    Monthly ARR is recorded only if active at month end, so a churn month has zero
    earned recurring contribution under this deliberately explicit cash convention.
    Hardware EOS dates are known at installation; future retirements are events.
    """
    if n_customers < 10 or months < 36:
        raise ValueError("Use at least 10 customers and 36 months.")
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=months, freq="ME")
    n = n_customers
    ids = np.array([f"C{i:05d}" for i in range(n)])
    starts = rng.integers(0, 19, n)
    starts[rng.random(n) < 0.55] = 0
    # A substantial launch cohort plus new acquisitions over the next 18 months.
    sizes = rng.choice(["Mid-market", "Enterprise", "Strategic"], n, p=[0.45, 0.4, 0.15])
    customers = pd.DataFrame(
        {
            "customer_id": ids,
            "industry": rng.choice(
                ["Financial Services", "Healthcare", "Technology", "Manufacturing", "Retail"], n
            ),
            "region": rng.choice(["North America", "EMEA", "APAC"], n, p=[0.45, 0.32, 0.23]),
            "company_size": sizes,
            "account_start_date": dates[starts],
        }
    )
    array_rows, hardware_events = [], []
    first_eos = np.full(n, 1000)
    for i in range(n):
        for _ in range(int(rng.integers(2, 4))):
            install = starts[i]
            family = rng.choice(
                ["Flash Performance", "Hybrid Core", "Capacity Archive"], p=[0.45, 0.35, 0.2]
            )
            capacity = float(rng.choice([50, 100, 250, 500, 1000]))
            while install < months:
                array_id = f"A{len(array_rows):06d}"
                service_months = int(rng.integers(36, 55))
                eos = install + service_months
                first_eos[i] = min(first_eos[i], eos)
                array_rows.append(
                    {
                        "array_id": array_id,
                        "customer_id": ids[i],
                        "product_family": family,
                        "install_date": dates[install],
                        "capacity_tb": capacity,
                        "end_of_service_date": dates[install] + pd.offsets.MonthEnd(service_months),
                    }
                )
                replace = rng.random() < 0.82
                retirement = eos - int(rng.integers(0, 4)) if replace else eos
                if retirement < months:
                    hardware_events.append(
                        {
                            "customer_id": ids[i],
                            "array_id": array_id,
                            "event_date": dates[retirement],
                            "event_type": "array_replacement" if replace else "end_of_service",
                        }
                    )
                if not replace:
                    break
                install = retirement
                capacity *= 1.5

    distress = rng.beta(1.5, 4, n)
    trend = rng.normal(0, 0.0015, n)
    utilization = np.clip(0.8 - 0.38 * distress + rng.normal(0, 0.05, n), 0.12, 0.95)
    arr = np.select([sizes == "Strategic", sizes == "Enterprise"], [400000, 150000], default=60000)
    arr = arr * rng.lognormal(0, 0.3, n)
    active = np.ones(n, dtype=bool)
    last_expansion = starts.copy()
    churn_idx = np.full(n, months)
    metrics, account_events = [], []
    for t, date in enumerate(dates):
        age = t - starts
        eligible = active & (age >= 0)
        shock = (rng.random(n) < 0.025) * rng.uniform(0.1, 0.35, n)
        distress = np.clip(0.96 * distress + 0.012 + shock + rng.normal(0, 0.015, n), 0, 1)
        previous_util = utilization.copy()
        utilization = np.clip(
            0.75 * utilization + 0.25 * (0.85 - 0.65 * distress) + trend + rng.normal(0, 0.02, n),
            0.08,
            0.98,
        )
        growth = (utilization - previous_util) / previous_util
        tickets = rng.poisson(0.5 + 5 * distress, n)
        incidents = rng.poisson(0.04 + 0.7 * distress, n)
        renewal_due = (age > 0) & (age % 12 == 0)
        approaching_eos = (first_eos - t <= 6) & (first_eos - t > 0)
        hazard = expit(
            -6.5
            + 3.2 * distress
            + 0.10 * tickets
            + 0.35 * incidents
            + 1.0 * renewal_due
            + 0.35 * approaching_eos
            + 0.45 * (t - last_expansion > 12)
            - 1.0 * (sizes == "Strategic")
        )
        churn = eligible & (age >= 3) & (rng.random(n) < hazard)
        churn_idx[churn] = t
        active[churn] = False
        for i in np.flatnonzero(churn):
            account_events.append(
                {
                    "customer_id": ids[i],
                    "array_id": None,
                    "event_date": date,
                    "event_type": "customer_churn",
                }
            )
        living = eligible & ~churn
        expands = living & (rng.random(n) < 0.035 * (1 - distress)) & (age > 2)
        expansion = np.where(expands, arr * rng.uniform(0.025, 0.12, n), 0)
        arr += expansion
        last_expansion[expands] = t
        for event_type, mask in [("renewal", living & renewal_due), ("expansion", expands)]:
            for i in np.flatnonzero(mask):
                account_events.append(
                    {
                        "customer_id": ids[i],
                        "array_id": None,
                        "event_date": date,
                        "event_type": event_type,
                    }
                )
        ix = np.flatnonzero(living)
        metrics.append(
            pd.DataFrame(
                {
                    "customer_id": ids[ix],
                    "month": date,
                    "storage_utilization": utilization[ix],
                    "data_growth": growth[ix],
                    "support_tickets": tickets[ix],
                    "critical_incidents": incidents[ix],
                    "ARR": np.round(arr[ix], 2),
                    "expansion_amount": np.round(expansion[ix], 2),
                }
            )
        )
    # Future scheduled EOS remains visible on installed assets; actual future events do not.
    last_active = {ids[i]: dates[min(churn_idx[i] - 1, months - 1)] for i in range(n)}
    arrays = pd.DataFrame(array_rows)
    arrays = arrays[arrays.install_date <= arrays.customer_id.map(last_active)].reset_index(
        drop=True
    )
    hardware_events = [
        e for e in hardware_events if e["event_date"] <= last_active[e["customer_id"]]
    ]
    events = pd.DataFrame(
        account_events + hardware_events,
        columns=["customer_id", "array_id", "event_date", "event_type"],
    )
    events = events.sort_values(["event_date", "customer_id", "event_type"]).reset_index(drop=True)
    return {
        "customers": customers,
        "arrays": arrays,
        "monthly_account_metrics": pd.concat(metrics, ignore_index=True),
        "events": events,
    }


def main() -> None:
    setup()
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customers", type=int, default=config["n_customers"])
    args = parser.parse_args()
    tables = generate(args.customers, config["seed"], config["months"], config["start_date"])
    for name, frame in tables.items():
        frame.to_parquet(DATA / f"{name}.parquet", index=False)
        logging.info("Generated %s: %s rows", name, f"{len(frame):,}")
    manifest = {
        "seed": config["seed"],
        "synthetic": True,
        "months": config["months"],
        "tables": {k: len(v) for k, v in tables.items()},
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
