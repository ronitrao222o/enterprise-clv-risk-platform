"""Validate referential integrity, monthly exposure completeness, and event semantics."""

from __future__ import annotations

import pandas as pd

from src.competing_risks import EVENT_CODES


def validate_sources(tables: dict[str, pd.DataFrame], observation_end: str) -> None:
    c, a, m, e = (tables[k] for k in ("customers", "arrays", "monthly_account_metrics", "events"))
    unknown_events = set(e.event_type.dropna()) - {*EVENT_CODES, "renewal", "expansion"}
    if e.event_type.isna().any() or unknown_events:
        raise ValueError(f"Unknown event types: {sorted(unknown_events)}")
    hardware_event = e.event_type.isin(["array_replacement", "end_of_service"])
    if (hardware_event != e.array_id.notna()).any():
        raise ValueError("Only hardware lifecycle events may reference an array ID.")
    if (
        c.customer_id.duplicated().any()
        or a.array_id.duplicated().any()
        or m.duplicated(["customer_id", "month"]).any()
    ):
        raise ValueError("Duplicate source primary keys.")
    for frame in (a, m, e):
        if not frame.customer_id.isin(c.customer_id).all():
            raise ValueError("Customer foreign-key violation.")
    hardware = e.loc[e.array_id.notna()]
    if not hardware.array_id.isin(a.array_id).all():
        raise ValueError("Array foreign-key violation.")
    owners = hardware.array_id.map(a.set_index("array_id").customer_id)
    if not (owners == hardware.customer_id).all():
        raise ValueError("Array event owner mismatch.")
    churn = e.loc[e.event_type == "customer_churn"]
    if churn.customer_id.duplicated().any():
        raise ValueError("Multiple terminal churns per customer.")
    end = pd.Timestamp(observation_end)
    stop = c.customer_id.map(churn.set_index("customer_id").event_date).fillna(
        end + pd.offsets.MonthEnd(1)
    )
    expected = (
        (stop.dt.year - c.account_start_date.dt.year) * 12
        + stop.dt.month
        - c.account_start_date.dt.month
    )
    observed = c.customer_id.map(m.groupby("customer_id").size()).fillna(0)
    if not (expected == observed).all():
        raise ValueError("Incomplete monthly exposure history.")
    joined = m.merge(c[["customer_id", "account_start_date"]], on="customer_id")
    churn_dates = joined.customer_id.map(churn.set_index("customer_id").event_date)
    if (
        (joined.month < joined.account_start_date)
        | (joined.month > end)
        | (joined.month >= churn_dates)
    ).any():
        raise ValueError("Metrics outside active exposure.")
    if (m.ARR < 0).any() or not m.storage_utilization.between(0, 1).all():
        raise ValueError("Invalid business measurements.")
