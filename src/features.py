"""Strict as-of features: filter every fact table before aggregating."""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "account_age_months",
    "months_until_renewal",
    "previous_renewals",
    "hardware_age",
    "months_until_end_of_service",
    "number_of_arrays",
    "total_capacity_tb",
    "storage_utilization",
    "data_growth",
    "utilization_3m_change",
    "utilization_6m_slope",
    "support_tickets_30d",
    "support_tickets_90d",
    "critical_incidents_90d",
    "ARR",
    "ARR_growth_12m",
    "expansion_last_12m",
    "number_of_previous_expansions",
]
CATEGORICAL_FEATURES = ["industry", "region", "company_size", "product_family"]
# Declared source-system vocabulary, not discovered from held-out observations.
CATEGORY_LEVELS = [
    ["Financial Services", "Healthcare", "Manufacturing", "Retail", "Technology"],
    ["APAC", "EMEA", "North America"],
    ["Enterprise", "Mid-market", "Strategic"],
    ["Capacity Archive", "Flash Performance", "Hybrid Core", "No active array"],
]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def make_features(tables: dict[str, pd.DataFrame], as_of: str | pd.Timestamp) -> pd.DataFrame:
    """Return one row per active account at month end, with immutable dimensions.

    EOS is a schedule known at install, never a future observed retirement. Account
    status requires a metric row at the landmark; no forward filling dead accounts.
    Missing historical windows use explicit zero-change / available-history rules.
    """
    cutoff = pd.Timestamp(as_of)
    if cutoff != cutoff + pd.offsets.MonthEnd(0):
        raise ValueError("Prediction dates must be month ends.")
    monthly = tables["monthly_account_metrics"]
    monthly = monthly.loc[monthly.month <= cutoff].sort_values(["customer_id", "month"]).copy()
    events = tables["events"].loc[lambda x: x.event_date <= cutoff]
    arrays = tables["arrays"].loc[lambda x: x.install_date <= cutoff].copy()
    retired = events.loc[
        events.event_type.isin(["array_replacement", "end_of_service"]), "array_id"
    ]
    arrays = arrays.loc[~arrays.array_id.isin(retired)].copy()
    current = monthly.loc[monthly.month == cutoff].set_index("customer_id")
    churned = events.loc[events.event_type == "customer_churn", "customer_id"]
    current = current.loc[~current.index.isin(churned)]
    dimensions = (
        tables["customers"].loc[lambda x: x.account_start_date <= cutoff].set_index("customer_id")
    )
    f = current[["ARR", "storage_utilization", "data_growth"]].join(dimensions, how="inner")
    start = f.account_start_date.dt
    f["account_age_months"] = (cutoff.year - start.year) * 12 + cutoff.month - start.month
    f["months_until_renewal"] = 12 - f.account_age_months % 12
    f["scoring_date"] = cutoff
    f["acquisition_cohort"] = f.account_start_date.dt.year.astype(str)

    for kind, name in [
        ("renewal", "previous_renewals"),
        ("expansion", "number_of_previous_expansions"),
    ]:
        counts = events.loc[events.event_type == kind].groupby("customer_id").size()
        f[name] = counts.reindex(f.index, fill_value=0)
    arrays["hardware_age"] = (
        (cutoff.year - arrays.install_date.dt.year) * 12
        + cutoff.month
        - arrays.install_date.dt.month
    )
    arrays["months_until_end_of_service"] = (
        (arrays.end_of_service_date.dt.year - cutoff.year) * 12
        + arrays.end_of_service_date.dt.month
        - cutoff.month
    ).clip(lower=0)
    hardware = arrays.groupby("customer_id").agg(
        number_of_arrays=("array_id", "size"),
        total_capacity_tb=("capacity_tb", "sum"),
        hardware_age=("hardware_age", "mean"),
        months_until_end_of_service=("months_until_end_of_service", "min"),
    )
    # Dominant product is chosen from CURRENT installed capacity, with deterministic tie-breaks.
    dominant = (
        arrays.groupby(["customer_id", "product_family"])
        .capacity_tb.sum()
        .reset_index()
        .sort_values(
            ["customer_id", "capacity_tb", "product_family"], ascending=[True, False, True]
        )
        .drop_duplicates("customer_id")
        .set_index("customer_id")
        .product_family
    )
    f = f.join(hardware).join(dominant)
    f["product_family"] = f.product_family.fillna("No active array")
    f["months_until_end_of_service"] = f.months_until_end_of_service.fillna(120)
    for col in ["number_of_arrays", "total_capacity_tb", "hardware_age"]:
        f[col] = f[col].fillna(0)

    for lookback, name, column in [
        (1, "support_tickets_30d", "support_tickets"),
        (3, "support_tickets_90d", "support_tickets"),
        (3, "critical_incidents_90d", "critical_incidents"),
        (12, "expansion_last_12m", "expansion_amount"),
    ]:
        window = monthly.loc[monthly.month > cutoff - pd.offsets.MonthEnd(lookback)]
        f[name] = window.groupby("customer_id")[column].sum().reindex(f.index, fill_value=0)
    for lag, source, name in [
        (3, "storage_utilization", "utilization_3m_change"),
        (12, "ARR", "ARR_growth_12m"),
    ]:
        old = monthly.loc[monthly.month == cutoff - pd.offsets.MonthEnd(lag)].set_index(
            "customer_id"
        )[source]
        old = old.reindex(f.index)
        delta = f[source] - old
        f[name] = (delta / old if source == "ARR" else delta).fillna(0)
    six = monthly.loc[monthly.month > cutoff - pd.offsets.MonthEnd(6)].copy()
    six["t"] = (six.month.dt.year - cutoff.year) * 12 + six.month.dt.month - cutoff.month
    six["ty"] = six.t * six.storage_utilization
    six["tt"] = six.t**2
    sums = six.groupby("customer_id").agg(
        n=("t", "size"),
        t=("t", "sum"),
        y=("storage_utilization", "sum"),
        ty=("ty", "sum"),
        tt=("tt", "sum"),
    )
    denominator = sums.n * sums.tt - sums.t**2
    slopes = (sums.n * sums.ty - sums.t * sums.y) / denominator.replace(0, np.nan)
    f["utilization_6m_slope"] = slopes.reindex(f.index).fillna(0)
    if not np.isfinite(f[NUMERIC_FEATURES].to_numpy()).all():
        raise ValueError("Non-finite features; check source completeness.")
    return f.reset_index()


def make_labels(
    features: pd.DataFrame, events: pd.DataFrame, observation_end: str | pd.Timestamp
) -> pd.DataFrame:
    """Right-censor targets at a declared knowledge cutoff, independent of feature building."""
    end = pd.Timestamp(observation_end)
    if (features.scoring_date >= end).any():
        raise ValueError("Observation end must follow every landmark.")
    churns = events.loc[(events.event_type == "customer_churn") & (events.event_date <= end)]
    if churns.customer_id.duplicated().any():
        raise ValueError("An account must have at most one terminal churn event.")
    result = features.copy()
    churn = result.customer_id.map(churns.set_index("customer_id").event_date)
    if (churn <= result.scoring_date).any():
        raise ValueError("A churned account cannot enter the landmark risk set.")
    result["event_observed"] = churn.notna().astype(int)
    stop = churn.fillna(end)
    result["duration"] = (
        (stop.dt.year - result.scoring_date.dt.year) * 12
        + stop.dt.month
        - result.scoring_date.dt.month
    ).astype(float)
    return result
