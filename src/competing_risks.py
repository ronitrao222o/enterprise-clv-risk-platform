"""Two units of analysis: account churn survival and array first-terminal-event CIF."""

from __future__ import annotations

import numpy as np
import pandas as pd

EVENT_CODES = {"customer_churn": 1, "array_replacement": 2, "end_of_service": 3}


def classify_event(event_type: str) -> int:
    """Hardware retirement never becomes customer churn; other events are nonterminal."""
    if event_type not in {*EVENT_CODES, "renewal", "expansion"}:
        raise ValueError(f"Unknown event type: {event_type}")
    return EVENT_CODES.get(event_type, 0)


def array_outcomes(tables: dict[str, pd.DataFrame], observation_end: str) -> pd.DataFrame:
    """For each installed asset, first of account churn, replacement, EOS, or censoring.

    Replacement is not a competing terminal outcome of an ACCOUNT because the
    customer remains at risk afterwards. It IS terminal for the old asset spell.
    Ties prioritize churn, then replacement; source events use month-end granularity.
    """
    end = pd.Timestamp(observation_end)
    assets = tables["arrays"].loc[lambda x: x.install_date <= end].copy()
    events = tables["events"].loc[lambda x: x.event_date <= end]
    churn = events.loc[events.event_type == "customer_churn"].set_index("customer_id").event_date
    retirement = (
        events.loc[events.event_type.isin(["array_replacement", "end_of_service"])]
        .sort_values("event_date")
        .drop_duplicates("array_id")
        .set_index("array_id")
    )
    assets["churn_date"] = assets.customer_id.map(churn)
    assets["retirement_date"] = assets.array_id.map(retirement.event_date)
    assets["retirement_type"] = assets.array_id.map(retirement.event_type)
    assets["stop_date"] = assets[["churn_date", "retirement_date"]].min(axis=1).fillna(end)
    assets["event_code"] = np.select(
        [
            assets.churn_date.notna() & (assets.churn_date == assets.stop_date),
            assets.retirement_type == "array_replacement",
            assets.retirement_type == "end_of_service",
        ],
        [1, 2, 3],
        default=0,
    )
    assets["duration"] = (
        (assets.stop_date.dt.year - assets.install_date.dt.year) * 12
        + assets.stop_date.dt.month
        - assets.install_date.dt.month
    ).astype(int)
    if (assets.duration < 0).any():
        raise ValueError("Array spells cannot end before installation.")
    # Installations exactly at the administrative cutoff have no follow-up.
    return assets.loc[
        assets.duration > 0, ["array_id", "customer_id", "duration", "event_code", "product_family"]
    ]


def cumulative_incidence(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Discrete Aalen–Johansen product integral, handling monthly ties without jitter.

    F_k(t)=F_k(t-)+S(t-)*d_k(t)/Y(t); S(t)=S(t-)*(1-sum(d_k)/Y).
    Censoring occurs after events at tied month-ends. Descriptive estimates only:
    arrays within an account are correlated, so naive asset-level CIs are omitted.
    """
    survival, cif = 1.0, np.zeros(3)
    rows = [
        {
            "month": 0,
            "event_free": survival,
            "customer_churn": 0.0,
            "array_replacement": 0.0,
            "end_of_service": 0.0,
        }
    ]
    for t in sorted(outcomes.duration.unique()):
        at_risk = int((outcomes.duration >= t).sum())
        d = np.array(
            [((outcomes.duration == t) & (outcomes.event_code == k)).sum() for k in (1, 2, 3)]
        )
        cif += survival * d / at_risk
        survival *= 1 - d.sum() / at_risk
        rows.append(
            dict(
                month=int(t), event_free=survival, **dict(zip(EVENT_CODES, cif.copy(), strict=True))
            )
        )
    return pd.DataFrame(rows)
