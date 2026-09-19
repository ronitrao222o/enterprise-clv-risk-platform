"""Frozen financial assumptions, established before any model evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.clv import discounted_clv


def baseline_survival(features: pd.DataFrame, horizon: int, assumptions: dict) -> np.ndarray:
    """Annual renewal cliff on each account's anniversary plus background attrition."""
    if horizon < 1:
        raise ValueError("Horizon must be positive.")
    renewal, churn = assumptions["annual_renewal_rate"], assumptions["monthly_background_churn"]
    if not 0 <= renewal <= 1 or not 0 <= churn <= 1:
        raise ValueError("Retention and churn rates must be probabilities.")
    t = np.arange(1, horizon + 1)[None, :]
    until = features.months_until_renewal.to_numpy()[:, None]
    renewals = np.maximum(0, (t - until) // 12 + 1)
    return (1 - churn) ** t * renewal**renewals


def baseline_clv(features: pd.DataFrame, horizon: int, assumptions: dict) -> np.ndarray:
    return discounted_clv(
        features.ARR.to_numpy(),
        baseline_survival(features, horizon, assumptions),
        assumptions["contribution_margin"],
        assumptions["annual_expansion_rate"],
        assumptions["annual_discount_rate"],
    )
