"""Exact Cox log-hazard decomposition, including both upward and downward terms."""

from __future__ import annotations

import pandas as pd

from src.survival import SurvivalModel


def explain_accounts(model: SurvivalModel, features: pd.DataFrame) -> pd.DataFrame:
    """No fabricated feature importance: terms sum exactly to the fitted linear predictor."""
    contributions = model.contributions(features).copy()
    contributions["customer_id"] = features.customer_id.to_numpy()
    result = contributions.melt(
        id_vars="customer_id", var_name="feature", value_name="log_hazard_contribution"
    )
    result["direction"] = result.log_hazard_contribution.map(
        lambda value: (
            "Increases risk" if value > 0 else "Decreases risk" if value < 0 else "Neutral"
        )
    )
    return result
