"""Validation-only, scalar cumulative-hazard recalibration preserving survival order."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import brier_score_loss


@dataclass
class HazardCalibrator:
    """Fit S_cal(t)=S_raw(t)^alpha on validation 12m log loss; freeze before test.

    Only one scalar is estimated, preserving both horizon monotonicity and account
    risk rank. Beyond 12m its transfer is an explicit, unvalidated assumption.
    """

    alpha: float = 1.0
    fitted_horizon: int = 12

    def fit(
        self, survival: np.ndarray, labeled: pd.DataFrame, horizon: int = 12
    ) -> "HazardCalibrator":
        if ((labeled.duration < horizon) & (labeled.event_observed == 0)).any():
            raise ValueError(
                "Calibration requires complete horizon follow-up (or IPCW, not implemented here)."
            )
        s = np.clip(survival[:, horizon - 1], 1e-8, 1 - 1e-8)
        y = ((labeled.event_observed == 1) & (labeled.duration <= horizon)).to_numpy().astype(float)

        def loss(log_alpha: float) -> float:
            risk = np.clip(1 - s ** np.exp(log_alpha), 1e-8, 1 - 1e-8)
            return float(-np.mean(y * np.log(risk) + (1 - y) * np.log1p(-risk)))

        result = minimize_scalar(loss, bounds=(-2, 2), method="bounded")
        if not result.success:
            raise RuntimeError("Hazard recalibration did not converge.")
        self.alpha, self.fitted_horizon = float(np.exp(result.x)), horizon
        return self

    def transform(self, survival: np.ndarray) -> np.ndarray:
        return np.asarray(survival) ** self.alpha


def horizon_brier(labeled: pd.DataFrame, risk: np.ndarray, horizon: int) -> float:
    """Ordinary Brier is valid here because each test landmark has complete H follow-up.

    Refuse earlier right censoring rather than silently labeling censored accounts
    as survivors. Production cohorts with loss to follow-up need IPCW estimation.
    """
    if ((labeled.duration < horizon) & (labeled.event_observed == 0)).any():
        raise ValueError("Incomplete follow-up: use an IPCW Brier estimator.")
    y = (labeled.event_observed == 1) & (labeled.duration <= horizon)
    return float(brier_score_loss(y, risk))


def calibration_table(
    labeled: pd.DataFrame, risk: np.ndarray, horizon: int = 12, bins: int = 8
) -> pd.DataFrame:
    horizon_brier(labeled, risk, horizon)  # also validates censoring support
    y = ((labeled.event_observed == 1) & (labeled.duration <= horizon)).astype(int)
    frame = pd.DataFrame({"predicted": risk, "observed": y.to_numpy()})
    frame["bucket"] = pd.qcut(frame.predicted, q=bins, duplicates="drop")
    if frame.bucket.isna().all():
        frame["bucket"] = "All accounts"
    result = (
        frame.groupby("bucket", observed=True)
        .agg(
            predicted=("predicted", "mean"),
            observed=("observed", "mean"),
            count=("observed", "size"),
        )
        .reset_index()
    )
    # Wilson interval communicates uncertainty in the observed event proportions.
    p, n, z = result.observed, result["count"], 1.96
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    result["lower"], result["upper"] = center - radius, center + radius
    result["bucket"] = result.bucket.astype(str)
    return result
