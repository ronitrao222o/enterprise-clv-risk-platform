"""Finite-horizon discounted recurring contribution (USD), excluding sunk hardware sales."""

from __future__ import annotations

import numpy as np


def discounted_clv(
    arr: np.ndarray,
    survival: np.ndarray,
    margin: float = 0.72,
    annual_expansion: float = 0.04,
    annual_discount: float = 0.1,
) -> np.ndarray:
    """Sum month-end profit weighted by survival at t=1..H, with annual compounding."""
    arr = np.asarray(arr, dtype=float).reshape(-1)
    survival = np.asarray(survival, dtype=float)
    if survival.ndim != 2 or survival.shape[0] != len(arr) or survival.shape[1] < 1:
        raise ValueError("Survival must have shape (accounts, positive horizon).")
    if not (np.isfinite(arr).all() and np.isfinite(survival).all()):
        raise ValueError("Inputs must be finite.")
    if (arr < 0).any() or (survival < 0).any() or (survival > 1).any():
        raise ValueError("ARR must be nonnegative and probabilities must be in [0, 1].")
    if (np.diff(survival, axis=1) > 1e-8).any():
        raise ValueError("Survival must be non-increasing.")
    if not 0 <= margin <= 1 or annual_expansion <= -1 or annual_discount <= -1:
        raise ValueError("Invalid financial assumptions.")
    months = np.arange(1, survival.shape[1] + 1)
    factor = ((1 + annual_expansion) / (1 + annual_discount)) ** (months / 12)
    return arr * margin / 12 * (survival * factor).sum(axis=1)


def restricted_remaining_lifetime(survival: np.ndarray) -> np.ndarray:
    """Discrete expected active month-ends through H, bounded by H; NOT total lifetime."""
    return np.asarray(survival).sum(axis=1)
