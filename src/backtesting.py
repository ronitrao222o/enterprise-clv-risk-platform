"""Historical landmark splits, observed financial outcomes, and honest forecast metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.baseline import baseline_clv, baseline_survival
from src.calibration import HazardCalibrator, calibration_table, horizon_brier
from src.clv import discounted_clv
from src.features import make_features, make_labels
from src.survival import SurvivalModel


def training_cohort(tables: dict[str, pd.DataFrame], config: dict) -> pd.DataFrame:
    """Assign each account one landmark using only ID and acquisition date.

    Crucially, assignment does not depend on surviving until a later landmark.
    Half of eligible early accounts use the first landmark; all others use the
    second. Accounts already churned at their assigned date are outside that risk set.
    """
    first, second = config["train_landmarks"]
    customers = tables["customers"].copy()
    early = (customers.customer_id.str[1:].astype(int) % 2 == 0) & (
        customers.account_start_date <= pd.Timestamp(first)
    )
    early_ids = set(customers.loc[early, "customer_id"])
    a = make_features(tables, first)
    b = make_features(tables, second)
    features = pd.concat(
        [a[a.customer_id.isin(early_ids)], b[~b.customer_id.isin(early_ids)]], ignore_index=True
    )
    return make_labels(features, tables["events"], config["train_observation_end"])


def realized_value(
    features: pd.DataFrame,
    monthly: pd.DataFrame,
    horizon: int,
    margin: float,
    annual_discount: float,
    observation_end: str,
) -> np.ndarray:
    """Observed future recurring contribution; missing post-churn months contribute zero.

    Only invoke with complete monthly source coverage. Missing active-account rows
    represent a data quality failure, checked by source validation before training.
    """
    if (features.scoring_date + pd.offsets.MonthEnd(horizon) > pd.Timestamp(observation_end)).any():
        raise ValueError("Cannot evaluate value beyond the available observation window.")
    joined = monthly.merge(features[["customer_id", "scoring_date"]], on="customer_id", how="inner")
    t = (
        (joined.month.dt.year - joined.scoring_date.dt.year) * 12
        + joined.month.dt.month
        - joined.scoring_date.dt.month
    )
    eligible = (t > 0) & (t <= horizon)
    joined = joined.loc[eligible].copy()
    joined["profit"] = joined.ARR / 12 * margin / (1 + annual_discount) ** (t[eligible] / 12)
    values = joined.groupby("customer_id").profit.sum()
    return features.customer_id.map(values).fillna(0).to_numpy()


def financial_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dimension in ["all", "company_size", "product_family", "acquisition_cohort"]:
        groups = (
            [("All accounts", predictions)]
            if dimension == "all"
            else predictions.groupby(dimension)
        )
        for segment, group in groups:
            for model in ["baseline", "cox_raw", "cox_calibrated"]:
                actual, predicted = group.realized_value, group[f"{model}_clv"]
                rows.append(
                    {
                        "dimension": dimension,
                        "segment": segment,
                        "model": model,
                        "accounts": len(group),
                        "mae": float(mean_absolute_error(actual, predicted)),
                        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
                        "mean_bias": float((predicted - actual).mean()),
                        "actual_total": float(actual.sum()),
                        "forecast_total": float(predicted.sum()),
                    }
                )
    return pd.DataFrame(rows)


def evaluate(
    features: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    observation_end: str,
    model: SurvivalModel,
    calibrator: HazardCalibrator,
    config: dict,
) -> dict:
    """Evaluate a frozen model; this function neither fits nor selects anything."""
    h, a = config["evaluation_horizon_months"], config["baseline"]
    labeled = make_labels(features, tables["events"], observation_end)
    raw = model.predict(features, h)
    calibrated = calibrator.transform(raw)
    baseline = baseline_survival(features, h, a)
    predictions = features[
        ["customer_id", "scoring_date", "company_size", "product_family", "acquisition_cohort"]
    ].copy()
    predictions["realized_value"] = realized_value(
        features,
        tables["monthly_account_metrics"],
        h,
        a["contribution_margin"],
        a["annual_discount_rate"],
        observation_end,
    )
    predictions["baseline_clv"] = baseline_clv(features, h, a)
    for name, curves in [("cox_raw", raw), ("cox_calibrated", calibrated)]:
        predictions[f"{name}_clv"] = discounted_clv(
            features.ARR.to_numpy(),
            curves,
            a["contribution_margin"],
            a["annual_expansion_rate"],
            a["annual_discount_rate"],
        )
    scores, buckets = [], []
    for name, curves in [("baseline", baseline), ("cox_raw", raw), ("cox_calibrated", calibrated)]:
        for t in [3, 6, 12]:
            if t <= h:
                scores.append(
                    {
                        "model": name,
                        "horizon_months": t,
                        "brier_score": horizon_brier(labeled, 1 - curves[:, t - 1], t),
                    }
                )
        table = calibration_table(labeled, 1 - curves[:, h - 1], h)
        table["model"] = name
        buckets.append(table)
    # Concordance uses model risk ranking, not financial value or survival time mean.
    cindex = float(
        concordance_index(
            labeled.duration,
            -model.cox.predict_partial_hazard(model.transform(features)),
            labeled.event_observed,
        )
    )
    summary = {
        "accounts": len(features),
        "churn_events": int(labeled.event_observed.sum()),
        "observed_churn_rate": float(labeled.event_observed.mean()),
        "c_index": cindex,
        "horizon_months": h,
    }
    return {
        "predictions": predictions,
        "financial": financial_metrics(predictions),
        "brier": pd.DataFrame(scores),
        "calibration": pd.concat(buckets, ignore_index=True),
        "summary": summary,
    }
