"""Score active accounts and publish the warehouse contract plus dashboard artifacts."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from src.baseline import baseline_clv
from src.clv import discounted_clv, restricted_remaining_lifetime
from src.config import MODELS, OUTPUTS, load_tables, setup
from src.database import get_warehouse
from src.explainability import explain_accounts
from src.features import make_features
from src.train import data_fingerprint

SCORE_COLUMNS = [
    "customer_id",
    "scoring_date",
    "churn_risk_3m",
    "churn_risk_6m",
    "churn_risk_12m",
    "churn_risk_24m",
    "expected_remaining_lifetime",
    "predicted_clv",
    "top_risk_driver",
    "model_version",
    "scored_at",
]


def validate_scores(scores: pd.DataFrame, horizon: int = 24) -> None:
    """Enforce the downstream scoring contract before writing any warehouse table."""
    if list(scores.columns) != SCORE_COLUMNS or scores.empty or scores.isna().any().any():
        raise ValueError("Invalid or incomplete scoring schema.")
    if scores.duplicated(["customer_id", "scoring_date"]).any():
        raise ValueError("Duplicate customer/date scores.")
    probabilities = scores[[f"churn_risk_{t}m" for t in (3, 6, 12, 24)]].to_numpy()
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or (np.diff(probabilities, axis=1) < -1e-8).any()
    ):
        raise ValueError("Invalid risk probabilities.")
    if not np.isfinite(scores[["predicted_clv", "expected_remaining_lifetime"]]).all().all():
        raise ValueError("Non-finite financial outputs.")
    if (scores.predicted_clv < 0).any() or not scores.expected_remaining_lifetime.between(
        0, horizon
    ).all():
        raise ValueError("Invalid financial outputs.")


def score_accounts(
    features: pd.DataFrame, bundle: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if features.empty:
        raise ValueError("No active accounts at this scoring date.")
    if (features.scoring_date < pd.Timestamp(bundle["calibrated_through"])).any():
        raise ValueError(
            "This calibrated model was not available at the requested historical date."
        )
    model, calibrator, config = bundle["model"], bundle["calibrator"], bundle["config"]
    horizon, a = config["clv_horizon_months"], config["baseline"]
    if horizon < 24:
        raise ValueError(
            "The warehouse score contract requires at least 24 months of model support."
        )
    curves = calibrator.transform(model.predict(features, horizon))
    explanation = explain_accounts(model, features)
    positive = explanation[explanation.log_hazard_contribution > 0]
    top = (
        positive.sort_values("log_hazard_contribution", ascending=False)
        .drop_duplicates("customer_id")
        .set_index("customer_id")
        .feature
    )
    scores = features[["customer_id", "scoring_date"]].copy()
    for t in [3, 6, 12, 24]:
        scores[f"churn_risk_{t}m"] = 1 - curves[:, t - 1]
    scores["expected_remaining_lifetime"] = restricted_remaining_lifetime(curves)
    scores["predicted_clv"] = discounted_clv(
        features.ARR.to_numpy(),
        curves,
        a["contribution_margin"],
        a["annual_expansion_rate"],
        a["annual_discount_rate"],
    )
    scores["top_risk_driver"] = scores.customer_id.map(top).fillna("No positive contribution")
    scores["model_version"] = bundle["version"]
    scores["scored_at"] = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    validate_scores(scores, horizon)
    survival = pd.DataFrame(curves, columns=[f"month_{t}" for t in range(1, horizon + 1)])
    survival.insert(0, "customer_id", features.customer_id.to_numpy())
    return scores, survival, explanation


def main() -> None:
    setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="Month-end landmark; default is last observed month.")
    parser.add_argument("--backend", choices=["auto", "duckdb", "snowflake"])
    args = parser.parse_args()
    path = MODELS / "survival_bundle.joblib"
    if not path.exists():
        raise FileNotFoundError("Model missing. Run: python -m src.train")
    bundle = joblib.load(path)  # Trusted local artifact only; never load untrusted pickles.
    if data_fingerprint() != bundle["data_fingerprint"]:
        raise ValueError("Source data changed since training. Retrain before scoring.")
    tables = load_tables()
    date = pd.Timestamp(args.date) if args.date else tables["monthly_account_metrics"].month.max()
    features = make_features(tables, date)
    scores, survival, explanations = score_accounts(features, bundle)
    scores.to_parquet(OUTPUTS / "customer_risk_scores.parquet", index=False)
    features.to_parquet(OUTPUTS / "account_features.parquet", index=False)
    survival.to_parquet(OUTPUTS / "survival_curves.parquet", index=False)
    explanations.to_parquet(OUTPUTS / "risk_explanations.parquet", index=False)
    dashboard = features.merge(scores, on=["customer_id", "scoring_date"], validate="one_to_one")
    dashboard["baseline_clv"] = baseline_clv(
        features, bundle["config"]["clv_horizon_months"], bundle["config"]["baseline"]
    )
    dashboard["revenue_at_risk"] = dashboard.ARR * dashboard.churn_risk_12m
    dashboard.to_parquet(OUTPUTS / "dashboard_accounts.parquet", index=False)
    with get_warehouse(args.backend) as warehouse:
        for name, table in {
            **tables,
            "account_features": features,
            "customer_risk_scores": scores,
        }.items():
            warehouse.write_table(name, table)
        backend = warehouse.backend
        count = warehouse.query("SELECT COUNT(*) AS n FROM customer_risk_scores").iloc[0, 0]
    (OUTPUTS / "scoring_manifest.json").write_text(
        json.dumps(
            {
                "backend": backend,
                "scoring_date": str(date.date()),
                "accounts": int(count),
                "model_version": bundle["version"],
            },
            indent=2,
        )
    )
    logging.info("Scored %s active customers at %s; warehouse=%s", count, date.date(), backend)


if __name__ == "__main__":
    main()
