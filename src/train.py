"""Train, calibrate, and perform a locked temporal backtest; emit auditable artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
from datetime import datetime, timezone

import joblib
import pandas as pd

from src.backtesting import evaluate, training_cohort
from src.baseline import baseline_clv
from src.calibration import HazardCalibrator
from src.competing_risks import array_outcomes, cumulative_incidence
from src.config import DATA, MODELS, OUTPUTS, ROOT, load_config, load_tables, setup
from src.data_quality import validate_sources
from src.features import make_features, make_labels
from src.survival import SurvivalModel


def data_fingerprint() -> str:
    digest = hashlib.sha256()
    for name in sorted(["customers", "arrays", "monthly_account_metrics", "events"]):
        with (DATA / f"{name}.parquet").open("rb") as stream:
            while chunk := stream.read(1 << 20):
                digest.update(chunk)
    return digest.hexdigest()


def render_results(metrics: dict, results: dict) -> str:
    """Generate a Markdown result table from measurements, never hand-entered metrics."""
    summary = metrics["test"]
    lines = [
        "<!-- GENERATED_RESULTS_START -->",
        f"Measured results from the seeded {metrics['table_rows']['customers']:,}-customer run (regenerate with `python -m src.train`).",
        "",
        f"Test: {metrics['config']['test_date']} → {metrics['config']['test_end']}; **{summary['accounts']:,} active accounts**, "
        f"**{summary['churn_events']:,} churn events**, Cox Harrell C-index **{summary['c_index']:.3f}**.",
        "",
        "| Model | 12m MAE (USD) | 12m RMSE (USD) | 12m Brier |",
        "|---|---:|---:|---:|",
    ]
    financial, brier = results["test"]["financial"], results["test"]["brier"]
    for model in ["baseline", "cox_raw", "cox_calibrated"]:
        f = financial[(financial.dimension == "all") & (financial.model == model)].iloc[0]
        b = brier[(brier.model == model) & (brier.horizon_months == 12)].iloc[0]
        lines.append(f"| {model} | {f.mae:,.2f} | {f.rmse:,.2f} | {b.brier_score:.4f} |")
    lines += [
        "",
        f"Validation-fitted hazard multiplier: **{metrics['calibration_alpha']:.3f}**. "
        "The calibration procedure was fixed before test evaluation; all three variants are reported, including regressions.",
        "",
        "These are synthetic-data results, not evidence of real-world performance. The 24-month risk/CLV output is **not** validated by the 12-month test window.",
        "<!-- GENERATED_RESULTS_END -->",
    ]
    return "\n".join(lines)


def main() -> None:
    setup()
    config, tables = load_config(), load_tables()
    config["n_customers"] = len(tables["customers"])
    validate_sources(tables, config["test_end"])
    if not (
        pd.Timestamp(config["train_observation_end"])
        <= pd.Timestamp(config["validation_date"])
        < pd.Timestamp(config["validation_end"])
        <= pd.Timestamp(config["test_date"])
        < pd.Timestamp(config["test_end"])
    ):
        raise ValueError("Training, calibration, and test knowledge cutoffs overlap incorrectly.")
    training = training_cohort(tables, config)
    validation = make_features(tables, config["validation_date"])
    test = make_features(tables, config["test_date"])
    # Save fixed rule assumptions and forecasts BEFORE any statistical fitting.
    (OUTPUTS / "baseline_assumptions.json").write_text(json.dumps(config["baseline"], indent=2))
    fixed_forecasts = []
    for name, features in [("validation", validation), ("test", test)]:
        f = features[["customer_id", "scoring_date"]].copy()
        f["split"] = name
        f["baseline_clv"] = baseline_clv(
            features, config["evaluation_horizon_months"], config["baseline"]
        )
        fixed_forecasts.append(f)
    pd.concat(fixed_forecasts).to_parquet(OUTPUTS / "baseline_forecasts.parquet", index=False)
    logging.info(
        "Training on %s independent account landmarks (%s events)",
        len(training),
        training.event_observed.sum(),
    )
    model = SurvivalModel(config["cox_penalizer"]).fit(training)
    val_labels = make_labels(validation, tables["events"], config["validation_end"])
    calibrator = HazardCalibrator().fit(model.predict(validation, 12), val_labels)
    logging.info("Validation-only hazard calibration alpha=%.4f", calibrator.alpha)
    results, summaries = {}, {}
    for split, features, end in [
        ("validation", validation, config["validation_end"]),
        ("test", test, config["test_end"]),
    ]:
        result = evaluate(features, tables, end, model, calibrator, config)
        results[split], summaries[split] = result, result["summary"]
        for name in ["predictions", "financial", "brier", "calibration"]:
            result[name].to_csv(OUTPUTS / f"{split}_{name}.csv", index=False)
        logging.info("%s: %s", split, result["summary"])
    fingerprint = data_fingerprint()
    source_hash = hashlib.sha256(
        b"".join(p.read_bytes() for p in sorted((ROOT / "src").glob("*.py")))
    ).hexdigest()
    version = (
        "cox-v1-"
        + hashlib.sha256(
            (json.dumps(config, sort_keys=True) + fingerprint + source_hash).encode()
        ).hexdigest()[:10]
    )
    bundle = {
        "model": model,
        "calibrator": calibrator,
        "config": config,
        "version": version,
        "data_fingerprint": fingerprint,
        "calibrated_through": config["validation_end"],
    }
    joblib.dump(bundle, MODELS / "survival_bundle.joblib")
    training.to_parquet(OUTPUTS / "training_landmarks.parquet", index=False)
    model.cox.summary.to_csv(OUTPUTS / "cox_coefficients.csv", index_label="feature")
    model.km.survival_function_.rename_axis("month").reset_index().to_csv(
        OUTPUTS / "kaplan_meier.csv", index=False
    )
    outcomes = array_outcomes(tables, config["test_end"])
    cumulative_incidence(outcomes).to_csv(OUTPUTS / "array_competing_risks.csv", index=False)
    metrics = {
        **summaries,
        "training": {
            "accounts": len(training),
            "events": int(training.event_observed.sum()),
            "max_followup_months": model.max_horizon,
        },
        "calibration_alpha": calibrator.alpha,
        "model_version": version,
        "config": config,
        "table_rows": {k: len(v) for k, v in tables.items()},
    }
    (OUTPUTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_version": version,
        "source_sha256": fingerprint,
        "code_sha256": source_hash,
        "config": config,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scikit-learn", "lifelines", "duckdb", "streamlit"]
        },
    }
    (OUTPUTS / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    report = render_results(metrics, results)
    (OUTPUTS / "results.md").write_text(report + "\n")
    readme = ROOT / "README.md"
    if readme.exists():
        text = readme.read_text()
        start, end = "<!-- GENERATED_RESULTS_START -->", "<!-- GENERATED_RESULTS_END -->"
        if start in text and end in text:
            before, rest = text.split(start, 1)
            _, after = rest.split(end, 1)
            readme.write_text(before + report + after)
    logging.info("Saved model %s and backtest artifacts to outputs/", version)


if __name__ == "__main__":
    main()
