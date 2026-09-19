import numpy as np
import pandas as pd
import pytest

from src.backtesting import evaluate, training_cohort
from src.calibration import HazardCalibrator, horizon_brier
from src.competing_risks import array_outcomes, classify_event, cumulative_incidence
from src.config import load_config
from src.features import make_features, make_labels
from src.score import SCORE_COLUMNS, score_accounts, validate_scores
from src.survival import SurvivalModel


@pytest.mark.parametrize(
    "event,code",
    [
        ("customer_churn", 1),
        ("array_replacement", 2),
        ("end_of_service", 3),
        ("renewal", 0),
        ("expansion", 0),
    ],
)
def test_event_classification(event, code):
    assert classify_event(event) == code


def test_competing_risk_probability_mass_with_ties():
    outcomes = pd.DataFrame({"duration": [1, 1, 2, 2], "event_code": [1, 2, 3, 0]})
    cif = cumulative_incidence(outcomes)
    np.testing.assert_allclose(
        cif[["event_free", "customer_churn", "array_replacement", "end_of_service"]].sum(axis=1), 1
    )
    np.testing.assert_allclose(
        cif.iloc[-1][
            ["customer_churn", "array_replacement", "end_of_service", "event_free"]
        ].to_numpy(dtype=float),
        [0.25] * 4,
    )
    with pytest.raises(ValueError):
        classify_event("unknown")


def test_first_terminal_array_event(tables):
    outcomes = array_outcomes(tables, "2025-12-31")
    assert outcomes.array_id.is_unique
    event = tables["events"].loc[lambda x: x.event_type == "array_replacement"].iloc[0]
    assert outcomes.set_index("array_id").loc[event.array_id, "event_code"] == 2


def test_brier_refuses_early_censoring():
    labeled = pd.DataFrame({"duration": [3, 12], "event_observed": [0, 1]})
    with pytest.raises(ValueError, match="IPCW"):
        horizon_brier(labeled, np.array([0.1, 0.2]), 12)


def test_calibration_monotonicity_and_complete_followup():
    raw = np.tile(np.linspace(0.99, 0.8, 12), (100, 1))
    labels = pd.DataFrame({"duration": [6] * 40 + [12] * 60, "event_observed": [1] * 40 + [0] * 60})
    cal = HazardCalibrator().fit(raw, labels)
    corrected = cal.transform(raw)
    assert np.all(np.diff(corrected, axis=1) <= 0)
    assert 1 - corrected[0, -1] == pytest.approx(0.4, abs=1e-4)
    assert horizon_brier(labels, 1 - corrected[:, -1], 12) < horizon_brier(
        labels, 1 - raw[:, -1], 12
    )


def test_end_to_end_models_scoring_and_explanation(tables):
    config = load_config()
    training = training_cohort(tables, config)
    model = SurvivalModel().fit(training)
    validation = make_features(tables, config["validation_date"])
    test = make_features(tables, config["test_date"])
    val_labels = make_labels(validation, tables["events"], config["validation_end"])
    calibrator = HazardCalibrator().fit(model.predict(validation, 12), val_labels)
    bundle = {
        "model": model,
        "calibrator": calibrator,
        "config": config,
        "version": "test",
        "calibrated_through": config["validation_end"],
    }
    result = evaluate(test, tables, config["test_end"], model, calibrator, config)
    assert 0 <= result["summary"]["c_index"] <= 1
    assert (result["financial"].mae >= 0).all()
    scores, curves, explanations = score_accounts(test, bundle)
    assert list(scores.columns) == SCORE_COLUMNS
    assert len(scores) == len(test)
    assert np.all(np.diff(curves.drop(columns="customer_id").to_numpy(), axis=1) <= 0)
    contributions = (
        explanations.groupby("customer_id").log_hazard_contribution.sum().reindex(test.customer_id)
    )
    np.testing.assert_allclose(
        contributions, model.cox.predict_log_partial_hazard(model.transform(test)), atol=1e-8
    )
    with pytest.raises(ValueError, match="not available"):
        score_accounts(validation, bundle)
    with pytest.raises(ValueError, match="support"):
        model.predict(test, 25)
    invalid = scores.copy()
    invalid.loc[0, "churn_risk_24m"] = -0.1
    with pytest.raises(ValueError):
        validate_scores(invalid)
