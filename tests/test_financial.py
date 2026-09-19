import numpy as np
import pandas as pd
import pytest

from src.backtesting import realized_value
from src.baseline import baseline_clv, baseline_survival
from src.clv import discounted_clv, restricted_remaining_lifetime
from src.config import load_config


def test_cashflows_have_correct_monthly_units():
    value = discounted_clv(
        np.array([1200]), np.ones((1, 12)), margin=1, annual_expansion=0, annual_discount=0
    )
    assert value[0] == pytest.approx(1200)
    assert discounted_clv(np.array([1200]), np.zeros((1, 12)), 1, 0, 0)[0] == 0


def test_annual_discount_and_margin():
    expected = 100 * 0.7 * sum(1 / 1.12 ** (t / 12) for t in range(1, 13))
    assert discounted_clv(np.array([1200]), np.ones((1, 12)), 0.7, 0, 0.12)[0] == pytest.approx(
        expected
    )
    assert restricted_remaining_lifetime(np.array([[1, 0.5, 0.25]]))[0] == 1.75


@pytest.mark.parametrize("curve", [[[1, 1.1]], [[0.5, 0.9]], [[np.nan, 0.5]], [[1, -0.1]]])
def test_invalid_survival_rejected(curve):
    with pytest.raises(ValueError):
        discounted_clv(np.array([1200]), np.array(curve))


def test_renewal_cliffs_and_baseline_identity():
    f = pd.DataFrame({"months_until_renewal": [3, 12], "ARR": [1200, 1200]})
    a = load_config()["baseline"] | {
        "monthly_background_churn": 0,
        "annual_renewal_rate": 0.8,
        "annual_expansion_rate": 0,
        "annual_discount_rate": 0,
        "contribution_margin": 1,
    }
    s = baseline_survival(f, 24, a)
    assert s[0, 1] == 1
    assert s[0, 2] == 0.8
    assert s[0, 14] == pytest.approx(0.64)
    np.testing.assert_allclose(baseline_clv(f, 24, a), 100 * s.sum(axis=1))


def test_realized_value_zeros_after_churn_and_no_future_evaluation():
    features = pd.DataFrame(
        {"customer_id": ["C1", "C2"], "scoring_date": pd.to_datetime(["2024-12-31"] * 2)}
    )
    monthly = pd.DataFrame(
        {
            "customer_id": ["C1"] * 3,
            "month": pd.to_datetime(["2024-12-31", "2025-01-31", "2025-02-28"]),
            "ARR": [900000, 1200, 2400],
        }
    )
    np.testing.assert_allclose(realized_value(features, monthly, 3, 1, 0, "2025-03-31"), [300, 0])
    with pytest.raises(ValueError, match="beyond"):
        realized_value(features, monthly, 4, 1, 0, "2025-03-31")
