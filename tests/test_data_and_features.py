import numpy as np
import pandas as pd
import pytest

from src.backtesting import training_cohort
from src.config import load_config
from src.data_quality import validate_sources
from src.features import MODEL_FEATURES, make_features, make_labels
from src.generate_data import generate


def test_reproducible_generator():
    a, b = generate(30, seed=7), generate(30, seed=7)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])


def test_source_relationships_and_exposures(tables):
    validate_sources(tables, "2025-12-31")
    assert set(tables["events"].event_type) == {
        "renewal",
        "customer_churn",
        "array_replacement",
        "end_of_service",
        "expansion",
    }
    corrupted = {k: v.copy() for k, v in tables.items()}
    corrupted["monthly_account_metrics"] = corrupted["monthly_account_metrics"].iloc[1:]
    with pytest.raises(ValueError, match="Incomplete monthly"):
        validate_sources(corrupted, "2025-12-31")


def test_future_rows_cannot_change_features(tables):
    cutoff = pd.Timestamp("2023-12-31")
    expected = make_features(tables, cutoff)
    changed = {k: v.copy() for k, v in tables.items()}
    monthly = changed["monthly_account_metrics"]
    monthly.loc[
        monthly.month > cutoff,
        ["ARR", "storage_utilization", "expansion_amount", "support_tickets"],
    ] = 999999
    # Future hardware retirement / expansion / churn events must all be invisible.
    changed["events"] = changed["events"].loc[lambda x: x.event_date <= cutoff]
    changed["arrays"] = changed["arrays"].loc[lambda x: x.install_date <= cutoff]
    pd.testing.assert_frame_equal(expected, make_features(changed, cutoff))
    assert not {"duration", "event_observed"}.intersection(MODEL_FEATURES)


def test_truncated_database_matches_full_database_asof(tables):
    cutoff = pd.Timestamp("2022-12-31")
    truncated = {k: v.copy() for k, v in tables.items()}
    for name, column in [
        ("customers", "account_start_date"),
        ("arrays", "install_date"),
        ("events", "event_date"),
        ("monthly_account_metrics", "month"),
    ]:
        truncated[name] = truncated[name].loc[lambda x: x[column] <= cutoff]
    pd.testing.assert_frame_equal(make_features(tables, cutoff), make_features(truncated, cutoff))


def test_exact_monthly_windows_and_renewal(tables):
    features = make_features(tables, "2023-12-31").set_index("customer_id")
    customer = features.index[0]
    history = (
        tables["monthly_account_metrics"]
        .loc[lambda x: (x.customer_id == customer) & (x.month <= "2023-12-31")]
        .sort_values("month")
    )
    assert features.loc[customer, "support_tickets_90d"] == history.tail(3).support_tickets.sum()
    assert features.loc[customer, "support_tickets_30d"] == history.iloc[-1].support_tickets
    slope = np.polyfit(np.arange(6), history.tail(6).storage_utilization, 1)[0]
    assert features.loc[customer, "utilization_6m_slope"] == pytest.approx(slope)
    assert features.months_until_renewal.between(1, 12).all()
    with pytest.raises(ValueError, match="month ends"):
        make_features(tables, "2023-12-15")


def test_training_labels_respect_knowledge_cutoff(tables):
    cohort = training_cohort(tables, load_config())
    assert cohort.customer_id.is_unique
    assert cohort.duration.max() == 24
    truncated = tables["events"].loc[lambda x: x.event_date <= "2023-12-31"]
    without_labels = cohort.drop(columns=["duration", "event_observed"])
    expected = make_labels(without_labels, truncated, "2023-12-31")
    pd.testing.assert_frame_equal(cohort, expected)


def test_replacements_do_not_remove_customer(tables):
    event = tables["events"].loc[lambda x: x.event_type == "array_replacement"].iloc[0]
    f = make_features(tables, event.event_date)
    assert event.customer_id in set(f.customer_id)
    retired = tables["events"].loc[
        lambda x: (
            (x.event_date <= event.event_date)
            & x.event_type.isin(["array_replacement", "end_of_service"])
        )
    ]
    installed = tables["arrays"].loc[
        lambda x: (
            (x.customer_id == event.customer_id)
            & (x.install_date <= event.event_date)
            & ~x.array_id.isin(retired.array_id)
        )
    ]
    assert f.set_index("customer_id").loc[event.customer_id, "number_of_arrays"] == len(installed)
