"""Shared deterministic synthetic fixture; production-scale run is also executed manually."""

import pytest

from src.generate_data import generate


@pytest.fixture(scope="session")
def tables():
    return generate(n_customers=500, seed=42)
