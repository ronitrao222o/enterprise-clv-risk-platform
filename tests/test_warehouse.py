import pandas as pd
import pytest

from src.config import ROOT
from src.database import DuckDBWarehouse, get_warehouse


def test_duckdb_roundtrip_and_safe_identifiers(tmp_path):
    with DuckDBWarehouse(tmp_path / "test.duckdb") as warehouse:
        expected = pd.DataFrame({"customer_id": ["C1"], "predicted_clv": [100.0]})
        warehouse.write_table("customer_risk_scores", expected)
        pd.testing.assert_frame_equal(
            warehouse.query("SELECT * FROM customer_risk_scores"), expected
        )
        with pytest.raises(ValueError, match="identifier"):
            warehouse.write_table("scores; DROP TABLE customers", expected)


def test_auto_falls_back_without_credentials(monkeypatch, tmp_path):
    for key in ["ACCOUNT", "USER", "PASSWORD", "WAREHOUSE", "DATABASE", "SCHEMA"]:
        monkeypatch.setenv(f"SNOWFLAKE_{key}", "")
    with get_warehouse("auto", tmp_path / "fallback.duckdb") as warehouse:
        assert warehouse.backend == "duckdb"
    with pytest.raises(RuntimeError, match="credentials"):
        get_warehouse("snowflake")


def test_portable_sql_contracts_and_queries(tables):
    with DuckDBWarehouse(":memory:") as warehouse:
        for filename in ["01_sources.sql", "02_features.sql", "03_scores.sql"]:
            warehouse.connection.execute((ROOT / "sql" / filename).read_text())
        for name, frame in tables.items():
            warehouse.write_table(name, frame)
        result = warehouse.query((ROOT / "sql" / "05_asof_feature_example.sql").read_text())
        assert len(result) > 0
        assert (result.support_tickets_90d >= 0).all()
