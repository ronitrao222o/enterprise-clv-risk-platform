"""Warehouse abstraction: local DuckDB by default, optional Snowflake publishing."""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path

import duckdb
import pandas as pd
from dotenv import load_dotenv

from src.config import DATA, ROOT


def identifier(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError("Unsafe SQL identifier.")
    return name


class Warehouse(ABC):
    backend: str

    @abstractmethod
    def write_table(self, name: str, frame: pd.DataFrame) -> None: ...

    @abstractmethod
    def query(self, sql: str) -> pd.DataFrame: ...

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> "Warehouse":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class DuckDBWarehouse(Warehouse):
    backend = "duckdb"

    def __init__(self, path: str | Path = DATA / "warehouse.duckdb") -> None:
        self.connection = duckdb.connect(str(path))

    def write_table(self, name: str, frame: pd.DataFrame) -> None:
        name = identifier(name)
        self.connection.register("incoming_frame", frame)
        try:
            # CREATE OR REPLACE is atomic in DuckDB; an interrupted write retains old data.
            self.connection.execute(
                f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM incoming_frame'
            )
        finally:
            self.connection.unregister("incoming_frame")

    def query(self, sql: str) -> pd.DataFrame:
        return self.connection.execute(sql).df()

    def close(self) -> None:
        self.connection.close()


class SnowflakeWarehouse(Warehouse):
    backend = "snowflake"

    def __init__(self) -> None:
        import snowflake.connector

        keys = ("account", "user", "password", "warehouse", "database", "schema")
        credentials = {k: os.environ[f"SNOWFLAKE_{k.upper()}"] for k in keys}
        self.connection = snowflake.connector.connect(
            **credentials, login_timeout=15, network_timeout=30
        )

    def write_table(self, name: str, frame: pd.DataFrame) -> None:
        from snowflake.connector.pandas_tools import write_pandas

        data = frame.copy()
        data.columns = [identifier(c).upper() for c in data.columns]
        success, _, count, _ = write_pandas(
            self.connection,
            data,
            identifier(name).upper(),
            auto_create_table=True,
            overwrite=True,
            use_logical_type=True,
        )
        if not success or count != len(frame):
            raise RuntimeError(f"Incomplete Snowflake write for {name}.")

    def query(self, sql: str) -> pd.DataFrame:
        with self.connection.cursor() as cursor:
            return cursor.execute(sql).fetch_pandas_all()

    def close(self) -> None:
        self.connection.close()


def get_warehouse(backend: str | None = None, path: Path | str | None = None) -> Warehouse:
    """Auto fallback on absent credentials/import/connect failure; explicit Snowflake fails loudly."""
    load_dotenv(ROOT / ".env")
    backend = backend or os.getenv("WAREHOUSE_BACKEND", "auto")
    if backend not in {"auto", "duckdb", "snowflake"}:
        raise ValueError("WAREHOUSE_BACKEND must be auto, duckdb, or snowflake.")
    keys = ("ACCOUNT", "USER", "PASSWORD", "WAREHOUSE", "DATABASE", "SCHEMA")
    available = all(os.getenv(f"SNOWFLAKE_{key}") for key in keys)
    if backend == "snowflake" and not available:
        raise RuntimeError(
            "Incomplete Snowflake credentials; set .env or use WAREHOUSE_BACKEND=auto."
        )
    if available and backend != "duckdb":
        try:
            return SnowflakeWarehouse()
        except Exception as error:
            if backend == "snowflake":
                raise RuntimeError(
                    "Snowflake connection failed. Check credentials, connector, and network."
                ) from None
            # Exception class only: connector messages may include account/user details.
            logging.warning("Snowflake unavailable (%s); using DuckDB.", type(error).__name__)
    logging.info("Using local DuckDB warehouse.")
    return DuckDBWarehouse(path or DATA / "warehouse.duckdb")
