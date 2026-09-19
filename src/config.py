"""Repository-relative configuration and reproducible artifact locations."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load the versioned experiment configuration."""
    return json.loads((path or ROOT / "config.json").read_text())


def setup() -> None:
    for path in (DATA, MODELS, OUTPUTS):
        path.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def month_distance(later: Any, earlier: Any) -> Any:
    """Exact calendar-month difference for month-end observations."""
    return (later.year - earlier.year) * 12 + later.month - earlier.month


def load_tables() -> dict[str, Any]:
    import pandas as pd

    names = ("customers", "arrays", "monthly_account_metrics", "events")
    if not all((DATA / f"{name}.parquet").exists() for name in names):
        raise FileNotFoundError("Missing source data. Run: python -m src.generate_data")
    return {name: pd.read_parquet(DATA / f"{name}.parquet") for name in names}
