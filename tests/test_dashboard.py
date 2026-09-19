from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.config import OUTPUTS, ROOT


@pytest.mark.skipif(
    not (OUTPUTS / "scoring_manifest.json").exists(),
    reason="Run the pipeline to enable dashboard artifact tests",
)
def test_all_dashboard_pages_render():
    app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=30).run()
    assert not app.exception
    for page in ["Risk Explorer", "Customer Detail", "Model Validation"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, f"Dashboard failed on {page}"
    assert Path(ROOT / "dashboard" / "app.py").exists()
