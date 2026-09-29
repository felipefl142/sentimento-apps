"""Smoke test do dashboard com os dados reais (pula se o pipeline ainda nao rodou)."""
import pytest

from config import settings

pytestmark = pytest.mark.skipif(
    not (settings.GOLD_DIR / "alertas_diario.parquet").exists()
    or not (settings.REPORTS_DIR / "monitoring.json").exists(),
    reason="rode `make all` antes",
)


def test_dashboard_runs_without_exceptions():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(settings.ROOT / "dashboard" / "app.py"),
                           default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.metric) >= 5
    at.selectbox[0].set_value(at.selectbox[0].options[-1]).run()
    assert not at.exception, [e.value for e in at.exception]
