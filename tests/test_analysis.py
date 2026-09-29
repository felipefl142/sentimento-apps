"""Termos (log-odds), alertas de pico e monitoramento sobre o pipeline sintetico completo."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from config import settings
from src.analysis import run as analysis
from src.analysis.termos import log_odds_z
from src.modeling import predict
from src.monitoring import monitor
from src.transform import build_gold
from tests.conftest import SPIKE_DAY


@pytest.fixture
def analyzed(trained):
    predict.run()
    build_gold.run()
    analysis.run()
    return trained


def test_log_odds_z_sinal_e_encolhimento():
    ci = np.array([50, 5, 1000])
    cj = np.array([5, 50, 1000])
    z = log_odds_z(ci, cj)
    assert z[0] > 3 and z[1] < -3 and abs(z[2]) < 1
    # mesma razao com contagens pequenas -> z menor (encolhimento)
    assert log_odds_z(np.array([5, 100]), np.array([1, 100]))[0] < z[0]


def test_alerta_no_pico_e_nao_no_ruido(analyzed):
    al = pd.read_parquet(settings.GOLD_DIR / "alertas.parquet")
    alpha = al[al["app"] == "alpha"]
    assert pd.Timestamp(SPIKE_DAY) in set(pd.to_datetime(alpha["data"]))
    row = alpha[pd.to_datetime(alpha["data"]) == pd.Timestamp(SPIKE_DAY)].iloc[0]
    assert "pix" in row["termos"] and row["tipo"] == "negatividade + volume"
    assert row["versao_dominante"] == "2.0" and len(row["exemplos"]) > 0
    assert (al["app"] == "beta").sum() == 0, al[al["app"] == "beta"][["data", "tipo", "n", "pct_neg", "baseline_neg", "z_neg", "z_volume"]].to_string()
    assert len(alpha) <= 2


def test_termos_e_topicos(analyzed):
    ts = pd.read_parquet(settings.GOLD_DIR / "termos_sentimento.parquet")
    neg90 = ts[(ts["app"] == "beta") & (ts["periodo"] == "90d") & (ts["lado"] == "neg")]
    pos90 = ts[(ts["app"] == "beta") & (ts["periodo"] == "90d") & (ts["lado"] == "pos")]
    assert {"pessimo", "horrivel"} & set(neg90["termo"])
    assert {"excelente", "adorei"} & set(pos90["termo"])
    top = pd.read_parquet(settings.GOLD_DIR / "topicos.parquet")
    assert set(top["app"]) == {"alpha", "beta"}
    # share = fracao com esse topico dominante; reviews so com stopwords ficam sem topico
    share = top.groupby("app")["share"].sum()
    assert ((share > 0.8) & (share <= 1.0 + 1e-9)).all(), share


def test_monitor(analyzed):
    out = monitor.run(today=date(2026, 7, 1))
    assert out["performance"]["test_macro_f1"] > 0.9
    assert out["performance"]["live_n"] == 0       # champion viu ate o ultimo dia
    assert set(out["drift"]) == {"alpha", "beta"}
    d = out["drift"]["beta"]
    assert d["label_psi"] < 0.1 and d["pred_vs_stars_tv"] < 0.05
    assert not out["freshness"]["retrain_recommended"], out["freshness"]["reasons"]
    assert (settings.REPORTS_DIR / "monitoring.json").exists()
