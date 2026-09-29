"""Alertas de pico: dias com % negativo (modelo) ou volume anormais por app.

z robusto contra os `window` dias ANTERIORES (mediana e MAD, sem olhar o proprio dia):
    z = (x - mediana) / max(1.4826 * MAD, sqrt(p(1-p)/n), piso)
O termo binomial impede que um dia de pouco volume vire alerta so por ruido amostral (MAD
de 28 pontos subestima o desvio com frequencia).
Negatividade: n >= min_n, z >= z e aumento absoluto >= min_delta. Volume: z de log(n).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAD_K = 1.4826
FLOOR_PCT = 0.01    # piso do desvio robusto do % negativo (serie muito estavel -> z explode)
FLOOR_LOG = 0.05    # idem para log(volume)


def _robust(x: pd.Series, window: int, floor, n: pd.Series | None = None
            ) -> tuple[pd.Series, pd.Series]:
    prev = x.shift(1).rolling(window, min_periods=max(7, window // 2))
    med = prev.median()
    mad = prev.apply(lambda w: np.median(np.abs(w - np.median(w))), raw=True)
    sd = np.maximum(MAD_K * mad, floor)
    if n is not None:  # proporcao: desvio amostral esperado no volume do proprio dia
        sd = np.maximum(sd, np.sqrt(med * (1 - med) / n))
    return med, (x - med) / sd


def score_days(daily: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """sentimento_diario + baseline/z/flags por app (serie completa, para o grafico)."""
    out = []
    for _app, g in daily.sort_values("data").groupby("app", sort=True):
        g = g.copy()
        g["baseline_neg"], g["z_neg"] = _robust(g["pct_neg_modelo"], cfg["window"], FLOOR_PCT,
                                                g["n"])
        g["baseline_log_n"], g["z_volume"] = _robust(np.log1p(g["n"]), cfg["window"], FLOOR_LOG)
        enough = g["n"] >= cfg["min_n"]
        g["alerta_neg"] = (enough & (g["z_neg"] >= cfg["z"])
                           & (g["pct_neg_modelo"] - g["baseline_neg"] >= cfg["min_delta"]))
        g["alerta_volume"] = enough & (g["z_volume"] >= cfg["volume_z"])
        out.append(g)
    cols = ["app", "data", "n", "pct_neg_modelo", "pct_neg_estrelas", "baseline_neg", "z_neg",
            "z_volume", "alerta_neg", "alerta_volume"]
    return pd.concat(out, ignore_index=True)[cols] if out else pd.DataFrame(columns=cols)


def tipo(row) -> str:
    if row.alerta_neg and row.alerta_volume:
        return "negatividade + volume"
    return "negatividade" if row.alerta_neg else "volume"


def context(reviews: pd.DataFrame, day, n_examples: int = 3) -> dict:
    """Versao dominante e exemplos das reviews negativas (estrelas) do dia."""
    neg = reviews[(reviews["data"] == day) & (reviews["label"] == "neg")]
    if neg.empty:
        return {"versao_dominante": None, "versao_share": None, "exemplos": []}
    vc = neg["app_version"].fillna("?").value_counts(normalize=True)
    ex = neg[neg["text"].str.len().between(30, 400)] \
        .sort_values(["thumbs_up", "text"], ascending=[False, True]).head(n_examples)
    return {"versao_dominante": vc.index[0], "versao_share": float(vc.iloc[0]),
            "exemplos": ex["text"].tolist()}
