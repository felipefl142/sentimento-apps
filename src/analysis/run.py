"""Analises sobre silver + gold -> gold/{termos_sentimento, termos_emergentes, topicos,
alertas_diario, alertas}.parquet

    python -m src.analysis.run
"""
from __future__ import annotations

import logging
import time

import pandas as pd

from config import settings
from src.analysis import alertas, termos, topicos
from src.utils import db

log = logging.getLogger("analysis")


def load_reviews(data_dir=None) -> pd.DataFrame:
    df = db.query("""
        SELECT r.app, r.data, r.label, r.text, r.app_version, r.thumbs_up, s.pred
        FROM silver.reviews r LEFT JOIN gold.review_scores s USING (app, review_id)
        WHERE r.n_chars > 0 ORDER BY r.app, r.data
    """, data_dir)
    df["data"] = pd.to_datetime(df["data"])
    return df


def run(data_dir=None) -> dict:
    cfg = settings.load_model_cfg()
    tcfg, pcfg, acfg = cfg["terms"], cfg["topics"], cfg["alerts"]
    sw = settings.load_stopwords()
    reviews = load_reviews(data_dir)
    daily = db.query("SELECT * FROM gold.sentimento_diario", data_dir)
    daily["data"] = pd.to_datetime(daily["data"])
    scored = alertas.score_days(daily, acfg)
    flagged = scored[scored["alerta_neg"] | scored["alerta_volume"]]

    t0 = time.time()
    by_month, emerg, tops, alert_rows = [], [], [], []
    for app, g in reviews.groupby("app", sort=True):
        c = termos.build_corpus(g, sw, tcfg["min_df"])
        by_month.append(termos.by_month(c, app, tcfg["top_n"]))
        end = g["data"].max()
        em = termos.emergentes(c, end, tcfg["recent_days"], tcfg["background_days"],
                               tcfg["top_n"])
        emerg.append(em[em["z"] >= tcfg["min_z"]].assign(app=app, data_ref=end,
                                                         janela=tcfg["recent_days"]))
        recent = g[(g["label"] == "neg") & (g["data"] > end - pd.Timedelta(days=pcfg["days"]))]
        tops.append(topicos.topics(recent, app, sw | set(pcfg.get("extra_stopwords", [])),
                                   pcfg["k"], pcfg["top_terms"], pcfg["min_docs"]))
        for row in flagged[flagged["app"] == app].itertuples():
            em = termos.emergentes(c, row.data, 1, tcfg["background_days"], 10)
            alert_rows.append({"app": app, "data": row.data, "tipo": alertas.tipo(row),
                               "n": int(row.n), "pct_neg": row.pct_neg_modelo,
                               "baseline_neg": row.baseline_neg, "z_neg": row.z_neg,
                               "z_volume": row.z_volume, "termos": ", ".join(em["termo"]),
                               **alertas.context(g, row.data)})
        log.info("%s: %d reviews, %d alertas (%.0fs)", app, len(g),
                 int((flagged["app"] == app).sum()), time.time() - t0)

    gold = settings.GOLD_DIR
    outputs = {
        "termos_sentimento": pd.concat(by_month, ignore_index=True),
        "termos_emergentes": pd.concat(emerg, ignore_index=True),
        "topicos": pd.concat([t for t in tops if len(t)], ignore_index=True)
        if any(len(t) for t in tops) else pd.DataFrame(),
        "alertas_diario": scored,
        "alertas": pd.DataFrame(alert_rows),
    }
    for name, df in outputs.items():
        df.to_parquet(gold / f"{name}.parquet", index=False)
    summary = {k: len(v) for k, v in outputs.items()}
    log.info("analises: %s", summary)
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run()


if __name__ == "__main__":
    main()
