"""Monitoramento -> reports/monitoring.json + monitoring/live_scores.parquet

- performance ao vivo: a estrela chega junto com o texto -> rotulo imediato. Macro-F1 semanal
  do champion nas reviews POSTERIORES aos dados de treino (`data_max_date` da versao)
- drift de rotulo: PSI da distribuicao neg/neu/pos recente x janela de referencia
- drift de vocabulario: OOV (tokens fora do vocabulario de referencia) e divergencia JS de
  unigramas, recente x referencia; comparados ao mesmo calculo test x referencia (baseline)
- drift de predicao: distribuicao predita x distribuicao das estrelas no periodo recente
- frescor: ultima coleta, ultima review por app, idade dos dados do champion
-> `retrain_recommended` + motivos (lidos por `make daily`)
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd

from config import settings
from src.modeling import metrics, registry
from src.transform.text import LABELS, tokens
from src.utils import db

log = logging.getLogger("monitor")

REF_DAYS = 90


def psi_cat(ref: pd.Series, cur: pd.Series) -> float:
    """PSI entre distribuicoes categoricas (Laplace +1)."""
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    r = ref.value_counts().reindex(LABELS, fill_value=0) + 1
    c = cur.value_counts().reindex(LABELS, fill_value=0) + 1
    r, c = r / r.sum(), c / c.sum()
    return float(np.sum((c - r) * np.log(c / r)))


def js_divergence(p: Counter, q: Counter) -> float:
    """Jensen-Shannon (base 2, 0..1) entre distribuicoes de unigramas."""
    keys = list(set(p) | set(q))
    a = np.array([p.get(k, 0) for k in keys], float)
    b = np.array([q.get(k, 0) for k in keys], float)
    if a.sum() == 0 or b.sum() == 0:
        return float("nan")
    a, b = a / a.sum(), b / b.sum()
    m = (a + b) / 2

    def kl(x, y):
        nz = x > 0
        return float(np.sum(x[nz] * np.log2(x[nz] / y[nz])))
    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


def vocab_stats(ref_texts: pd.Series, cur_texts: pd.Series, sw: set[str]) -> dict:
    ref = Counter(t for x in ref_texts for t in tokens(x, sw))
    cur = Counter(t for x in cur_texts for t in tokens(x, sw))
    vocab = {t for t, n in ref.items() if n >= 2}
    total = sum(cur.values())
    oov = sum(n for t, n in cur.items() if t not in vocab) / total if total else float("nan")
    return {"oov_rate": float(oov), "js": js_divergence(ref, cur)}


def load(data_dir=None) -> pd.DataFrame:
    df = db.query("""
        SELECT r.app, r.data, r.label, r.text, s.pred
        FROM silver.reviews r LEFT JOIN gold.review_scores s USING (app, review_id)
        WHERE r.n_chars > 0
    """, data_dir)
    df["data"] = pd.to_datetime(df["data"]).dt.date
    return df


def live_scores(df: pd.DataFrame, after: date) -> pd.DataFrame:
    live = df[(df["data"] > after) & df["pred"].notna()].copy()
    if live.empty:
        return pd.DataFrame(columns=["app", "semana", "n", "macro_f1", "accuracy"])
    live["semana"] = pd.to_datetime(live["data"]).dt.to_period("W-SUN").dt.start_time
    rows = []
    for (app, wk), g in pd.concat([live, live.assign(app="todos")]).groupby(["app", "semana"]):
        m = metrics.evaluate(g["label"], g["pred"])
        rows.append({"app": app, "semana": wk, "n": m["n"], "macro_f1": m["macro_f1"],
                     "accuracy": m["accuracy"]})
    return pd.DataFrame(rows)


def run(data_dir=None, today: date | None = None) -> dict:
    cfg = settings.load_model_cfg()["monitoring"]
    today = today or datetime.now(UTC).date()
    client = registry.setup()
    mv = registry.champion_version(client)
    if mv is None:
        raise RuntimeError("sem champion - rode `make train`")
    data_max = date.fromisoformat(mv.tags["data_max_date"])
    test_start = date.fromisoformat(mv.tags["test_start"])
    test_f1 = float(mv.tags["primary_value"])
    sw = settings.load_stopwords()
    df = load(data_dir)
    last = max(df["data"])
    recent_start = last - timedelta(days=cfg["recent_days"] - 1)
    ref_mask = (df["data"] < test_start) & (df["data"] >= test_start - timedelta(days=REF_DAYS))
    recent_mask = df["data"] >= recent_start
    test_mask = (df["data"] >= test_start) & (df["data"] <= data_max)

    # --- performance ao vivo
    ls = live_scores(df, data_max)
    settings.MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    ls.to_parquet(settings.MONITORING_DIR / "live_scores.parquet", index=False)
    live = df[(df["data"] > data_max) & df["pred"].notna()]
    live_f1 = metrics.macro_f1(live["label"], live["pred"]) if len(live) else None
    perf = {"test_macro_f1": test_f1, "live_n": int(len(live)), "live_macro_f1": live_f1,
            "live_from": str(data_max + timedelta(days=1)),
            "drop": (test_f1 - live_f1) if live_f1 is not None else None,
            "enough": len(live) >= cfg["min_live_n"]}

    # --- drift por app (rotulo, vocabulario, predicao)
    drift = {}
    for app in sorted(df["app"].unique()):
        a = df["app"] == app
        ref, cur, tst = df[a & ref_mask], df[a & recent_mask], df[a & test_mask]
        v_cur = vocab_stats(ref["text"], cur["text"], sw)
        v_tst = vocab_stats(ref["text"], tst["text"], sw)
        pred_dist = cur["pred"].value_counts(normalize=True).reindex(LABELS, fill_value=0)
        star_dist = cur["label"].value_counts(normalize=True).reindex(LABELS, fill_value=0)
        drift[app] = {
            "n_ref": int(len(ref)), "n_recent": int(len(cur)),
            "label_psi": psi_cat(ref["label"], cur["label"]),
            "label_dist_ref": ref["label"].value_counts(normalize=True)
            .reindex(LABELS, fill_value=0).round(4).to_dict(),
            "label_dist_recent": star_dist.round(4).to_dict(),
            "pred_dist_recent": pred_dist.round(4).to_dict(),
            "pred_vs_stars_tv": float(0.5 * np.abs(pred_dist - star_dist).sum()),
            "oov_rate": v_cur["oov_rate"], "oov_rate_test": v_tst["oov_rate"],
            "js": v_cur["js"], "js_test": v_tst["js"],
        }

    # --- frescor
    ingest = json.loads((settings.BRONZE_DIR / "ingest_log.json").read_text()) \
        if (settings.BRONZE_DIR / "ingest_log.json").exists() else {}
    per_app_last = df.groupby("app")["data"].max()
    fresh = {"last_ingest": ingest.get("playstore", {}).get("finished_at"),
             "last_review": {a: str(d) for a, d in per_app_last.items()},
             "stale_apps": [a for a, d in per_app_last.items()
                            if (today - d).days > cfg["stale_days"]],
             "model_data_age_days": (today - data_max).days}

    reasons = []
    if perf["enough"] and perf["drop"] is not None and perf["drop"] > cfg["f1_drop"]:
        reasons.append(f"macro-F1 ao vivo caiu {perf['drop']:.3f} (> {cfg['f1_drop']})")
    for app, d in drift.items():
        if d["n_recent"] >= 100 and d["label_psi"] > cfg["label_psi"]:
            reasons.append(f"{app}: PSI do rotulo {d['label_psi']:.2f} (> {cfg['label_psi']})")
        if d["n_recent"] >= 100 and d["js"] - d["js_test"] > cfg["js_threshold"]:
            reasons.append(f"{app}: vocabulario mudou (JS {d['js']:.3f} vs {d['js_test']:.3f} no test)")
    if fresh["model_data_age_days"] > cfg["max_model_age_days"]:
        reasons.append(f"dados do champion com {fresh['model_data_age_days']} dias "
                       f"(> {cfg['max_model_age_days']})")
    fresh["retrain_recommended"] = bool(reasons)
    fresh["reasons"] = reasons

    out = {"generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
           "champion": {"version": str(mv.version), "kind": mv.tags.get("kind"),
                        "data_max_date": str(data_max), "test_start": str(test_start)},
           "performance": perf, "drift": drift, "freshness": fresh,
           "window": {"recent_start": str(recent_start), "last": str(last),
                      "ref": f"{test_start - timedelta(days=REF_DAYS)}..{test_start - timedelta(days=1)}"}}
    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (settings.REPORTS_DIR / "monitoring.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str))
    log.info("monitor: F1 ao vivo %s (n=%d) | retreino %s %s",
             f"{live_f1:.4f}" if live_f1 is not None else "-", len(live),
             "RECOMENDADO" if reasons else "nao necessario", reasons)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run()


if __name__ == "__main__":
    main()
