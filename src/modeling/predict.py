"""Pontua o silver com o champion -> gold/review_scores.parquet (incremental).

Cada linha guarda a versao do modelo; ao trocar o champion, tudo e repontuado (as linhas da
versao anterior saem). Reviews editadas (texto novo) tambem sao repontuadas: a chave inclui
um hash do texto.
"""
from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

import pandas as pd

from config import settings
from src.modeling import registry
from src.utils import db

log = logging.getLogger("predict")

KEY = ["app", "review_id"]


def run(data_dir=None) -> dict:
    client = registry.setup()
    model, mv = registry.load_champion(client)
    version, kind = str(mv.version), mv.tags.get("kind")
    path = settings.GOLD_DIR / "review_scores.parquet"
    reviews = db.query("SELECT app, review_id, text, md5(text) AS text_hash FROM silver.reviews",
                       data_dir)
    prev = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    if len(prev):
        prev = prev[prev["model_version"] == version]
        prev = prev.merge(reviews[KEY + ["text_hash"]], on=KEY + ["text_hash"], how="inner")
    todo = reviews.merge(prev[KEY + ["text_hash"]], on=KEY + ["text_hash"], how="left",
                         indicator=True) if len(prev) else reviews.assign(_merge="left_only")
    todo = todo[todo["_merge"] == "left_only"].drop(columns="_merge")
    t0 = time.time()
    if len(todo):
        scores = model.predict(todo[["text"]].reset_index(drop=True))
        new = pd.concat([todo[KEY + ["text_hash"]].reset_index(drop=True), scores], axis=1)
        new = new.assign(model_version=version, model_kind=kind,
                         scored_at=datetime.now(UTC).isoformat(timespec="seconds"))
        out = pd.concat([prev, new], ignore_index=True) if len(prev) else new
    else:
        out = prev
    settings.ensure_dirs()
    out.to_parquet(path, index=False)
    summary = {"model_version": version, "kind": kind, "novas": len(todo), "total": len(out),
               "seconds": round(time.time() - t0, 1)}
    log.info("predict: %s", summary)
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run()


if __name__ == "__main__":
    main()
