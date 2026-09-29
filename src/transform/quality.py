"""Checagens de qualidade do silver -> gold/quality_checks.json (exit 1 se alguma falhar).

Falha (bloqueia o pipeline): silver vazio, review_id duplicado, datas no futuro, score fora
de 1-5, fracao de texto vazio alta. Aviso (nao bloqueia): app sem review nova ha N dias.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

import pandas as pd

from config import settings
from src.utils import db

log = logging.getLogger("quality")

MAX_EMPTY_FRAC = 0.2


def run(data_dir=None, today=None) -> dict:
    cfg = settings.load_model_cfg()["monitoring"]
    today = today or datetime.now(UTC).date()
    df = db.query("""
        SELECT app, count(*) AS n, count(*) - count(DISTINCT review_id) AS dup,
               sum((data > CAST($today AS DATE) + 1)::INT) AS futuro,
               sum((score NOT BETWEEN 1 AND 5)::INT) AS score_ruim,
               avg((n_chars = 0)::INT) AS frac_vazio, max(data) AS data_max
        FROM silver.reviews GROUP BY 1 ORDER BY 1
    """, data_dir, today=str(today))
    checks = []

    def add(name, ok, detail, level="falha"):
        checks.append({"check": name, "ok": bool(ok), "nivel": level, "detalhe": detail})

    total = int(df["n"].sum()) if len(df) else 0
    add("silver_nao_vazio", total > 0, f"{total} reviews")
    for r in df.itertuples():
        add(f"{r.app}:sem_duplicatas", r.dup == 0, f"{r.dup} duplicadas")
        add(f"{r.app}:sem_datas_futuras", r.futuro == 0, f"{r.futuro} no futuro")
        add(f"{r.app}:score_valido", r.score_ruim == 0, f"{r.score_ruim} fora de 1-5")
        add(f"{r.app}:texto_vazio", r.frac_vazio <= MAX_EMPTY_FRAC, f"{r.frac_vazio:.1%} vazios")
        lag = (today - pd.Timestamp(r.data_max).date()).days
        add(f"{r.app}:frescor", lag <= cfg["stale_days"], f"ultima review ha {lag} dias",
            level="aviso")
    configured = set(settings.load_apps()["apps"])
    missing = configured - set(df["app"])
    add("apps_presentes", not missing, f"sem dados: {sorted(missing)}" if missing else "todos")
    ok = all(c["ok"] for c in checks if c["nivel"] == "falha")
    out = {"generated_at": datetime.now(UTC).isoformat(timespec="seconds"), "ok": ok,
           "checks": checks}
    settings.GOLD_DIR.mkdir(parents=True, exist_ok=True)
    (settings.GOLD_DIR / "quality_checks.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    for c in checks:
        if not c["ok"]:
            log.warning("%s [%s]: %s", c["check"], c["nivel"], c["detalhe"])
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    out = run()
    log.info("qualidade: %s", "OK" if out["ok"] else "FALHOU")
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
