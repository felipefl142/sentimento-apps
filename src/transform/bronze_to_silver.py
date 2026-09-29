"""bronze -> silver (DuckDB + Parquet).

- `reviews`: versao mais recente de cada (app, review_id); datas em America/Sao_Paulo,
  rotulo pelas estrelas, `text` limpo (src.transform.text.clean). userName fica so no bronze.
- `reviews_edicoes`: reviews que mudaram de nota/texto entre coletas (1a versao x atual).
- `apps_dim`: apps configurados + volume/periodo coberto.
"""
from __future__ import annotations

import logging

import pandas as pd

from config import settings
from src.transform import text as tx
from src.utils import db

log = logging.getLogger("silver")

BRONZE_COLUMNS = {
    "review_id": "VARCHAR", "app": "VARCHAR", "package": "VARCHAR", "user_name": "VARCHAR",
    "content": "VARCHAR", "score": "INTEGER", "thumbs_up": "INTEGER",
    "created_version": "VARCHAR", "app_version": "VARCHAR", "at_utc": "TIMESTAMPTZ",
    "reply_content": "VARCHAR", "replied_at_utc": "TIMESTAMPTZ", "fetched_at": "TIMESTAMPTZ",
}


def bronze_glob() -> str:
    return (settings.BRONZE_DIR / "playstore" / "*" / "*.jsonl.gz").as_posix()


def run() -> dict:
    settings.ensure_dirs()
    apps = settings.load_apps()["apps"]
    con = db.connect()
    con.execute("SET TimeZone = 'UTC'")
    cols = ", ".join(f"'{k}': '{v}'" for k, v in BRONZE_COLUMNS.items())
    con.execute(f"""
        CREATE TEMP VIEW raw AS
        SELECT * FROM read_json('{bronze_glob()}', format = 'newline_delimited',
                                columns = {{{cols}}})
        WHERE score BETWEEN 1 AND 5
    """)
    # versao mais recente: maior `at` (edicao) e, em empate, a coleta mais nova
    latest = con.execute(f"""
        SELECT app, review_id, content, score, thumbs_up, app_version, created_version,
               at_utc, timezone('{settings.TZ}', at_utc) AS at_local,
               CAST(timezone('{settings.TZ}', at_utc) AS DATE) AS data,
               reply_content IS NOT NULL AS has_reply, replied_at_utc, fetched_at
        FROM raw
        QUALIFY row_number() OVER (PARTITION BY app, review_id
                                   ORDER BY at_utc DESC, fetched_at DESC) = 1
    """).fetchdf()
    latest = latest[latest["app"].isin(apps)]
    latest["text"] = tx.clean_series(latest["content"])
    latest["n_chars"] = latest["text"].str.len()
    latest["curto"] = latest["text"].map(tx.n_alnum) < 3
    latest["label"] = tx.label_from_score(latest["score"])
    con.register("latest_df", latest.drop(columns=["content"]))
    n = db.copy_to_parquet(con, "SELECT * FROM latest_df ORDER BY app, at_utc",
                           settings.SILVER_DIR / "reviews.parquet")

    n_ed = db.copy_to_parquet(con, """
        WITH versoes AS (
            SELECT DISTINCT app, review_id, at_utc, score, md5(coalesce(content, '')) AS h
            FROM raw),
        agg AS (
            SELECT app, review_id, count(*) AS n_versoes,
                   arg_min(score, at_utc) AS score_inicial, arg_max(score, at_utc) AS score_atual,
                   min(at_utc) AS primeira_at, max(at_utc) AS ultima_at
            FROM versoes GROUP BY 1, 2)
        SELECT * FROM agg WHERE n_versoes > 1
    """, settings.SILVER_DIR / "reviews_edicoes.parquet")

    meta = latest.groupby("app").agg(n_reviews=("review_id", "size"), data_min=("data", "min"),
                                     data_max=("data", "max")).reset_index()
    dim = [{"app": k, "package": v["package"], "nome": v["nome"]} for k, v in apps.items()]
    con.register("dim_df", pd.DataFrame(dim).merge(meta, on="app", how="left"))
    db.copy_to_parquet(con, "SELECT * FROM dim_df", settings.SILVER_DIR / "apps_dim.parquet")
    con.close()
    out = {"reviews": n, "edicoes": n_ed}
    log.info("silver: %s", out)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run()


if __name__ == "__main__":
    main()
