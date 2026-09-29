"""silver + gold/review_scores -> agregados de sentimento (gold).

- `sentimento_diario` (app x dia): % neg/neu/pos pelo MODELO e pelas ESTRELAS, net = pos - neg,
  media de estrelas, net_prob = media(p_pos - p_neg). Texto vazio fica de fora.
- `sentimento_versao` (app x app_version, >= 30 reviews): % negativo e estrelas por versao.
"""
from __future__ import annotations

import logging

from config import settings
from src.utils import db

log = logging.getLogger("gold")

MIN_VERSION_N = 30

BASE = """
    SELECT r.app, r.data, r.app_version, r.label, r.score, s.pred, s.p_neg, s.p_pos
    FROM silver.reviews r
    JOIN gold.review_scores s USING (app, review_id)
    WHERE r.n_chars > 0
"""


def run(data_dir=None) -> dict:
    con = db.connect(data_dir)
    con.execute(f"CREATE TEMP VIEW base AS {BASE}")
    n_d = db.copy_to_parquet(con, """
        SELECT app, data, count(*) AS n,
               avg((pred = 'neg')::INT) AS pct_neg_modelo,
               avg((pred = 'neu')::INT) AS pct_neu_modelo,
               avg((pred = 'pos')::INT) AS pct_pos_modelo,
               avg((label = 'neg')::INT) AS pct_neg_estrelas,
               avg((label = 'neu')::INT) AS pct_neu_estrelas,
               avg((label = 'pos')::INT) AS pct_pos_estrelas,
               pct_pos_modelo - pct_neg_modelo AS net_modelo,
               pct_pos_estrelas - pct_neg_estrelas AS net_estrelas,
               avg(p_pos - p_neg) AS net_prob,
               avg(score) AS estrelas_media
        FROM base GROUP BY 1, 2 ORDER BY 1, 2
    """, settings.GOLD_DIR / "sentimento_diario.parquet")
    n_v = db.copy_to_parquet(con, f"""
        SELECT app, app_version, count(*) AS n, min(data) AS data_min, max(data) AS data_max,
               avg((pred = 'neg')::INT) AS pct_neg_modelo,
               avg((label = 'neg')::INT) AS pct_neg_estrelas, avg(score) AS estrelas_media
        FROM base WHERE app_version IS NOT NULL
        GROUP BY 1, 2 HAVING count(*) >= {MIN_VERSION_N} ORDER BY 1, data_min
    """, settings.GOLD_DIR / "sentimento_versao.parquet")
    con.close()
    out = {"sentimento_diario": n_d, "sentimento_versao": n_v}
    log.info("gold: %s", out)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run()


if __name__ == "__main__":
    main()
