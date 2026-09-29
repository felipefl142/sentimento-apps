"""Helpers de DuckDB sobre as camadas de dados (Parquet).

Conexao em memoria com views para cada Parquet: nada de arquivo .duckdb com lock, entao
dashboard e pipeline podem rodar ao mesmo tempo.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from config import settings

LAYERS = ("silver", "gold", "monitoring")


def connect(data_dir: Path | str | None = None) -> duckdb.DuckDBPyConnection:
    """Conexao DuckDB com silver/gold/monitoring registrados como schemas de views."""
    data_dir = Path(data_dir or settings.DATA_DIR)
    con = duckdb.connect()
    for layer in LAYERS:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {layer}")
        folder = data_dir / layer
        if not folder.exists():
            continue
        for pq in sorted(folder.glob("*.parquet")):
            con.execute(
                f"CREATE OR REPLACE VIEW {layer}.{pq.stem} AS "
                f"SELECT * FROM read_parquet('{pq.as_posix()}')"
            )
    return con


def query(sql: str, data_dir: Path | str | None = None, **params) -> pd.DataFrame:
    con = connect(data_dir)
    try:
        return con.execute(sql, params or None).fetchdf()
    finally:
        con.close()


def copy_to_parquet(con: duckdb.DuckDBPyConnection, sql: str, path: Path) -> int:
    """Materializa uma query em Parquet e devolve o numero de linhas."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({sql}) TO '{path.as_posix()}' (FORMAT parquet)")
    return con.execute(f"SELECT count(*) FROM read_parquet('{path.as_posix()}')").fetchone()[0]
