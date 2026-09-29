"""Dados rotulados do silver e split temporal (nunca aleatorio).

test = ultimos `test_days` dias do silver, valid = `valid_days` antes, train = o resto.
Subamostragem so no TREINO (teto por app x classe); valid/test mantem a distribuicao real.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from src.utils import db

COLUMNS = ["app", "review_id", "data", "text", "label", "score", "app_version", "thumbs_up"]


@dataclass
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    test_start: date
    valid_start: date
    data_max: date


def load_labeled(data_dir=None) -> pd.DataFrame:
    df = db.query(f"SELECT {', '.join(COLUMNS)} FROM silver.reviews WHERE n_chars > 0 "
                  "ORDER BY app, data, review_id", data_dir)
    df["data"] = pd.to_datetime(df["data"]).dt.date
    return df


def temporal_split(df: pd.DataFrame, test_days: int, valid_days: int) -> Split:
    data_max = max(df["data"])
    test_start = data_max - timedelta(days=test_days - 1)
    valid_start = test_start - timedelta(days=valid_days)
    return Split(train=df[df["data"] < valid_start].reset_index(drop=True),
                 valid=df[(df["data"] >= valid_start) & (df["data"] < test_start)]
                 .reset_index(drop=True),
                 test=df[df["data"] >= test_start].reset_index(drop=True),
                 test_start=test_start, valid_start=valid_start, data_max=data_max)


def cap_per_group(df: pd.DataFrame, cap: int, seed: int = 42) -> pd.DataFrame:
    """No maximo `cap` linhas por app x classe (sorteio so nos grupos acima do teto)."""
    parts = [g if len(g) <= cap else g.sample(cap, random_state=seed)
             for _, g in df.groupby(["app", "label"], sort=True)]
    return pd.concat(parts).sort_values(["app", "data", "review_id"]).reset_index(drop=True)
