"""Texto: limpeza unica (treino = serving) e tokenizacao para termos/topicos.

- `clean`: entra no silver e DENTRO do modelo pyfunc (idempotente) -> texto livre do
  dashboard passa pelo mesmo caminho. Mantem emojis, pontuacao e caixa (sinal de sentimento).
- `normalize`/`tokens`: so para analises (minusculo, sem acento, sem stopwords).
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

LABELS = ["neg", "neu", "pos"]

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_REPEAT = re.compile(r"(.)\1{3,}")          # "muuuuuito" -> "muuuito" (preserva enfase)
_SPACE = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9]+")


def clean(text: str | None) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFC", str(text))
    t = _URL.sub(" <url> ", t)
    t = _EMAIL.sub(" <email> ", t)
    t = _REPEAT.sub(r"\1\1\1", t)
    return _SPACE.sub(" ", t).strip()


def clean_series(s: pd.Series) -> pd.Series:
    return s.fillna("").map(clean)


def label_from_score(score) -> np.ndarray:
    """1-2 estrelas neg, 3 neu, 4-5 pos."""
    s = np.asarray(score)
    return np.where(s <= 2, "neg", np.where(s == 3, "neu", "pos"))


def n_alnum(text: str) -> int:
    return sum(ch.isalnum() for ch in text)


def normalize(text: str) -> str:
    """Minusculo e sem acento (para casar com a lista de stopwords)."""
    t = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in t if not unicodedata.combining(ch))


def tokens(text: str, stopwords: set[str], min_len: int = 2) -> list[str]:
    return [w for w in _WORD.findall(normalize(text))
            if len(w) >= min_len and w not in stopwords and not w.isdigit()]
