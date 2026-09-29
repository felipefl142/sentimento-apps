"""Termos distintivos por log-odds com prior de Dirichlet informativo (Monroe et al., 2008).

z-score de cada termo entre dois grupos de textos, com encolhimento pelo corpus de referencia:
termo raro com contagem pequena nao domina o ranking (ao contrario de razao de frequencias).

- neg x pos (estrelas) por app e mes -> "o que distingue reclamacao de elogio"
- emergentes: negativos da janela recente x negativos dos `background_days` anteriores
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from src.transform.text import tokens


@dataclass
class Corpus:
    """Reviews de UM app com a matriz documento x termo (unigramas + bigramas)."""
    df: pd.DataFrame          # data, label, pred, text, app_version, thumbs_up
    X: sparse.csr_matrix
    vocab: np.ndarray


def build_corpus(df: pd.DataFrame, stopwords: set[str], min_df: int = 3) -> Corpus:
    vec = CountVectorizer(tokenizer=lambda t: tokens(t, stopwords), preprocessor=None,
                          lowercase=False, token_pattern=None, ngram_range=(1, 2),
                          min_df=min_df)
    X = vec.fit_transform(df["text"]).tocsr()
    return Corpus(df.reset_index(drop=True), X, vec.get_feature_names_out())


def log_odds_z(ci: np.ndarray, cj: np.ndarray, prior: np.ndarray | None = None) -> np.ndarray:
    """z do log-odds-ratio de i vs j (positivo = mais tipico de i)."""
    ci, cj = np.asarray(ci, float), np.asarray(cj, float)
    a = (ci + cj if prior is None else np.asarray(prior, float)) + 0.01
    a0, ni, nj = a.sum(), ci.sum(), cj.sum()
    li = np.log((ci + a) / (ni + a0 - ci - a))
    lj = np.log((cj + a) / (nj + a0 - cj - a))
    return (li - lj) / np.sqrt(1.0 / (ci + a) + 1.0 / (cj + a))


def distinctive(c: Corpus, mask_i: np.ndarray, mask_j: np.ndarray, top_n: int,
                min_count: int = 5) -> pd.DataFrame:
    """Top termos de i contra j (z desc), exigindo `min_count` ocorrencias em i."""
    if not mask_i.any() or not mask_j.any():
        return pd.DataFrame(columns=["termo", "z", "n"])
    ci = np.asarray(c.X[mask_i].sum(0)).ravel()
    cj = np.asarray(c.X[mask_j].sum(0)).ravel()
    z = log_odds_z(ci, cj)
    ok = np.flatnonzero(ci >= min_count)
    top = ok[np.argsort(-z[ok])][:top_n]
    return pd.DataFrame({"termo": c.vocab[top], "z": z[top], "n": ci[top].astype(int)})


def by_month(c: Corpus, app: str, top_n: int, months: int = 12) -> pd.DataFrame:
    """neg x pos (estrelas) em cada um dos ultimos `months` meses + ultimos 90 dias."""
    d = pd.to_datetime(c.df["data"])
    neg, pos = (c.df["label"] == "neg").to_numpy(), (c.df["label"] == "pos").to_numpy()
    periods = {p: (d.dt.to_period("M") == p).to_numpy()
               for p in sorted(d.dt.to_period("M").unique())[-months:]}
    periods["90d"] = (d > d.max() - pd.Timedelta(days=90)).to_numpy()
    out = []
    for name, m in periods.items():
        for lado, (a, b) in {"neg": (neg, pos), "pos": (pos, neg)}.items():
            t = distinctive(c, m & a, m & b, top_n)
            out.append(t.assign(app=app, periodo=str(name), lado=lado))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def emergentes(c: Corpus, end: date, window: int, background_days: int, top_n: int,
               min_count: int = 3) -> pd.DataFrame:
    """Termos que cresceram nos negativos de [end-window+1, end] vs `background_days` antes."""
    d = pd.to_datetime(c.df["data"])
    neg = (c.df["label"] == "neg").to_numpy()
    end = pd.Timestamp(end)
    start = end - pd.Timedelta(days=window - 1)
    cur = neg & ((d >= start) & (d <= end)).to_numpy()
    ref = neg & ((d < start) & (d >= start - pd.Timedelta(days=background_days))).to_numpy()
    t = distinctive(c, cur, ref, top_n, min_count)
    return t.assign(n_alvo=int(cur.sum()), n_base=int(ref.sum()))
