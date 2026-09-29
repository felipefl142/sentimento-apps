"""Topicos das reclamacoes recentes: NMF sobre TF-IDF das reviews negativas (estrelas).

Por app, ultimos `days` dias: k topicos com os termos de maior peso, participacao (fracao das
reviews cujo topico dominante e este) e reviews exemplo (maior peso no topico).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from src.transform.text import tokens


def topics(df: pd.DataFrame, app: str, stopwords: set[str], k: int, top_terms: int,
           min_docs: int, n_examples: int = 3) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    if len(df) < min_docs:
        return pd.DataFrame()
    vec = TfidfVectorizer(tokenizer=lambda t: tokens(t, stopwords), preprocessor=None,
                          lowercase=False, token_pattern=None, ngram_range=(1, 2), min_df=5,
                          max_df=0.5, sublinear_tf=True)
    X = vec.fit_transform(df["text"])
    if X.shape[1] < k * 3:
        return pd.DataFrame()
    nmf = NMF(n_components=k, init="nndsvda", random_state=0, max_iter=400)
    W = nmf.fit_transform(X)
    vocab = vec.get_feature_names_out()
    dominant = np.where(W.max(1) > 0, W.argmax(1), -1)
    rows = []
    for t in range(k):
        terms = vocab[np.argsort(-nmf.components_[t])[:top_terms]]
        # exemplos legiveis: nem muito curtos nem textao
        ok = df["text"].str.len().between(40, 400).to_numpy()
        cand = np.flatnonzero(ok & (dominant == t))
        ex = cand[np.argsort(-W[cand, t])][:n_examples]
        rows.append({"app": app, "topico": t, "rotulo": " · ".join(terms[:3]),
                     "termos": ", ".join(terms), "share": float((dominant == t).mean()),
                     "n_docs": int((dominant == t).sum()), "exemplos": df.loc[ex, "text"].tolist(),
                     "data_min": df["data"].min(), "data_max": df["data"].max()})
    return pd.DataFrame(rows).sort_values("share", ascending=False)
