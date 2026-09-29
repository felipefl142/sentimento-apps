"""Metricas de classificacao (3 classes) - macro-F1 e a metrica primaria."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.transform.text import LABELS


def evaluate(y_true, y_pred, apps=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    out = {"macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro",
                                      zero_division=0)),
           "accuracy": float(accuracy_score(y_true, y_pred)), "n": int(len(y_true)),
           **{f"f1_{c}": float(v) for c, v in zip(LABELS, per_class, strict=True)}}
    if apps is not None:
        apps = np.asarray(apps)
        for a in np.unique(apps):
            m = apps == a
            out[f"macro_f1__{a}"] = float(f1_score(y_true[m], y_pred[m], labels=LABELS,
                                                   average="macro", zero_division=0))
    return out


def confusion(y_true, y_pred) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    return pd.DataFrame(cm, index=[f"real_{c}" for c in LABELS],
                        columns=[f"pred_{c}" for c in LABELS])


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))
