"""Candidatos: cada `fit_*` devolve um impl com `predict_proba(textos)` e `save(pasta)`.

- tfidf_lr: TF-IDF palavra (1-2) + char_wb (3-5) -> LogisticRegression balanceada; C no valid
- xlmr_zeroshot: cardiffnlp/twitter-xlm-roberta-base-sentiment (tweets multilingues), sem treino
- bertimbau_ft: fine-tune do BERTimbau base, loop torch proprio (bf16, loss ponderada,
  early stopping pelo macro-F1 do valid)
"""
from __future__ import annotations

import copy
import logging
import time

import mlflow
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src.modeling import metrics
from src.modeling.dataset import Split, cap_per_group
from src.modeling.sentiment_model import HFImpl, SklearnImpl, device
from src.transform.text import LABELS

log = logging.getLogger("candidates")

KINDS = ("tfidf_lr", "xlmr_zeroshot", "bertimbau_ft")
TRANSFORMERS = ("xlmr_zeroshot", "bertimbau_ft")


def _features(cfg: dict) -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                 strip_accents="unicode", max_features=cfg["word_max_features"])),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                 sublinear_tf=True, max_features=cfg["char_max_features"])),
    ])


def fit_tfidf_lr(split: Split, cfg: dict) -> SklearnImpl:
    train = cap_per_group(split.train, cfg["cap_per_app_class"])
    feats = _features(cfg)
    t0 = time.time()
    Xtr = feats.fit_transform(train["text"])
    Xva = feats.transform(split.valid["text"])
    mlflow.log_params({"n_train": len(train), "n_features": Xtr.shape[1],
                       "cap_per_app_class": cfg["cap_per_app_class"]})
    best = (-1.0, None)
    for C in cfg["C_grid"]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
        clf.fit(Xtr, train["label"])
        f1 = metrics.macro_f1(split.valid["label"], clf.predict(Xva))
        mlflow.log_metric(f"valid_macro_f1__C_{C}", f1)
        log.info("tfidf_lr C=%s valid macro-F1 %.4f", C, f1)
        if f1 > best[0]:
            best = (f1, C, clf)
    f1, C, clf = best
    mlflow.log_params({"C": C})
    mlflow.log_metric("valid_macro_f1", f1)
    log.info("tfidf_lr: C=%s (%.0fs)", C, time.time() - t0)
    # feats ja ajustado; o pipeline final reaproveita (sem refit)
    return SklearnImpl(Pipeline([("tfidf", feats), ("clf", clf)]))


def fit_xlmr_zeroshot(split: Split, cfg: dict) -> HFImpl:
    from huggingface_hub import model_info
    try:
        revision = model_info(cfg["model_id"]).sha
    except Exception:  # noqa: BLE001 - offline: usa o que estiver no cache
        revision = None
    impl = HFImpl.load(cfg["model_id"], revision, cfg["max_len"], cfg["batch_size"],
                       ref={"model_id": cfg["model_id"], "revision": revision})
    mlflow.log_params({"model_id": cfg["model_id"], "revision": revision,
                       "max_len": cfg["max_len"]})
    mlflow.log_metric("valid_macro_f1", metrics.macro_f1(
        split.valid["label"], np.asarray(LABELS)[impl.predict_proba(
            split.valid["text"].tolist()).argmax(1)]))
    return impl


def fit_bertimbau_ft(split: Split, cfg: dict) -> HFImpl:
    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    torch.manual_seed(42)
    dev = device()
    train = cap_per_group(split.train, cfg["cap_per_app_class"])
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_id"], num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
        label2id={c: i for i, c in enumerate(LABELS)}).to(dev)
    impl = HFImpl(model, tok, cfg["max_len"], cfg["eval_batch_size"])

    enc = tok(train["text"].tolist(), truncation=True, max_length=cfg["max_len"])
    y = train["label"].map({c: i for i, c in enumerate(LABELS)}).to_numpy()
    rows = [{"input_ids": ids, "attention_mask": am, "labels": int(lab)}
            for ids, am, lab in zip(enc["input_ids"], enc["attention_mask"], y, strict=True)]

    def collate(batch):
        return tok.pad(batch, return_tensors="pt")

    loader = DataLoader(rows, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate)
    counts = np.bincount(y, minlength=len(LABELS))
    weights = torch.tensor(len(y) / (len(LABELS) * np.maximum(counts, 1)), dtype=torch.float32,
                           device=dev)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)],
         "weight_decay": cfg["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=cfg["lr"])
    total = len(loader) * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup"] * total), total)
    mlflow.log_params({k: cfg[k] for k in ("model_id", "max_len", "batch_size", "epochs", "lr",
                                           "warmup", "weight_decay", "cap_per_app_class")}
                      | {"n_train": len(train), "device": dev,
                         "class_weights": [round(float(w), 3) for w in weights]})

    best_f1, best_state, bad, step = -1.0, None, 0, 0
    for epoch in range(cfg["epochs"]):
        model.train()
        t0, run_loss = time.time(), 0.0
        for i, batch in enumerate(loader):
            batch = batch.to(dev)
            labels = batch.pop("labels")
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            run_loss += loss.item()
            step += 1
            if (i + 1) % 100 == 0:
                mlflow.log_metric("train_loss", run_loss / 100, step=step)
                run_loss = 0.0
        model.eval()
        pred = np.asarray(LABELS)[impl.predict_proba(split.valid["text"].tolist()).argmax(1)]
        f1 = metrics.macro_f1(split.valid["label"], pred)
        mlflow.log_metric("valid_macro_f1_epoch", f1, step=epoch + 1)
        log.info("bertimbau_ft epoca %d: valid macro-F1 %.4f (%.0fs)", epoch + 1, f1,
                 time.time() - t0)
        if f1 > best_f1:
            best_f1, bad = f1, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            mlflow.set_tag("best_epoch", epoch + 1)
        else:
            bad += 1
            if bad > cfg["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    mlflow.log_metric("valid_macro_f1", best_f1)
    return impl


FIT = {"tfidf_lr": fit_tfidf_lr, "xlmr_zeroshot": fit_xlmr_zeroshot,
       "bertimbau_ft": fit_bertimbau_ft}


def predict_labels(impl, texts: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    proba = impl.predict_proba(texts.tolist())
    return proba, np.asarray(LABELS)[proba.argmax(1)]
