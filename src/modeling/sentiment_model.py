"""Modelo servido: um `mlflow.pyfunc.PythonModel` unico para todos os candidatos.

Entrada: DataFrame com coluna `text` (ou lista de str). Saida: `p_neg, p_neu, p_pos, pred`.
O texto passa por `clean` aqui dentro (idempotente) -> texto livre do dashboard e reviews do
silver seguem o mesmo caminho. Implementacoes:

- `SklearnImpl`: pipeline sklearn com predict_proba (artefato `sklearn` = joblib)
- `HFImpl`: AutoModelForSequenceClassification (artefato `hf` = diretorio save_pretrained,
  ou `hf_ref` = JSON {model_id, revision} para o zero-shot, que vem do cache do HF Hub)

Nao importa `config` (o pickle do PythonModel referencia este modulo por caminho).
"""
from __future__ import annotations

import json
import os

import mlflow.pyfunc
import numpy as np
import pandas as pd

from src.transform.text import LABELS, clean


def device() -> str:
    import torch
    forced = os.getenv("DEVICE", "")
    if forced:
        return forced
    return "cuda" if torch.cuda.is_available() else "cpu"


def _canon(name: str) -> str:
    """negative/Negative/NEG/neg -> neg (casa id2label de qualquer checkpoint com LABELS)."""
    return str(name).lower()[:3]


class SklearnImpl:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        p = self.pipeline.predict_proba(texts)
        order = [list(self.pipeline.classes_).index(c) for c in LABELS]
        return p[:, order]

    def save(self, folder: str) -> dict:
        import joblib
        path = os.path.join(folder, "pipeline.joblib")
        joblib.dump(self.pipeline, path, compress=3)
        return {"sklearn": path}


class HFImpl:
    def __init__(self, model, tokenizer, max_len: int = 128, batch_size: int = 256,
                 ref: dict | None = None):
        self.model, self.tok = model, tokenizer
        self.max_len, self.batch_size, self.ref = max_len, batch_size, ref
        id2label = {int(k): _canon(v) for k, v in model.config.id2label.items()}
        self.order = [next(i for i, lab in id2label.items() if lab == c) for c in LABELS]

    @classmethod
    def load(cls, name_or_path: str, revision: str | None = None, max_len: int = 128,
             batch_size: int = 256, ref: dict | None = None) -> HFImpl:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name_or_path, revision=revision)
        model = AutoModelForSequenceClassification.from_pretrained(name_or_path,
                                                                   revision=revision)
        model.to(device()).eval()
        return cls(model, tok, max_len, batch_size, ref)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        import torch
        dev = self.model.device
        texts = [t if t else " " for t in texts]
        # ordena por tamanho: lotes com padding minimo (3-5x mais rapido que na ordem original)
        idx = np.argsort([len(t) for t in texts])
        out = np.zeros((len(texts), len(LABELS)), dtype=np.float32)
        use_amp = dev.type == "cuda"
        with torch.inference_mode():
            for s in range(0, len(texts), self.batch_size):
                ids = idx[s:s + self.batch_size]
                enc = self.tok([texts[i] for i in ids], truncation=True, max_length=self.max_len,
                               padding=True, return_tensors="pt").to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    logits = self.model(**enc).logits
                out[ids] = torch.softmax(logits.float(), -1).cpu().numpy()
        return out[:, self.order]

    def to(self, dev: str) -> HFImpl:
        self.model.to(dev)
        return self

    def save(self, folder: str) -> dict:
        if self.ref:  # zero-shot: so a referencia (modelo publico, fixado por revision)
            path = os.path.join(folder, "hf_ref.json")
            with open(path, "w") as f:
                json.dump({**self.ref, "max_len": self.max_len}, f)
            return {"hf_ref": path}
        import torch
        path = os.path.join(folder, "hf")
        dev = self.model.device
        # pesos em bf16 no artefato (metade do tamanho); o modelo em memoria volta a fp32
        self.model.to("cpu", dtype=torch.bfloat16).save_pretrained(path)
        self.model.to(dev, dtype=torch.float32)
        self.tok.save_pretrained(path)
        with open(os.path.join(path, "serving.json"), "w") as f:
            json.dump({"max_len": self.max_len}, f)
        return {"hf": path}


def load_impl(artifacts: dict, batch_size: int = 256):
    if "sklearn" in artifacts:
        import joblib
        return SklearnImpl(joblib.load(artifacts["sklearn"]))
    if "hf_ref" in artifacts:
        with open(artifacts["hf_ref"]) as f:
            ref = json.load(f)
        return HFImpl.load(ref["model_id"], ref.get("revision"), ref.get("max_len", 128),
                           batch_size, ref={"model_id": ref["model_id"],
                                            "revision": ref.get("revision")})
    path = artifacts["hf"]
    with open(os.path.join(path, "serving.json")) as f:
        max_len = json.load(f)["max_len"]
    impl = HFImpl.load(path, max_len=max_len, batch_size=batch_size)
    impl.model.float()  # pesos salvos em bf16; inferencia em autocast (cuda) ou fp32 (cpu)
    return impl


def to_frame(proba: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(proba.astype("float64"), columns=[f"p_{c}" for c in LABELS])
    df["pred"] = np.asarray(LABELS)[proba.argmax(1)]
    return df


def as_texts(model_input) -> list[str]:
    if isinstance(model_input, pd.DataFrame):
        col = "text" if "text" in model_input else model_input.columns[0]
        return model_input[col].fillna("").astype(str).tolist()
    if isinstance(model_input, str):
        return [model_input]
    return [str(t) for t in model_input]


class SentimentModel(mlflow.pyfunc.PythonModel):
    def __init__(self, kind: str, batch_size: int = 256):
        self.kind, self.batch_size = kind, batch_size
        self.impl = None

    def load_context(self, context):
        self.impl = load_impl(context.artifacts, self.batch_size)

    def predict(self, context, model_input, params=None):
        texts = [clean(t) for t in as_texts(model_input)]
        if not texts:
            return to_frame(np.zeros((0, len(LABELS))))
        return to_frame(self.impl.predict_proba(texts))
