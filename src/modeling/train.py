"""Treino: candidatos no split temporal -> MLflow (runs aninhados) -> registry + gate.

    python -m src.modeling.train                      # tfidf_lr + xlmr_zeroshot + bertimbau_ft
    python -m src.modeling.train --skip-transformers  # so tfidf_lr (CPU, segundos)

- Metrica primaria: macro-F1 no test (ultimos `test_days` dias, distribuicao real).
- Registra o melhor candidato. Gate: o champion atual e reavaliado no MESMO test; o novo so
  recebe o alias se F1 >= F1_champion - `gate_tolerance`. Sem refit (servido = avaliado).
- Artefato de modelo: tfidf sempre; transformers so do vencedor (centenas de MB cada).
- Saidas: gold/backtest_predictions.parquet (test x candidatos), reports/train_summary.json
"""
from __future__ import annotations

import argparse
import json
import logging
import tempfile
from datetime import UTC, datetime

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

from config import settings
from src.modeling import candidates, metrics, registry
from src.modeling.dataset import load_labeled, temporal_split
from src.modeling.sentiment_model import SentimentModel, to_frame
from src.transform.text import LABELS

log = logging.getLogger("train")


def _log_eval(kind: str, split, proba: np.ndarray) -> tuple[dict, pd.DataFrame]:
    test = split.test
    pred = np.asarray(LABELS)[proba.argmax(1)]
    m = metrics.evaluate(test["label"], pred, test["app"])
    mlflow.log_metrics({f"test_{k}": v for k, v in m.items()})
    with tempfile.TemporaryDirectory() as d:
        metrics.confusion(test["label"], pred).to_csv(f"{d}/confusion_test.csv")
        # erros mais confiantes: ruido de rotulo (estrela x texto) ou falha do modelo
        err = test.assign(pred=pred, conf=proba.max(1))[pred != test["label"].to_numpy()]
        err.nlargest(200, "conf")[["app", "data", "score", "label", "pred", "conf", "text"]] \
            .to_csv(f"{d}/erros_confiantes.csv", index=False)
        mlflow.log_artifacts(d)
    out = test[["app", "review_id", "data", "label", "score"]].assign(kind=kind) \
        .join(to_frame(proba))
    log.info("%s: test macro-F1 %.4f  acc %.4f  (neg %.3f neu %.3f pos %.3f)", kind,
             m["macro_f1"], m["accuracy"], m["f1_neg"], m["f1_neu"], m["f1_pos"])
    return m, out


def log_model(impl, kind: str, batch_size: int, register: bool):
    sample = pd.DataFrame({"text": ["app ótimo", "não funciona"]})
    with tempfile.TemporaryDirectory() as d:
        artifacts = impl.save(d)
        sig = infer_signature(sample, to_frame(impl.predict_proba(sample["text"].tolist())))
        return mlflow.pyfunc.log_model(
            name="model", python_model=SentimentModel(kind, batch_size), artifacts=artifacts,
            signature=sig, input_example=sample, pip_requirements=registry.PIP_REQUIREMENTS,
            registered_model_name=registry.MODEL_NAME if register else None)


def run(kinds=candidates.KINDS, data_dir=None) -> dict:
    cfg = settings.load_model_cfg()
    client = registry.setup()
    df = load_labeled(data_dir)
    split = temporal_split(df, cfg["split"]["test_days"], cfg["split"]["valid_days"])
    if split.train.empty or split.test.empty:
        raise RuntimeError("dados insuficientes para o split temporal")
    log.info("split: train %d (< %s) | valid %d | test %d (>= %s .. %s)", len(split.train),
             split.valid_start, len(split.valid), len(split.test), split.test_start,
             split.data_max)
    bs = cfg["serving"]["batch_size"]
    results, preds = {}, []
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
    with mlflow.start_run(run_name=f"treino-{stamp}") as parent:
        mlflow.log_params({"test_start": str(split.test_start), "data_max": str(split.data_max),
                           "valid_start": str(split.valid_start), "n_train": len(split.train),
                           "n_valid": len(split.valid), "n_test": len(split.test)})
        for kind in kinds:
            with mlflow.start_run(run_name=kind, nested=True) as run:
                mlflow.set_tag("kind", kind)
                impl = candidates.FIT[kind](split, cfg["candidates"][kind])
                proba = impl.predict_proba(split.test["text"].tolist())
                m, out = _log_eval(kind, split, proba)
                preds.append(out)
                if kind not in candidates.TRANSFORMERS:
                    log_model(impl, kind, bs, register=False)
                elif hasattr(impl, "to"):
                    impl.to("cpu")  # libera a GPU para o proximo candidato
                results[kind] = {"run_id": run.info.run_id, "impl": impl, "metrics": m}

        best = max(results, key=lambda k: results[k]["metrics"]["macro_f1"])
        best_f1 = results[best]["metrics"]["macro_f1"]

        champ = registry.champion_version(client)
        champ_f1 = None
        if champ is not None:
            model = mlflow.pyfunc.load_model(f"models:/{registry.MODEL_NAME}@{registry.CHAMPION}")
            champ_pred = model.predict(split.test[["text"]])["pred"].to_numpy()
            champ_f1 = metrics.macro_f1(split.test["label"], champ_pred)
            mlflow.log_metric("champion_atual_test_macro_f1", champ_f1)
            del model
        promote = champ_f1 is None or best_f1 >= champ_f1 - cfg["gate_tolerance"]
        reason = ("primeiro modelo" if champ_f1 is None else
                  f"{best_f1:.4f} vs champion v{champ.version} {champ_f1:.4f} "
                  f"(tol {cfg['gate_tolerance']})")

        impl = results[best]["impl"]
        if best in candidates.TRANSFORMERS and hasattr(impl, "to"):
            from src.modeling.sentiment_model import device
            impl.to(device())
        with mlflow.start_run(run_id=results[best]["run_id"], nested=True):
            info = log_model(impl, best, bs, register=True)
        version = str(info.registered_model_version)
        tags = {"kind": best, "data_max_date": str(split.data_max),
                "test_start": str(split.test_start), "primary_value": f"{best_f1:.6f}",
                "labels": json.dumps(LABELS), "run_id": results[best]["run_id"],
                "promotion": ("promovido: " if promote else "rejeitado: ") + reason}
        for k, v in tags.items():
            client.set_model_version_tag(registry.MODEL_NAME, version, k, v)
        if promote:
            client.set_registered_model_alias(registry.MODEL_NAME, registry.CHAMPION, version)
        mlflow.set_tags({"best": best, "registered_version": version, "promoted": promote})
        log.info("melhor: %s (macro-F1 %.4f) -> v%s %s", best, best_f1, version,
                 tags["promotion"])

    settings.ensure_dirs()
    pd.concat(preds, ignore_index=True).to_parquet(
        settings.GOLD_DIR / "backtest_predictions.parquet", index=False)
    summary = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "parent_run_id": parent.info.run_id,
        "split": {"valid_start": str(split.valid_start), "test_start": str(split.test_start),
                  "data_max": str(split.data_max), "n_train": len(split.train),
                  "n_valid": len(split.valid), "n_test": len(split.test)},
        "candidates": {k: v["metrics"] for k, v in results.items()},
        "best": best, "registered_version": version, "promoted": promote, "reason": reason,
        "champion_before": {"version": str(champ.version), "test_macro_f1": champ_f1}
        if champ is not None else None,
    }
    (settings.REPORTS_DIR / "train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    for noisy in ("httpx", "urllib3", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-transformers", action="store_true")
    ap.add_argument("--only", nargs="+", choices=candidates.KINDS)
    args = ap.parse_args()
    kinds = args.only or [k for k in candidates.KINDS
                          if not (args.skip_transformers and k in candidates.TRANSFORMERS)]
    run(kinds)


if __name__ == "__main__":
    main()
