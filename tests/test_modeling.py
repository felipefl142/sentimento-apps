"""Split temporal, treino/registro do tfidf_lr, gate do champion, pyfunc, predict incremental."""
import numpy as np
import pandas as pd

from config import settings
from src.modeling import predict, registry, train
from src.modeling.dataset import cap_per_group, load_labeled, temporal_split


def test_split_temporal_sem_vazamento(silver):
    df = load_labeled()
    s = temporal_split(df, test_days=30, valid_days=30)
    assert max(s.train["data"]) < min(s.valid["data"]) <= max(s.valid["data"]) < min(s.test["data"])
    assert min(s.test["data"]) == s.test_start and max(s.test["data"]) == s.data_max
    assert len(s.train) + len(s.valid) + len(s.test) == len(df)
    capped = cap_per_group(s.train, 100)
    assert capped.groupby(["app", "label"]).size().max() <= 100


def test_treino_registra_champion(trained):
    _, summary = trained
    m = summary["candidates"]["tfidf_lr"]
    assert m["macro_f1"] > 0.9            # textos sinteticos sao separaveis
    assert summary["promoted"] and summary["best"] == "tfidf_lr"
    client = registry.setup()
    mv = registry.champion_version(client)
    assert str(mv.version) == summary["registered_version"]
    assert mv.tags["kind"] == "tfidf_lr" and mv.tags["data_max_date"] == "2026-06-30"
    bt = pd.read_parquet(settings.GOLD_DIR / "backtest_predictions.parquet")
    assert set(bt["kind"]) == {"tfidf_lr"} and bt["data"].min() >= pd.Timestamp("2026-06-01").date()


def test_pyfunc_probabilidades(trained):
    model, _ = registry.load_champion(registry.setup())
    out = model.predict(pd.DataFrame({"text": ["péssimo, não funciona", "adorei, muito bom", ""]}))
    assert list(out.columns) == ["p_neg", "p_neu", "p_pos", "pred"]
    np.testing.assert_allclose(out[["p_neg", "p_neu", "p_pos"]].sum(1), 1.0, atol=1e-6)
    assert out["pred"].iloc[0] == "neg" and out["pred"].iloc[1] == "pos"


def test_gate_rejeita_candidato_pior(trained, monkeypatch):
    _, first = trained
    cfg = settings.load_model_cfg()
    cfg["gate_tolerance"] = -1.0          # exige ganho impossivel -> rejeita
    monkeypatch.setattr(settings, "load_model_cfg", lambda: cfg)
    second = train.run(kinds=["tfidf_lr"])
    assert not second["promoted"]
    assert second["champion_before"]["version"] == first["registered_version"]
    mv = registry.champion_version(registry.setup())
    assert str(mv.version) == first["registered_version"]
    assert registry.setup().get_model_version(registry.MODEL_NAME, second["registered_version"]) \
        .tags["promotion"].startswith("rejeitado")


def test_predict_incremental(trained):
    first = predict.run()
    assert first["novas"] == first["total"] > 0
    again = predict.run()
    assert again["novas"] == 0 and again["total"] == first["total"]
    scores = pd.read_parquet(settings.GOLD_DIR / "review_scores.parquet")
    assert not scores.duplicated(["app", "review_id"]).any()
