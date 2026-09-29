"""Fixtures: bronze sintetico (formato do src.ingestion.playstore) num DATA_DIR temporario
e MLflow temporario. Nada acessa a rede."""
from __future__ import annotations

import copy
import gzip
import json
import random
from datetime import UTC, date, datetime, timedelta

import pytest

from config import settings

APPS_CFG = {
    "lang": "pt", "country": "br",
    "apps": {"alpha": {"package": "com.alpha", "nome": "Alpha"},
             "beta": {"package": "com.beta", "nome": "Beta"}},
    "ingest": {"page_size": 50, "revision_days": 7, "backfill_days": 60, "max_backfill": 10000},
}
START, END = date(2026, 1, 1), date(2026, 6, 30)
SPIKE_DAY = date(2026, 6, 10)          # alpha: "pix fora do ar"
EDITED_ID = "alpha-edit-1"
PER_DAY = 60

TEXTS = {
    "pos": ["adorei o app, muito bom", "excelente, recomendo demais", "ótimo e rápido",
            "perfeito, funciona muito bem", "top demais 😍", "maravilhoso, nota dez"],
    "neg": ["péssimo, não funciona", "app horrível, trava toda hora",
            "não consigo entrar na conta", "atendimento ruim, demora demais",
            "cobrança indevida, um absurdo", "lixo, só dá erro"],
    "neu": ["é ok, mas poderia melhorar", "razoável, tem alguns problemas",
            "mais ou menos, às vezes trava", "bom mas falta função", "mediano, nada demais"],
}
SUFFIX = ["", " hoje", " sempre", " agora", " de novo", " sinceramente", " na real"]
SCORES = {"pos": [4, 5], "neg": [1, 2], "neu": [3]}


def review(app: str, rid: str, d: date, label: str, rng: random.Random, text=None,
           score=None, hour=12, version="1.0") -> dict:
    at = datetime(d.year, d.month, d.day, hour, rng.randrange(60), tzinfo=UTC)
    return {"review_id": rid, "app": app, "package": f"com.{app}", "user_name": "Fulano",
            "content": text or rng.choice(TEXTS[label]) + rng.choice(SUFFIX),
            "score": score or rng.choice(SCORES[label]), "thumbs_up": rng.randrange(5),
            "created_version": version, "app_version": version,
            "at_utc": at.isoformat(timespec="seconds"), "reply_content": None,
            "replied_at_utc": None, "fetched_at": "2026-07-01T00:00:00+00:00"}


def synthetic_reviews() -> dict[str, list[dict]]:
    rng = random.Random(7)
    out = {}
    for app in APPS_CFG["apps"]:
        recs, d, i = [], START, 0
        while d <= END:
            for _ in range(PER_DAY):
                label = rng.choices(["pos", "neg", "neu"], [0.6, 0.3, 0.1])[0]
                version = "2.0" if d >= date(2026, 6, 1) else "1.0"
                recs.append(review(app, f"{app}-{i}", d, label, rng, version=version))
                i += 1
            if app == "alpha" and d == SPIKE_DAY:
                for _ in range(120):
                    recs.append(review(app, f"{app}-{i}", d, "neg", rng, score=1, version="2.0",
                                       text="pix fora do ar, não consigo pagar nada"))
                    i += 1
            d += timedelta(days=1)
        out[app] = recs
    rng2 = random.Random(1)
    out["alpha"].append(review("alpha", EDITED_ID, date(2026, 5, 1), "neg", rng2, score=1,
                               text="horrível, perdi meu dinheiro"))
    return out


def write_bronze(data_dir, recs_by_app: dict, ts: str) -> None:
    for app, recs in recs_by_app.items():
        folder = data_dir / "bronze" / "playstore" / app
        folder.mkdir(parents=True, exist_ok=True)
        body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs).encode()
        (folder / f"{ts}.jsonl.gz").write_bytes(gzip.compress(body))


def model_cfg_for_tests() -> dict:
    cfg = settings._load_yaml(settings.MODEL_FILE)
    cfg = copy.deepcopy(cfg)
    cfg["candidates"]["tfidf_lr"].update(C_grid=[1.0], word_max_features=5000,
                                         char_max_features=5000, cap_per_app_class=2000)
    return cfg


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    for attr, sub in (("DATA_DIR", ""), ("BRONZE_DIR", "bronze"), ("SILVER_DIR", "silver"),
                      ("GOLD_DIR", "gold"), ("MONITORING_DIR", "monitoring")):
        monkeypatch.setattr(settings, attr, tmp_path / sub if sub else tmp_path)
    monkeypatch.setattr(settings, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(settings, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(settings, "MLFLOW_ARTIFACT_ROOT", str(tmp_path / "mlartifacts"))
    monkeypatch.setattr(settings, "MLFLOW_EXPERIMENT", "teste")
    monkeypatch.setattr(settings, "load_apps", lambda: APPS_CFG)
    cfg = model_cfg_for_tests()
    monkeypatch.setattr(settings, "load_model_cfg", lambda: copy.deepcopy(cfg))
    monkeypatch.setenv("DEVICE", "cpu")
    settings.ensure_dirs()
    return tmp_path


@pytest.fixture
def bronze(data_dir):
    """2 apps x 181 dias; 2a coleta com a review EDITED_ID editada (nota 1 -> 5)."""
    write_bronze(data_dir, synthetic_reviews(), "20260701T000000Z")
    rng = random.Random(3)
    edited = review("alpha", EDITED_ID, date(2026, 5, 3), "pos", rng, score=5,
                    text="resolveram rápido, agora está ótimo")
    edited["fetched_at"] = "2026-07-02T00:00:00+00:00"
    write_bronze(data_dir, {"alpha": [edited]}, "20260702T000000Z")
    return data_dir


@pytest.fixture
def silver(bronze):
    from src.transform import bronze_to_silver
    bronze_to_silver.run()
    return bronze


@pytest.fixture
def trained(silver):
    from src.modeling import train
    summary = train.run(kinds=["tfidf_lr"])
    return silver, summary
