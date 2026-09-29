"""Coleta com scraper falso: paginacao, limite por data, watermark, coleta incompleta."""
import gzip
import json
from datetime import UTC, datetime, timedelta

from src.ingestion import playstore


class Token:
    def __init__(self, token):
        self.token = token


def fake_store(n_days: int, per_day: int, now: datetime, fail_after_pages: int | None = None):
    """Reviews em ordem NEWEST (datetime ingenuo LOCAL, como a lib)."""
    items = []
    for d in range(n_days):
        for k in range(per_day):
            at = (now - timedelta(days=d, minutes=k)).astimezone().replace(tzinfo=None)
            items.append({"reviewId": f"r{d}-{k}", "userName": "x", "content": "ok", "score": 5,
                          "thumbsUpCount": 0, "reviewCreatedVersion": "1", "appVersion": "1",
                          "at": at, "replyContent": None, "repliedAt": None})
    calls = {"n": 0}

    def fetch(package, lang, country, count, token):
        calls["n"] += 1
        start = token.token if token else 0
        if fail_after_pages is not None and calls["n"] > fail_after_pages:
            return [], Token(None)          # a lib engole o erro: pagina vazia, token vazio
        page = items[start:start + count]
        nxt = start + count if start + count < len(items) else None
        return page, Token(nxt)
    return fetch, calls


def read_bronze(data_dir, app):
    files = sorted((data_dir / "bronze" / "playstore" / app).glob("*.jsonl.gz"))
    return [json.loads(line) for f in files for line in gzip.decompress(f.read_bytes()).splitlines()]


def test_backfill_para_no_limite_de_dias(data_dir, monkeypatch):
    monkeypatch.setattr(playstore.time, "sleep", lambda s: None)
    now = datetime(2026, 7, 1, 12, tzinfo=UTC)
    fetch, _ = fake_store(n_days=100, per_day=10, now=now)
    s = playstore.ingest(only="alpha", fetch=fetch, now=now)
    assert s["alpha"]["mode"] == "backfill" and s["alpha"]["completo"]
    recs = read_bronze(data_dir, "alpha")
    ats = [datetime.fromisoformat(r["at_utc"]) for r in recs]
    assert min(ats) >= now - timedelta(days=60)          # backfill_days = 60
    assert len(recs) == 601                              # 60 dias x 10 + a do limite exato
    assert all(a.tzinfo is not None for a in ats)       # UTC explicito no bronze


def test_incremental_usa_watermark(data_dir, monkeypatch):
    monkeypatch.setattr(playstore.time, "sleep", lambda s: None)
    now = datetime(2026, 7, 1, 12, tzinfo=UTC)
    fetch, _ = fake_store(n_days=100, per_day=10, now=now)
    playstore.ingest(only="alpha", fetch=fetch, now=now)
    later = now + timedelta(days=2)
    fetch2, calls = fake_store(n_days=100, per_day=10, now=later)
    s = playstore.ingest(only="alpha", fetch=fetch2, now=later)
    assert s["alpha"]["mode"] == "incremental"
    # watermark (now) - revision_days (7) -> ~9-10 dias de reviews, nao 60
    assert 80 <= s["alpha"]["n"] <= 100
    assert calls["n"] <= 3


def test_coleta_incompleta_nao_avanca_watermark(data_dir, monkeypatch):
    monkeypatch.setattr(playstore.time, "sleep", lambda s: None)
    now = datetime(2026, 7, 1, 12, tzinfo=UTC)
    fetch, _ = fake_store(n_days=100, per_day=10, now=now)
    playstore.ingest(only="alpha", fetch=fetch, now=now)
    mark = json.loads(playstore.watermarks_path().read_text())["alpha"]["max_at"]
    later = now + timedelta(days=20)
    broken, _ = fake_store(n_days=100, per_day=10, now=later, fail_after_pages=1)
    s = playstore.ingest(only="alpha", fetch=broken, now=later)
    assert not s["alpha"]["completo"]
    assert json.loads(playstore.watermarks_path().read_text())["alpha"]["max_at"] == mark
