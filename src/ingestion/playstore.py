"""Reviews da Google Play -> bronze/playstore/<app>/<run_ts>.jsonl.gz

    python -m src.ingestion.playstore            # incremental (watermark - revision_days)
    python -m src.ingestion.playstore --full     # backfill (backfill_days / max_backfill)
    python -m src.ingestion.playstore --app nubank

Pagina em ordem NEWEST ate passar do limite. Review editada ganha `at` novo e reaparece
no topo com o mesmo reviewId -> o silver fica com a versao mais recente. Datas saem da lib
como datetime ingenuo no fuso LOCAL da maquina (fromtimestamp) -> convertidas para UTC aqui.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import socket
import sys
import time
from datetime import UTC, datetime, timedelta

from google_play_scraper import Sort, reviews

from config import settings
from src.ingestion.common import log_run, read_json, write_atomic

log = logging.getLogger("playstore")

SOURCE = "playstore"
# a lib usa urlopen SEM timeout: uma conexao pendurada travaria a coleta para sempre
socket.setdefaulttimeout(settings.HTTP_TIMEOUT)


def _utc(dt: datetime | None) -> str | None:
    # datetime ingenuo da lib = hora local da maquina; astimezone() interpreta assim
    return dt.astimezone(UTC).isoformat(timespec="seconds") if dt else None


def to_record(r: dict, app: str, package: str, fetched_at: str) -> dict:
    return {
        "review_id": r["reviewId"], "app": app, "package": package,
        "user_name": r.get("userName"), "content": r.get("content"), "score": r.get("score"),
        "thumbs_up": r.get("thumbsUpCount"), "created_version": r.get("reviewCreatedVersion"),
        "app_version": r.get("appVersion"), "at_utc": _utc(r.get("at")),
        "reply_content": r.get("replyContent"), "replied_at_utc": _utc(r.get("repliedAt")),
        "fetched_at": fetched_at,
    }


def _fetch_page(package: str, lang: str, country: str, count: int, token):
    """Uma pagina com retry/backoff (a lib levanta excecoes variadas em falha de rede)."""
    for attempt in range(settings.HTTP_RETRIES + 1):
        try:
            return reviews(package, lang=lang, country=country, sort=Sort.NEWEST, count=count,
                           continuation_token=token)
        except Exception as e:  # noqa: BLE001
            if attempt == settings.HTTP_RETRIES:
                raise
            wait = 2 ** attempt * 5
            log.warning("%s: %s (%s) - nova tentativa em %ss", package, type(e).__name__, e, wait)
            time.sleep(wait)
    raise RuntimeError("inalcancavel")


def fetch_app(package: str, lang: str, country: str, since: datetime, max_reviews: int,
              page_size: int, fetch=_fetch_page) -> tuple[list[dict], bool]:
    """(reviews da lib com `at >= since`, completo?). Completo = alcancou `since` ou o teto.

    A lib engole erro de rede no meio da paginacao e devolve pagina parcial com token vazio -
    indistinguivel de "acabou o historico". Por isso parar antes de `since` conta como
    incompleto (o chamador nao avanca o watermark e a proxima coleta cobre o buraco).
    """
    out, token = [], None
    since_local = since.astimezone().replace(tzinfo=None)  # mesmo referencial da lib
    while len(out) < max_reviews:
        page, token = fetch(package, lang, country, page_size, token)
        out.extend(r for r in page if r["at"] >= since_local)
        if page and min(r["at"] for r in page) < since_local:
            return out, True
        if not page or token is None or getattr(token, "token", "") is None:
            return out, False
        time.sleep(settings.HTTP_DELAY)
    return out[:max_reviews], True


def watermarks_path():
    return settings.BRONZE_DIR / SOURCE / "_watermarks.json"


def ingest(full: bool = False, only: str | None = None, fetch=_fetch_page,
           now: datetime | None = None) -> dict:
    cfg = settings.load_apps()
    icfg = cfg["ingest"]
    now = now or datetime.now(UTC)
    ts = now.strftime("%Y%m%dT%H%M%SZ")
    marks = read_json(watermarks_path(), {})
    summary, failed = {}, []
    for app, meta in cfg["apps"].items():
        if only and app != only:
            continue
        mark = marks.get(app)
        if full or mark is None:
            mode = "backfill"
            since, cap = now - timedelta(days=icfg["backfill_days"]), icfg["max_backfill"]
        else:
            mode = "incremental"
            since = datetime.fromisoformat(mark["max_at"]) - timedelta(days=icfg["revision_days"])
            cap = icfg["max_backfill"]
        t0 = time.time()
        try:
            raw, complete = fetch_app(meta["package"], cfg["lang"], cfg["country"], since,
                                      cap, icfg["page_size"], fetch=fetch)
        except Exception as e:  # noqa: BLE001
            log.error("%s: falhou - %s", app, e)
            failed.append(app)
            continue
        recs = [to_record(r, app, meta["package"], now.isoformat(timespec="seconds"))
                for r in raw]
        if recs:
            body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs).encode()
            write_atomic(settings.BRONZE_DIR / SOURCE / app / f"{ts}.jsonl.gz",
                         gzip.compress(body))
            ats = [r["at_utc"] for r in recs]
            if complete or mark is None:
                # backfill incompleto ainda vale: o historico antigo so falta, nao tem buraco
                prev_max = mark["max_at"] if mark else min(ats)
                marks[app] = {"max_at": max(max(ats), prev_max),
                              "min_at": min([min(ats)] + ([mark["min_at"]] if mark else []))}
        summary[app] = {"mode": mode, "since": since.isoformat(timespec="seconds"),
                        "n": len(recs), "completo": complete, "min_at": min((r["at_utc"] for r in recs), default=None),
                        "max_at": max((r["at_utc"] for r in recs), default=None),
                        "seconds": round(time.time() - t0, 1)}
        log.info("%s: %s %d reviews (%s .. %s) em %.0fs%s", app, mode, len(recs),
                 summary[app]["min_at"], summary[app]["max_at"], summary[app]["seconds"],
                 "" if complete else " - INCOMPLETO (paginacao parou antes do limite)")
    write_atomic(watermarks_path(), json.dumps(marks, indent=2).encode())
    log_run(SOURCE, {"run_ts": ts, "apps": summary, "failed": failed})
    if failed:
        raise RuntimeError(f"falha na coleta: {failed}")
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="backfill completo")
    ap.add_argument("--app", help="so este app (chave do apps.yaml)")
    args = ap.parse_args()
    settings.ensure_dirs()
    try:
        ingest(full=args.full, only=args.app)
    except Exception as e:  # noqa: BLE001
        log.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
