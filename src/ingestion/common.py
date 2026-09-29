"""Utilitarios de bronze: arquivos imutaveis por coleta e log de ingestao."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from config import settings


def run_ts() -> str:
    """Carimbo da coleta (UTC), usado como nome do arquivo de bronze."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def write_atomic(path: Path, data: bytes) -> Path:
    """Grava via arquivo temporario (nunca deixa arquivo pela metade no bronze)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return path


def read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def log_run(source: str, entry: dict) -> None:
    """Atualiza bronze/ingest_log.json (ultima coleta por fonte) e anexa ao historico."""
    entry = {"source": source, "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
             **entry}
    settings.BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    log_file = settings.BRONZE_DIR / "ingest_log.json"
    current = read_json(log_file, {})
    current[source] = entry
    log_file.write_text(json.dumps(current, indent=2, ensure_ascii=False))
    with open(settings.BRONZE_DIR / "ingest_history.jsonl", "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
