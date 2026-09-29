"""Configuracao central do projeto.

Carrega variaveis de ambiente (.env) e expoe caminhos e parametros
usados por ingestao, transform, modelagem, analises, monitoramento e dashboard.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
# .env antes de ler qualquer variavel (variaveis ja exportadas no shell tem precedencia)
load_dotenv(ROOT / ".env")

CONFIG_DIR = ROOT / "config"
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", ROOT / "reports"))

BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
MONITORING_DIR = DATA_DIR / "monitoring"

APPS_FILE = CONFIG_DIR / "apps.yaml"
MODEL_FILE = CONFIG_DIR / "model.yaml"
STOPWORDS_FILE = CONFIG_DIR / "stopwords_pt.txt"

TZ = "America/Sao_Paulo"

# Google Play (scraper, sem chave)
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30"))
HTTP_RETRIES = int(os.getenv("HTTP_RETRIES", "4"))
HTTP_DELAY = float(os.getenv("HTTP_DELAY", "0.5"))

# Transformers: "cuda" se disponivel; DEVICE=cpu forca CPU
DEVICE = os.getenv("DEVICE", "")

# MLflow: tracking + registry em sqlite local, artefatos em ./mlartifacts
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
MLFLOW_ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT", str(ROOT / "mlartifacts"))
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "sentimento-apps")


def ensure_dirs() -> None:
    """Cria a arvore de diretorios de dados."""
    for d in (DATA_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR, MONITORING_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_apps() -> dict:
    """Apps monitorados e parametros de coleta (config/apps.yaml)."""
    return _load_yaml(APPS_FILE)


def load_model_cfg() -> dict:
    """Parametros de modelagem, analises e monitoramento (config/model.yaml)."""
    return _load_yaml(MODEL_FILE)


def load_stopwords() -> set[str]:
    """Stopwords pt-BR (sem acento, minusculas) para termos/topicos."""
    return {w.strip() for w in STOPWORDS_FILE.read_text().splitlines()
            if w.strip() and not w.startswith("#")}
