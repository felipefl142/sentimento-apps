"""MLflow: setup do tracking/registry local e acesso ao champion.

Modelo registrado: `sentimento_reviews` - pyfunc `SentimentModel` (texto -> p_neg/p_neu/p_pos).
O alias `champion` aponta a versao servida por `make predict` e pelo dashboard.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
import mlflow.pyfunc  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from config import settings  # noqa: E402

CHAMPION = "champion"
MODEL_NAME = "sentimento_reviews"
PIP_REQUIREMENTS = ["mlflow", "pandas", "numpy", "scikit-learn", "joblib", "torch",
                    "transformers", "sentencepiece"]


def setup() -> MlflowClient:
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(settings.MLFLOW_EXPERIMENT) is None:
        Path(settings.MLFLOW_ARTIFACT_ROOT).mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(settings.MLFLOW_EXPERIMENT,
                                 artifact_location=Path(settings.MLFLOW_ARTIFACT_ROOT).as_uri())
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT)
    return MlflowClient()


def champion_version(client: MlflowClient, name: str = MODEL_NAME):
    """ModelVersion do champion ou None (modelo/alias ainda nao existe)."""
    try:
        return client.get_model_version_by_alias(name, CHAMPION)
    except MlflowException:
        return None


def load_champion(client: MlflowClient, name: str = MODEL_NAME):
    """(pyfunc carregado, ModelVersion). `model.predict(df_com_text)` -> probabilidades."""
    mv = champion_version(client, name)
    if mv is None:
        raise RuntimeError(f"{name}: sem champion no registry - rode `make train` antes")
    return mlflow.pyfunc.load_model(f"models:/{name}@{CHAMPION}"), mv
