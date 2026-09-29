# Orquestracao do pipeline sentimento-apps
PY      := .venv/bin/python
ST      := .venv/bin/streamlit
MLFLOW  := .venv/bin/mlflow
export MLFLOW_DISABLE_AGENT_HINT := 1
export TOKENIZERS_PARALLELISM := false
# wheel do torch com CUDA (RTX 30xx+, driver >= 570); troque por /whl/cpu sem GPU NVIDIA
TORCH_INDEX := https://download.pytorch.org/whl/cu128

.PHONY: help install ingest ingest-full silver quality train train-fast predict gold analyze \
        monitor update retrain daily all all-fast mlflow dashboard test lint clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:        ## Cria venv e instala dependencias (torch pelo indice CUDA)
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install torch --index-url $(TORCH_INDEX)
	$(PY) -m pip install -r requirements.txt

ingest:         ## Google Play -> bronze (incremental; re-coleta os ultimos revision_days)
	$(PY) -m src.ingestion.playstore

ingest-full:    ## Backfill: ate backfill_days / max_backfill por app (~2 min por app)
	$(PY) -m src.ingestion.playstore --full

silver:         ## bronze -> silver (versao mais recente por review, rotulo, texto limpo)
	$(PY) -m src.transform.bronze_to_silver

quality:        ## Checagens do silver (exit 1 se falhar)
	$(PY) -m src.transform.quality

train:          ## tfidf_lr + xlmr_zeroshot + bertimbau_ft (GPU) -> MLflow + gate do champion
	$(PY) -m src.modeling.train

train-fast:     ## So tfidf_lr (CPU, ~1 min)
	$(PY) -m src.modeling.train --skip-transformers

predict:        ## Champion pontua o silver (incremental) -> gold/review_scores
	$(PY) -m src.modeling.predict

gold:           ## Agregados de sentimento por dia e por versao do app
	$(PY) -m src.transform.build_gold

analyze:        ## Termos (log-odds), topicos (NMF) e alertas de pico
	$(PY) -m src.analysis.run

monitor:        ## Performance ao vivo, drift (rotulo/vocabulario/predicao), frescor -> reports/
	$(PY) -m src.monitoring.monitor

# etapas em sequencia explicita (pre-requisitos rodariam em paralelo com make -j)
SERVE   := predict gold analyze monitor
UPDATE  := ingest silver quality $(SERVE)
update:         ## Rotina diaria: dados novos -> pontuacao -> analises -> monitoramento
	@for step in $(UPDATE); do $(MAKE) --no-print-directory $$step || exit 1; done

retrain:        ## Retreino + pontuacao + analises + monitoramento
	@for step in train $(SERVE); do $(MAKE) --no-print-directory $$step || exit 1; done

daily:          ## Agendavel (cron/systemd): update + retreino se o monitor recomendar
	@$(MAKE) --no-print-directory update
	@if $(PY) -c "import json,sys; sys.exit(0 if json.load(open('reports/monitoring.json'))['freshness']['retrain_recommended'] else 1)"; \
	then $(MAKE) --no-print-directory retrain; else echo "retreino nao necessario"; fi

all:            ## Pipeline completo (ingestao -> treino com transformers -> ... -> monitor)
	@for step in ingest silver quality train $(SERVE); do $(MAKE) --no-print-directory $$step || exit 1; done

all-fast:       ## Pipeline completo so com tfidf_lr (sem GPU)
	@for step in ingest silver quality train-fast $(SERVE); do $(MAKE) --no-print-directory $$step || exit 1; done

mlflow:         ## MLflow UI (tracking + registry) em http://localhost:5000
	$(MLFLOW) ui --backend-store-uri sqlite:///mlflow.db --port 5000

dashboard:      ## Dashboard Streamlit em http://localhost:8501
	$(ST) run dashboard/app.py

test:           ## Testes (offline, dados sinteticos)
	$(PY) -m pytest tests/ -v

lint:           ## Lint (ruff)
	$(PY) -m ruff check .

clean:          ## Remove dados gerados, MLflow e relatorios (backfill de novo ~10 min)
	rm -rf data/bronze/* data/silver/* data/gold/* data/monitoring/* mlflow.db mlartifacts reports/*.json
	touch data/bronze/.gitkeep data/silver/.gitkeep data/gold/.gitkeep data/monitoring/.gitkeep
