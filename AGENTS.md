# AGENTS.md

3-class sentiment (neg/neu/pos) of pt-BR Google Play reviews for 5 apps (Nubank, iFood, Mercado Livre, gov.br, WhatsApp): incremental scraping (bronze JSONL.gz, never overwritten) → DuckDB silver/gold Parquet → 3 candidates (TF-IDF+LogReg, XLM-R zero-shot, BERTimbau fine-tune on CUDA) compared on a temporal test and registered in MLflow as one pyfunc (alias `champion`) → incremental scoring → daily aggregates, log-odds terms, NMF topics, robust-z spike alerts → monitoring (live macro-F1, label/vocab/prediction drift, freshness) → Streamlit. `README.md` has results and design decisions.

## Commands (Makefile, all via `.venv`)

- `make install`: venv, then torch from the CUDA index (`TORCH_INDEX`, cu128), then `requirements.txt` (pip freeze WITHOUT torch/nvidia-*/triton; it's the dependency source of truth, pyproject has none).
- `make ingest-full` (backfill ~10 min) then `make all` (train with transformers ~20 min on an RTX 3070). `make all-fast` / `make train-fast` = TF-IDF only (CPU, ~4 min).
- `make update` = ingest → silver → quality → predict → gold → analyze → monitor (no training). `make daily` = update + `retrain` only if `reports/monitoring.json` says `retrain_recommended`.
- `make test` (offline pytest: synthetic bronze + tmp MLflow, `DEVICE=cpu`, TF-IDF only; dashboard smoke test skips without real data), `make lint` (ruff).
- `make mlflow` (:5000, `sqlite:///mlflow.db`), `make dashboard` (:8501).
- The user's shell exports `MAKEFLAGS=-j`: `make silver quality` runs IN PARALLEL and breaks. Composite targets use explicit for-loops; call single targets one at a time.
- Run modules as `python -m src....` from the repo root (`config` has no `__init__.py`).
- Env overrides: `DATA_DIR`, `REPORTS_DIR`, `MLFLOW_TRACKING_URI`, `MLFLOW_ARTIFACT_ROOT`, `MLFLOW_EXPERIMENT`, `HTTP_*`, `DEVICE` (cpu/cuda).

## Source gotchas (google-play-scraper 1.2.7)

- Uses `urlopen` WITHOUT a timeout: a stalled connection hangs forever → `playstore.py` sets `socket.setdefaulttimeout(HTTP_TIMEOUT)`.
- Swallows network errors mid-pagination: it returns a partial page and a token whose `.token is None`, which looks the same as "end of history". The returned token object is never `None`, so check `.token`. Stopping before `since` = `completo: false`, and the watermark is NOT advanced (the next run covers the gap).
- `at`/`repliedAt` come as NAIVE datetimes in the MACHINE's local timezone (`datetime.fromtimestamp`) → converted to UTC ISO in bronze (`.astimezone(UTC)`). Silver converts to America/Sao_Paulo for `data`.
- NEWEST sort, ~1000 reviews/s. Newest review lags ~1 day. Edited reviews get a new `at` and reappear at the top with the same `reviewId` → silver keeps the latest version per (app, review_id); `reviews_edicoes` tracks score/text changes.
- Volume differs a lot: Nubank ~190/day and WhatsApp ~600/day, so the 80k `max_backfill` covers 365 days for Nubank/iFood but only ~4–6 months for the others.

## Data flow

- bronze: `playstore/<app>/<run_ts>.jsonl.gz` (the only place with `user_name`), `playstore/_watermarks.json`, `ingest_log.json`, `ingest_history.jsonl`.
- silver: `reviews` (`text` = `text.clean(content)`, `label` from stars, `curto`, `n_chars`), `reviews_edicoes`, `apps_dim`.
- gold: `quality_checks.json`, `review_scores` (champion version + `text_hash`; a new champion or an edited text triggers rescoring), `backtest_predictions` (test × candidates), `sentimento_diario`, `sentimento_versao`, `termos_sentimento`, `termos_emergentes`, `topicos`, `alertas_diario`, `alertas`.
- monitoring: `live_scores` (weekly macro-F1 after the champion's `data_max_date`). reports: `train_summary.json`, `monitoring.json`.

## Modeling conventions

- Label: 1–2★ neg, 3★ neu, 4–5★ pos. Neutral is ~2–5% of reviews and is inherently noisy → macro-F1 (primary) is dominated by F1_neu. Don't switch to accuracy.
- Split by date relative to the silver max date: test = last 30 days, valid = the 30 before, train = the rest. Never random. Train is capped per app×class (`cap_per_group`); valid/test keep the real distribution.
- Everything served goes through `SentimentModel` (`src/modeling/sentiment_model.py`), an MLflow pyfunc: DataFrame[`text`] → `p_neg, p_neu, p_pos, pred`. It applies `text.clean` itself (idempotent). Never call sklearn/HF directly for serving.
- `sentiment_model.py` must not import `config`: the pickled PythonModel references the module by path, so models load only from the repo root.
- Artifacts: `sklearn` (joblib), `hf` (save_pretrained in bf16 + `serving.json`), or `hf_ref` (zero-shot: model_id + revision SHA, loaded from the HF cache). Only the WINNING transformer gets a model artifact; TF-IDF is always logged.
- HF label mapping goes through `_canon` (first 3 chars of `id2label`, lowercased) → works for `negative/neutral/positive` and `neg/neu/pos`.
- Gate: re-score the current champion on the NEW test; alias only if `F1 >= F1_champion - gate_tolerance`. The version is registered anyway, with a `promotion` tag. No refit: what's served is what was evaluated.
- `search_model_versions` does NOT populate `aliases` (use `get_model_version_by_alias`). `ModelVersion.version` may be int, so compare as str.
- XLM-R/Cardiff tokenizer needs `sentencepiece` (transformers 5 falls back to tiktoken and errors otherwise).
- pandas 3 reads Parquet/DuckDB DATE as datetime64: convert explicitly (`pd.to_datetime(...).dt.date` or `pd.Timestamp`) before comparing with `date`.

## Analysis conventions

- Terms: Monroe log-odds with an informative Dirichlet prior (`termos.log_odds_z`). Stopwords in `config/stopwords_pt.txt` (no accents, via `text.normalize`). Topics add `topics.extra_stopwords` (sentiment adjectives and app names), which would otherwise become generic "topics".
- Alerts: robust z against the previous 28 days, with denominator `max(1.4826·MAD, sqrt(p(1-p)/n), floor)`. Without the binomial term, low-volume days trigger false alarms (the test covers this).

## Style

- Code comments/docstrings in Portuguese (pt-BR, mostly without accents in code); match the existing style. UI text with accents.
- ruff: line-length 100, E501 ignored; isort first-party `src`, `config`.
- Dashboard colors come from `dashboard/style.py`. Sentiment is diverging (neg red, neu gray, pos blue): always shown with a legend or labels. Models use categorical slots 1–3 (`ENTITY_SLOT`). Apps are NEVER color-coded: use small multiples or the selector. Model × stars = primary ink × muted dotted. Status uses the reserved palette with icon + label. Set plotly titles BEFORE `style.layout()`. Streamlit doesn't reload imported modules, so restart after editing `style.py`.
- Beware Streamlit "magic": a bare expression (e.g. a ternary of calls) renders its return value (`None`).
