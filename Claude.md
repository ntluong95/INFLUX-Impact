# Claude Roadmap: INFLUX Impact Pipeline

This document consolidates the project instructions and operating rules into one practical roadmap.

## 1. Project Mission

Build a reproducible pathogen-news pipeline to study cascading social-ecological impacts across:

- `human`
- `animal`
- `plant`

and across four languages:

- `en`
- `fr`
- `es`
- `pt`

Core workflow:

1. Build multilingual disease/pathogen search strings.
2. Retrieve Google News RSS at scale.
3. Remove blocked/non-news domains.
4. Classify headline relevance via batch API (OpenAI or Anthropic, selected by config).
5. Retrieve and extract full-text for kept rows.
6. Run BERTopic for impact-oriented topic exploration.
7. Extract impact-topics (in development)

## 2. Source-of-Truth Priority

When guidance conflicts, use this order:

1. Code behavior in `src/` (especially `src/run_pipeline.py`, `src/config/pipeline.yaml`, and stage modules).
2. `src/README.md` for pipeline contracts and outputs.
3. `src/epp_namelist/README.md` for name-list generation rules.
4. Root `README.md` for research framing and paper-level goals.
5. `src/prompt_codex*.md` as historical implementation intent, not always the current runtime truth.

## 3. Non-Negotiable Rules

### 3.1 Security and Secrets

- Do not hardcode secrets.
- Keep API credentials in local `.env`.
- At minimum, set one of:
  - `OPENAI_API_KEY` for OpenAI headline classification.
  - `ANTHROPIC_API_KEY` for Anthropic headline classification.
  - `ENTREZ_KEY` optionally for faster NCBI lookups in R name-list scripts.

### 3.2 Scope and Dataset Key Contract

- Every output is separated by dataset key: `<pathogen_domain>_<language_code>`.
- Expected keys:
  - `human_en`, `human_fr`, `human_es`, `human_pt`
  - `animal_en`, `animal_fr`, `animal_es`, `animal_pt`
  - `plant_en`, `plant_fr`, `plant_es`, `plant_pt`

### 3.3 Modular Structure

- Keep stage logic inside existing module folders:
  - `src/rss_retrieval`
  - `src/classification`
  - `src/fulltext_retrieval`
  - `src/extraction`
  - `src/utils`

### 3.4 Reproducibility and Resume Behavior

- Pipeline must be restartable.
- Existing artifacts are reused unless `--force` is set.
- Stage checkpoint/state files are written under `data/` and reused between runs.

### 3.5 Runtime Defaults

- Search timeframe default: `2005-01-01` to `2025-12-31`.
- Headline classification provider: set `classification.provider` in `pipeline.yaml` to `openai` or `anthropic`.
  - OpenAI default model: `gpt-5-nano`
  - Anthropic default model: `claude-haiku-4-5`
- Only one provider is used per run. No ensemble logic.
- Domain blocklist source: `data/domains_removed.csv`, using rows where `is_news_outlet == "No"`.

## 4. Environment and Dependencies

- Python requirement from `pyproject.toml`: `>=3.14,<3.15`.
- Main dependency groups:
  - Data IO and parsing: `pandas`, `pyyaml`, `pyarrow`.
  - Retrieval/scraping: `requests`, `pygooglenews`, `feedparser`, `beautifulsoup4`, `lxml`, `readability-lxml`, `trafilatura`.
  - LLM integration: `openai`, `anthropic`.
  - Topic modeling: `sentence-transformers`, `bertopic`, `umap-learn`, `hdbscan`, `scikit-learn`.

## 5. Input Data Contracts

### 5.1 Search Workbooks

Expected files:

- `data/inputs/human_diseases.xlsx`
- `data/inputs/animal_diseases.xlsx`
- `data/inputs/plant_diseases.xlsx`

Search column preference:

1. `search_string_<language>`
2. `search_string`
3. `search_string_en`

### 5.2 Domain Blocklist

Expected file:

- `data/domains_removed.csv`

Required columns:

- `domain`
- `is_news_outlet`

Blocking rule:

- Rows with `is_news_outlet == "No"` become blocklist entries.
- Domain matching includes subdomains.

## 6. Stage-by-Stage Rules

### Stage 0. Multilingual Name List Generation (`src/epp_namelist`)

Purpose:

- Build multilingual pathogen and disease term expansions used by RSS retrieval.

Rules:

- Human/animal workflows use shared helper `common_multilingual_names.R`.
- Extra columns after `search_string_en` are dropped during import in shared helper.
- Languages targeted: `en`, `es`, `fr`, `pt`.
- Retrieval sources:
  - NCBI/taxize for taxonomy and common names.
  - Wikidata for multilingual labels/aliases.
- Plant workflow is EPPO-first when `EPPO_taxonID` exists, then Wikidata fallback.
- Always retain scientific Latin names in search strings.

Outputs:

- `data/inputs/human_diseases.xlsx`
- `data/inputs/animal_diseases.xlsx`
- `data/inputs/plant_diseases.xlsx`

### Stage 1. RSS Retrieval (`src/rss_retrieval/retrieve_google_rss.py`)

Purpose:

- Retrieve Google News RSS items for each dataset key.

Rules:

- One initial whole-period scan per search string and locale.
- If results are below split threshold, scrape window directly.
- If results meet/exceed threshold (`split_threshold_entries`, default `100`):
  - First split whole-period into coarse windows (default `5` years).
  - Then split hot windows down hierarchy (default days: `30`, `14`, `7`, `3`, `1`).
- Breadth-first processing by stage rank to complete broader scans before finer windows.
- Per-locale retrieval uses configured `Accept-Language` and locale parameters.
- Throttling/captcha detection triggers cooldown and resumable retries.
- Manifest tracks status (`pending`, `success`, `empty`, `split`, `throttled`, `failed`).
- Parsed outputs aggregate only completed `success` and `empty` windows.
- Deduplication runs on parsed RSS records before writing final stage outputs.

Locale strategy:

- Each language has multiple locales representing different Google News country editions.
- Locales define the `lang`, `country`, and `Accept-Language` header for RSS requests.
- This ensures geographic coverage (e.g. en_us, en_gb, en_ca for English news).
- The previous `language_variants` field was removed after experimentation showed it had no effect on Google News results — only country-level locale configuration matters.

Primary outputs:

- `data/raw/rss/<dataset_key>/...` (cached payloads + manifest)
- `data/intermediate/rss/<dataset_key>.csv`
- `data/intermediate/rss/<dataset_key>.parquet`
- `data/intermediate/rss/<dataset_key>_metrics.csv`
- `data/intermediate/rss/<dataset_key>_scan_summary.csv`

### Stage 2A. Domain Filtering (`src/classification/domain_filter.py`)

Purpose:

- Remove clearly irrelevant/non-news domains before LLM classification.

Rules:

- Read RSS output for dataset key.
- Ensure `source_domain` is present; derive from URL when missing.
- Remove rows matching blocked domain list from `data/domains_removed.csv`.
- Emit metrics for input, blocked, kept rows.

Outputs:

- `data/intermediate/classification/<dataset_key>_headlines_prefiltered.csv`
- `data/intermediate/classification/<dataset_key>_domain_filter_metrics.csv`

### Stage 2B. Headline Batch Classification (`src/classification/headline_batch.py`)

Purpose:

- Classify headline-level relevance using a batch API provider (OpenAI or Anthropic).

Architecture:

- Provider-agnostic design via `src/utils/batch_provider.py` protocol.
- Provider implementations: `src/utils/openai_batch.py`, `src/utils/anthropic_batch.py`.
- Provider is selected by `classification.provider` in `pipeline.yaml`.
- Only one provider is used per run. No ensemble logic.
- Mixed-provider hydration is supported: batches submitted by one provider can be polled/hydrated even if the config has since changed to a different provider.

Rules:

- Requests use JSON response format with keys:
  - `label` (`relevant|irrelevant|unsure`)
  - `confidence` (`0..1`)
  - `rationale` (short text)
- Action mapping:
  - `relevant -> keep`
  - `irrelevant -> drop`
  - `unsure -> review`
- Batch chunking respects configured caps:
  - max requests per batch
  - max input bytes per batch
  - estimated prompt-token budget per batch
  - dataset-level pending-batch and enqueued-token caps
- Batch naming convention:
  - `<dataset_key>_headline_batch_<nnnn>`
- Classification is asynchronous and resumable.
- Pipeline `--stage all` must pause downstream stages when headline results are still pending.

State columns (provider-agnostic):

- `batch_custom_id`, `batch_name`, `batch_id`, `batch_provider`, `batch_state`, `batch_submitted_at`

Outputs:

- `data/intermediate/classification/<dataset_key>_headlines_classified.csv`
- `data/intermediate/batches/<dataset_key>/...`
  - request JSONL files
  - output/error JSONL hydration files
  - batch registry CSV

### Stage 3. Full-Text Retrieval (`src/fulltext_retrieval/retrieve_fulltext.py`)

Purpose:

- Retrieve article full text for rows with `final_action == keep` (configurable).

Rules:

- Input comes from classified headline state.
- Resolve Google wrapper URLs to publisher URLs when possible.
- Fetch politely with retry/backoff/rate-limit controls from config.
- Parse metadata (title, canonical URL, date, authors, language, domain).
- Full-text extraction fallback order:
  1. `trafilatura`
  2. `readability-lxml`
  3. `newspaper3k` (if available)
- Enforce minimum word count (`fulltext.min_text_words`, default `120`).
- Deduplicate by canonical/final URL and content hash.
- Checkpoint writing every `fulltext.checkpoint_every` rows.

Outputs:

- `data/final/fulltext/<dataset_key>.csv`
- `data/final/fulltext/<dataset_key>.parquet`
- `data/final/fulltext/<dataset_key>_failed.csv`

### Stage 4. BERTopic Exploration (`src/extraction/bertopic_fulltext.py`)

Purpose:

- Generate impact-oriented topic overviews from full text.

Rules:

- Model only documents meeting minimum text thresholds.
- Skip modeling when prepared docs < `bertopic.min_documents` (default `20`) and still write empty outputs + summary.
- Embedding model configurable per pathogen domain.
- Clustering strategy configurable (`kmeans` default or `hdbscan`).
- Save document-topic assignments, topic info, 2D projection, stage timings, and summary JSON.

Outputs:

- `data/final/bertopic/<dataset_key>/<dataset_key>_document_topics.csv`
- `data/final/bertopic/<dataset_key>/<dataset_key>_topic_info.csv`
- `data/final/bertopic/<dataset_key>/<dataset_key>_projection.csv`
- `data/final/bertopic/<dataset_key>/<dataset_key>_stage_timings.csv`
- `data/final/bertopic/<dataset_key>/<dataset_key>_summary.json`

## 7. Pipeline Execution Roadmap

### 7.1 Full pipeline

```bash
python3 src/run_pipeline.py --stage all
```

### 7.2 Stage-specific runs

```bash
python3 src/run_pipeline.py --stage rss
python3 src/run_pipeline.py --stage classify
python3 src/run_pipeline.py --stage fulltext
python3 src/run_pipeline.py --stage bertopic
```

### 7.3 Restrict by domain/language

```bash
python3 src/run_pipeline.py --stage rss --pathogen-domains human,animal --languages en,fr,es,pt
```

### 7.4 Force recompute

```bash
python3 src/run_pipeline.py --stage all --force
```

### 7.5 Methodology diagrams (`diagrams/`)

Purpose:

- Generate publication-quality methodology figures from `Methodology Impact-revised.md` with the external `papervizagent/` checkout and its separate Python 3.12 environment.

Directory contract:

- `diagrams/prepare_inputs.py` writes the three diagram specs to `diagrams/inputs/`.
- `diagrams/inputs/` stores the editable JSON prompts for `pipeline_overview`, `ensemble_filtering`, and `uig_discovery`.
- `diagrams/references/` holds curated PNG/JPG reference diagrams for PaperVizAgent manual retrieval mode.
- `diagrams/outputs/` holds rendered PNG exports decoded from PaperVizAgent result JSON.

Rules:

- Run PaperVizAgent only through `diagrams/run_paperviz.sh`; it stages local inputs into `../../papervizagent/data/PaperBananaBench/diagram/`.
- Keep PaperVizAgent isolated from the main INFLUX `.venv`; use `../../papervizagent/.venv` from within `diagrams/`.
- Treat `diagrams/inputs/*.json` as the source of truth for figure content and style; regenerate outputs instead of editing exported images by hand.
- Keep API keys out of repo files; source them from `../../papervizagent/.env` or the shell before generation.

Usage:

```bash
cd diagrams
./run_paperviz.sh              # default: dev_full mode, 3 critic rounds
./run_paperviz.sh dev_full 5   # override mode and max rounds
```

## 8. Quality and Traceability Rules

- Use atomic writes for CSV/Parquet/JSON outputs where implemented.
- Keep dataset-level logging under `data/logs/`.
- Preserve manifest/batch registry/state artifacts for auditability.
- Prefer configuration changes in `src/config/pipeline.yaml` over hardcoded logic changes.
- Keep naming explicit with dataset key in output artifacts and batch metadata.

## 9. Research Program Context (Paper-Level Roadmap)

This codebase supports a broader research trajectory:

1. Compare cascading impacts across selected pathogens/pests.
2. Build archetypes of cascading pathways across larger pathogen sets.
3. Link qualitative cascades with quantitative impact metrics.
4. Explore short-term vs long-term impact propagation.

Operational pipeline outputs are intended to feed these analyses iteratively.

## 10. Historical Notes

- `src/archive/classification/` contains legacy Zika-focused pipeline material and reports.
- `src/archive/rss_retrieval/` contains the original RSS retrieval code and language variant experiment (now concluded).
- `src/prompt_codex.md` and `src/prompt_codex_20260321.md` capture earlier implementation instructions; current runtime behavior should be validated against `src/` code and `src/config/pipeline.yaml`.

## 11. Updates docs
- Always update Claude.md, files in .claude folder, and any documents in the sub-folder for any changes
