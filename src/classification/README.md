# Zika 2015 Processing + Classification Pipeline

This folder contains a reproducible pipeline for Step 2:
1. Inspect RSS CSV schema
2. Canonicalize + deduplicate URLs and persist URL registry
3. Fetch raw HTML with polite crawling and retries
4. Extract article title/body text (BeautifulSoup baseline)
5. Run ensemble relevance classification via OpenAI-compatible API with weighted probability pooling and agreement gating

## Files
- `run_pipeline.py`: single CLI entrypoint
- `run_classification_only.py`: classification-only CLI using existing extracted data
- `inspect_csv.py`: CSV inspection + schema inference
- `../cleaning/deduplicate_urls.py`: canonicalization, de-dup, SQLite registry
- `../cleaning/fetch_html.py`: HTML fetch + cache
- `../cleaning/extract_text.py`: text extraction heuristics
- `llm_backends.py`: OpenAI-compatible chat wrapper
- `classify_ensemble.py`: strict-JSON ensemble classifier
- `../utils/common.py`: shared filesystem, logging, hashing/text helpers
- `../utils/schema.py`: RSS schema detection + robust CSV loading
- `../utils/deduplication.py`: URL canonicalization + Google wrapper resolution
- `../utils/llm_helpers.py`: JSON parsing/repair + extractive summary helpers
- `config.yaml`: configuration template
- `requirements.txt`: dependencies

## Setup
1. Create and activate a Python environment (Python 3.10+ recommended).
2. Install dependencies:

```bash
pip install -r src/classification/requirements.txt
```

3. Configure `.env` in repo root with API credentials:

```env
OPENAI_API_KEY=your_api_key_here
```

## Run
From repo root:

```bash
python src/classification/run_pipeline.py \
  --input data/Zika_News_2015.csv \
  --outdir data/processed \
  --config src/classification/config.yaml
```

Google News wrapper strategy:
- If RSS links are Google wrapper URLs (`news.google.com/rss/articles/...`), the pipeline resolves them to publisher URLs before canonicalization/fetch.
- This avoids Google consent/interstitial pages ("Before you continue") and improves extracted headline/body quality.

Force recomputation of extraction + classification (and refetch cached HTML):

```bash
python src/classification/run_pipeline.py \
  --input data/Zika_News_2015.csv \
  --outdir data/processed \
  --config src/classification/config.yaml \
  --force
```

## Pure Ensemble Mode (No Heuristic Prefilter)
- Every row with non-empty `extracted_title` and `extracted_text` is sent to all 3 configured models.
- No keyword/stem gating is applied before LLM calls.
- Rows with missing extracted content are marked `final_label=skipped` with explicit reason in per-model `*_error` fields.
- Classification is scheduled model-first (all rows for model A, then model B, then model C) to reduce local model reload overhead.
- Final decisions are produced from weighted relevance probabilities with a configurable threshold and optional minimum agreement constraint.
- The pipeline writes `data/processed/classification_metrics.json` with agreement, calibration, and optional validation metrics.

## Classification-Only Mode (No fetch/extract)
Use existing extracted output and only run ensemble classification:

```bash
python src/classification/run_classification_only.py \
  --input_extracted data/processed/articles_extracted.parquet \
  --out data/processed/articles_classified.csv \
  --config src/classification/config.yaml
```

Recompute all rows from scratch:

```bash
python src/classification/run_classification_only.py \
  --input_extracted data/processed/articles_extracted.parquet \
  --out data/processed/articles_classified.csv \
  --config src/classification/config.yaml \
  --force
```

### Local Ollama note (Apple silicon)
If Ollama fails with Metal BF16/tensor kernel initialization errors, start server with:

```bash
GGML_METAL_BF16_DISABLE=1 \
GGML_METAL_TENSOR_DISABLE=1 \
OLLAMA_CONTEXT_LENGTH=1024 \
OLLAMA_MAX_LOADED_MODELS=3 \
OLLAMA_NUM_PARALLEL=1 \
ollama serve
```

## Outputs
- `data/processed/zika_2015_unique.csv`
- `data/url_registry.sqlite`
- `data/raw_html/<sha256(canonical_url)>.html`
- `data/processed/articles_extracted.parquet` (or `.csv` fallback)
- `data/processed/articles_classified.csv`
- `data/logs/classification_pipeline.log`
- `src/classification/classification_report.qmd` (Quarto report source)

## Report
Render the classification report from repo root:

```bash
quarto render src/classification/classification_report.qmd
```

The rendered HTML will be written next to the source file:
- `src/classification/classification_report.html`

## Model configuration
`config.yaml` uses OpenAI-compatible chat completions and defines model IDs under:

```yaml
classification:
  openai:
    api_key_env: OPENAI_API_KEY
    base_url: "http://127.0.0.1:11434/v1"  # local Ollama OpenAI-compatible endpoint
  models:
    - alias: mistral
      model_id: mistral:latest
    - alias: zephyr
      model_id: zephyr:latest
    - alias: llama70b
      model_id: llama3.1:8b
```

If your local provider uses different IDs, change only `model_id` values in `config.yaml`. No code changes are required.

## Validation Tuning
If you have a labeled reference set, configure it under `classification.validation` in `config.yaml`:

```yaml
classification:
  validation:
    enabled: true
    labels_path: data/reference/relevance_labels.csv
    url_column: canonical_url
    label_column: relevant
```

When enabled, the classifier will tune ensemble weights and decision thresholds against the labeled subset, then write the selected settings and metrics to `classification_metrics.json`.

## Restart behavior
- Fetch step is restartable by cached HTML files in `data/raw_html` (skips unless `--force`).
- Classification-only mode is restartable by `canonical_url` (skips already classified rows unless `--force`).
