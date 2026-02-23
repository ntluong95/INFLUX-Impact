# Zika 2015 Processing + Classification Pipeline

This folder contains a reproducible pipeline for Step 2:
1. Inspect RSS CSV schema
2. Canonicalize + deduplicate URLs and persist URL registry
3. Fetch raw HTML with polite crawling and retries
4. Extract article title/body text (BeautifulSoup baseline)
5. Run 3-model ensemble relevance classification via OpenAI-compatible API (including local endpoints such as Ollama)

## Files
- `run_pipeline.py`: single CLI entrypoint
- `inspect_csv.py`: CSV inspection + schema inference
- `deduplicate_urls.py`: canonicalization, de-dup, SQLite registry
- `fetch_html.py`: HTML fetch + cache
- `extract_text.py`: text extraction heuristics
- `llm_backends.py`: OpenAI-compatible chat wrapper
- `classify_ensemble.py`: strict-JSON ensemble classifier
- `utils.py`: shared utilities
- `config.yaml`: configuration template
- `requirements.txt`: dependencies

## Setup
1. Create and activate a Python environment (Python 3.10+ recommended).
2. Install dependencies:

```bash
pip install -r src/classification/requirements.txt
```

3. Configure `.env` in repo root.

For local Ollama usage (no API key required):

```env
# Optional override (already set in config.yaml by default):
OPENAI_BASE_URL=http://localhost:11434/v1
```

For hosted OpenAI usage:

```env
OPENAI_API_KEY=your_api_key_here
```

## Local 3-model setup (Ollama)
Install and start Ollama, then pull the default local models used by `config.yaml`:

```bash
brew install ollama
ollama serve
ollama pull mistral
ollama pull zephyr
ollama pull llama3.1:8b
```

Note: on Apple silicon with 16GB RAM, `Meta-Llama-3-70B-Instruct` is not practical locally. The default config uses `llama3.1:8b` for the third ensemble slot.

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
    allow_missing_api_key_for_local: true
    api_key_fallback: ollama
    base_url: http://localhost:11434/v1
  models:
    - alias: mistral_openorca
      model_id: mistral
    - alias: zephyr_beta
      model_id: zephyr
    - alias: llama3_70b
      model_id: llama3.1:8b
```

If your local provider uses different IDs, change only `model_id` values in `config.yaml`. No code changes are required.

## Restart behavior
- Fetch step is restartable by cached HTML files in `data/raw_html` (skips unless `--force`).
- Extraction/classification steps skip if final outputs already exist (unless `--force`).
