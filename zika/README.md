# Zika News-Mining MVP

Reproducible MVP pipeline for the first three stages of the Zika news-mining protocol, plus an initial BERTopic analysis over the retrieved full-text subset:

1. Retrieve Google News RSS results for `zika` in English from `2010-01-01` to `2015-12-31`
2. Classify headlines with an OpenAI + local LLM ensemble and prepare human validation assets
3. Retrieve article full text for the URLs retained by the ensemble
4. Run BERTopic over the successful full-text subset

The pipeline is isolated under `zika/` and is designed to be restartable. Existing cached Stage 1 payloads, scored headline outputs, and full-text outputs are reused on rerun.

## Structure

```text
zika/
  README.md
  Makefile
  config/
    zika.yaml
    .env.example
  data/
    raw/
      rss/
    intermediate/
    validation/
    final/
  logs/
  report/
    zika_pipeline_report.qmd
  R/
    01_retrieve_google_rss.R  (deprecated wrapper)
    02_filter_headlines_ensemble.R  (legacy)
    utils.R
  python/
    01_retrieve_google_rss.py
    01_parse_google_rss.py
    02_filter_headlines_ensemble.py
    run_stage2_with_patched_ollama.py
    03_retrieve_fulltext.py
    04_bertopic_fulltext.py
    filter_utils.py
    rss_utils.py
    scrape_utils.py
    requirements_stage1.txt
    requirements_stage2.txt
```

## Dependencies

R packages:

- `httr2`
- `xml2`
- `rvest`
- `jsonlite`
- `lubridate`
- `stringr`
- `dplyr`
- `purrr`
- `readr`
- `glue`
- `yaml`
- `digest`
- `tibble`
- `tidyr`

Python packages:

- repo `uv` environment from the project root `pyproject.toml`
- required at runtime for stage 1: `pandas`, `pyarrow`, `requests`, `python-dotenv`, `beautifulsoup4`, `pygooglenews`
- required at runtime for stage 2: `pandas`, `pyarrow`, `requests`, `python-dotenv`, `sentence-transformers`, `scikit-learn`
- required at runtime for stage 3: `pandas`, `pyarrow`, `requests`, `beautifulsoup4`, `lxml`, `readability-lxml`, `trafilatura`
- required at runtime for BERTopic: `bertopic`, `umap-learn`, `hdbscan`, `sentence-transformers`, `scikit-learn`
- optional fallback: `newspaper3k`
- optional for Stage 2 advanced metrics: `krippendorff`

Local LLM runtime:

- Ollama running locally
- a model such as `qwen2.5:14b` or `deepseek-r1:14b`

## Configuration

1. Copy `zika/config/.env.example` to `zika/config/.env`
2. Fill in at minimum:
   - `OPENAI_API_KEY`
   - `OPENAI_MODEL`
   - optional but recommended: `HF_TOKEN` for Hugging Face model caching/rate limits
   - `LOCAL_LLM_PROVIDER`
   - `LOCAL_LLM_BASE_URL`
   - `LOCAL_LLM_MODEL`
   - optional Stage 1 proxy settings:
     - `GOOGLE_NEWS_PROXY_BACKEND=direct|requests|scraping_bee`
     - `GOOGLE_NEWS_HTTP_PROXY`
     - `GOOGLE_NEWS_HTTPS_PROXY`
     - `SCRAPING_BEE_API_KEY`
3. Start Ollama and make sure the configured local model is available

Settings live in `zika/config/zika.yaml`. Environment variables in `.env` override the matching runtime settings. Conservative Stage 1 defaults use direct requests with daily windows and backoff. Proxy rotation is optional infrastructure, not the default path.

For the local model, `LOCAL_LLM_PROVIDER=ollama` uses Ollama endpoints and `LOCAL_LLM_PROVIDER=openai_compatible` uses `/v1/chat/completions` on the configured `LOCAL_LLM_BASE_URL`. That gives you a supported escape hatch if Ollama is unstable on the current machine.

Stage 2 defaults to `gpt-5-nano` via the OpenAI Batch API. OpenAI batch submission/hydration and local DeepSeek scoring now run independently: one run can submit or poll the batch while also continuing local scoring, and later reruns will join the two result streams into the final ensemble as soon as both sides are available for a row.

## Run

From the repository root:

```bash
make -C zika zika-rss-fetch
make -C zika zika-rss-parse
make -C zika zika-rss
make -C zika zika-filter
make -C zika zika-fulltext
make -C zika zika-bertopic
make -C zika zika-report
make -C zika zika-all
```

Or from inside `zika/`:

```bash
make zika-rss-fetch
make zika-rss-parse
make zika-rss
make zika-filter
make zika-fulltext
make zika-bertopic
make zika-report
make zika-all
```

## Outputs

Stage 1:

- `data/raw/rss/*.json`: cached normalized pygooglenews payloads by date window
- `data/raw/rss/zika_rss_manifest.csv`: resumable manifest with status, attempts, HTTP status, retry timing, and cache path per window
- `data/intermediate/zika_rss_raw.csv`
- `data/intermediate/zika_rss_raw.parquet`
- `logs/01_retrieve_google_rss.log`
- `logs/01_parse_google_rss.log`

Stage 2:

- `data/intermediate/zika_headlines_scored.csv`
- `data/intermediate/zika_headlines_scored.parquet`
- `data/validation/zika_headline_validation_sample.csv`
- `data/validation/zika_headline_metrics.json`
- `data/validation/zika_headline_metrics.csv`
- `logs/02_filter_headlines_ensemble.log`

Stage 3:

- `data/final/zika_fulltext.csv`
- `data/final/zika_fulltext.parquet`
- `data/final/zika_fulltext_failed.csv`
- `logs/03_retrieve_fulltext.log`

BERTopic:

- `data/final/zika_bertopic_document_topics.csv`
- `data/final/zika_bertopic_document_topics.parquet`
- `data/final/zika_bertopic_topic_info.csv`
- `data/final/zika_bertopic_summary.json`
- `logs/04_bertopic_fulltext.log`

Stage 3 is resumable. Reruns skip valid cached successes and cached failures by `record_id`, and the log now reports `pending`, `cached success`, `cached failed`, `new_success`, and `new_failed` so a no-op rerun is distinguishable from a fresh fetch.

Report:

- `report/zika_pipeline_report.qmd`

## Validation Workflow

The pipeline does not invent gold labels.

1. Run `make -C zika zika-filter`
2. Open `data/validation/zika_headline_validation_sample.csv`
3. Fill `gold_label` using `relevant`, `irrelevant`, or `unsure`
4. Optionally add reviewer notes
5. Rerun `make -C zika zika-filter`

The stage-2 script preserves existing human labels in the validation sample and recomputes evaluation metrics when gold labels are present.

## Notes

- Stage 1 retrieval is split from parsing. `zika-rss-fetch` only touches Google News and only writes cached JSON plus the manifest. `zika-rss-parse` rebuilds the normalized table from cache without making network calls.
- Stage 2 fails fast on provider errors. If OpenAI or the local LLM is unavailable or returns unusable output, the script checkpoints artifacts, clears invalid pseudo-labels, and exits instead of silently converting failures into `unsure`.
- Stage 2 preflight now surfaces local HTTP 500 response bodies in the error message when possible. That makes Ollama runtime failures easier to distinguish from prompt or parsing issues.
- `make -C zika zika-filter` now runs through a small wrapper when the local provider is Ollama on `127.0.0.1:11434`. The wrapper builds the PR `14604` patched Ollama once into `~/.cache/zika/ollama-pr14604/`, starts it on `127.0.0.1:11437` for the duration of the run, points Stage 2 at that port, and then shuts it down.
- Google News historical RSS availability is rate-limited. Stage 1 detects `429`, `503`, and Google “Sorry” pages, retries the same window with exponential backoff, writes the manifest, and exits cleanly before moving on to adjacent windows.
- Resume is manifest-driven. Completed windows with a valid cache file are skipped. If the first unfinished window has a future `next_eligible_attempt_at`, the fetch step stops there and waits for a later rerun instead of hammering later dates.
- Optional proxy support is available through `requests` proxies or ScrapingBee. The default path remains direct requests with conservative throttling.
- Parquet writing in R falls back to the repository Python environment via `uv run python` when the R `arrow` package is not installed.
- Rendering the report requires the Quarto CLI. On this machine, `quarto` is not currently installed, so the `.qmd` can be edited now and rendered later after installing Quarto.
- The BERTopic step currently models the successful Stage 3 full-text subset, applies a lightweight English-language filter, and truncates each document to a configurable number of words before embedding for more stable runtime.
