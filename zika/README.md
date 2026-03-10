# Zika News-Mining MVP

Reproducible MVP pipeline for the first three stages of the Zika news-mining protocol:

1. Retrieve Google News RSS results for `zika` in English from `2010-01-01` to `2015-12-31`
2. Classify headlines with an OpenAI + local LLM ensemble and prepare human validation assets
3. Retrieve article full text for the URLs retained by the ensemble

The pipeline is isolated under `zika/` and is designed to be restartable. Existing raw RSS XML, scored headline outputs, and full-text outputs are reused on rerun.

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
  R/
    01_retrieve_google_rss.R
    02_filter_headlines_ensemble.R
    utils.R
  python/
    03_retrieve_fulltext.py
    scrape_utils.py
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
- required at runtime for stage 3: `pandas`, `pyarrow`, `requests`, `beautifulsoup4`, `lxml`, `readability-lxml`, `trafilatura`
- optional fallback: `newspaper3k`

Local LLM runtime:

- Ollama running locally
- a model such as `qwen2.5:14b` or `deepseek-r1:14b`

## Configuration

1. Copy `zika/config/.env.example` to `zika/config/.env`
2. Fill in at minimum:
   - `OPENAI_API_KEY`
   - `OPENAI_MODEL`
   - `LOCAL_LLM_BASE_URL`
   - `LOCAL_LLM_MODEL`
3. Start Ollama and make sure the configured local model is available

Settings live in `zika/config/zika.yaml`. Environment variables in `.env` override the matching runtime settings.

## Run

From the repository root:

```bash
make -C zika zika-rss
make -C zika zika-filter
make -C zika zika-fulltext
make -C zika zika-all
```

Or from inside `zika/`:

```bash
make zika-rss
make zika-filter
make zika-fulltext
make zika-all
```

## Outputs

Stage 1:

- `data/raw/rss/*.xml`: raw Google News RSS payloads by date window
- `data/intermediate/zika_rss_raw.csv`
- `data/intermediate/zika_rss_raw.parquet`
- `logs/01_retrieve_google_rss.log`

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

## Validation Workflow

The pipeline does not invent gold labels.

1. Run `make -C zika zika-filter`
2. Open `data/validation/zika_headline_validation_sample.csv`
3. Fill `gold_label` using `relevant`, `irrelevant`, or `unsure`
4. Optionally add reviewer notes
5. Rerun `make -C zika zika-filter`

The stage-2 script preserves existing human labels in the validation sample and recomputes evaluation metrics when gold labels are present.

## Notes

- Google News historical RSS availability can vary over time. The retrieval step logs empty windows explicitly and still preserves the raw XML response for each requested chunk.
- Parquet writing in R falls back to the repository Python environment via `uv run python` when the R `arrow` package is not installed.
