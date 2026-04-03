# Pathogen Pipeline in `src`

This folder contains the generalized news pipeline adapted from the Zika workflow for three pathogen domains:

- `human`
- `animal`
- `plant`

The pipeline keeps outputs separated by dataset key, where each dataset key is one domain-language combination:

- `human_en`, `human_fr`, `human_es`, `human_pt`
- `animal_en`, `animal_fr`, `animal_es`, `animal_pt`
- `plant_en`, `plant_fr`, `plant_es`, `plant_pt`

Within each language family bucket, RSS retrieval runs multiple Google News locale variants and merges them back into the same output. For example, `pt` includes both `pt-BR` and `pt-PT` retrieval passes, while `en`, `fr`, and `es` also include multiple regional editions. Locale configuration is under `rss.language_specs` in `pipeline.yaml`.

RSS retrieval uses a hierarchical adaptive scan by default: it starts with one `whole-period` scan per search string across `2005-01-01` to `2025-12-31`. Searches with fewer than `100` RSS items are scraped immediately. Searches with at least `100` RSS items first split into `5-year` periods, and only hot `5-year` periods continue down the `30-day`, `14-day`, `7-day`, `3-day`, and `1-day` hierarchy. Processing is breadth-first by stage, so the pipeline finishes the whole-period scan across the dataset before moving on to `5-year`, then finishes `5-year` before moving on to finer windows. Split parent payloads are kept on disk for auditability, but only `success` and `empty` windows are aggregated into the normalized RSS CSV/Parquet outputs.

## Expected inputs

Create one search workbook per pathogen domain:

- `data/inputs/human_diseases.xlsx`
- `data/inputs/animal_diseases.xlsx`
- `data/inputs/plant_diseases.xlsx`

The pipeline prefers `search_string_<language>` columns such as:

- `search_string_en`
- `search_string_fr`
- `search_string_es`
- `search_string_pt`

If a language-specific column is missing, it falls back to `search_string`.

The pipeline also expects `data/domains_removed.csv` with columns:

- `domain`
- `is_news_outlet`

Rows where `is_news_outlet == "No"` are treated as a blocklist before headline filtering.

## Headline classification setup

Headline classification uses a **batch API** — either OpenAI or Anthropic, configured by `classification.provider` in `pipeline.yaml`. Only one provider is used per run.

Add the appropriate API key to the local `.env` file at repo root:

```dotenv
# For OpenAI (default):
OPENAI_API_KEY=...

# For Anthropic:
ANTHROPIC_API_KEY=...
```

To switch providers, set `classification.provider` in `src/config/pipeline.yaml`:

```yaml
classification:
  provider: "openai"    # or "anthropic"
```

Provider-specific settings (model, max_tokens, batch caps) are under `classification.openai` and `classification.anthropic` respectively. The rest of the pipeline is provider-agnostic — outputs are normalized into a consistent internal schema regardless of which provider runs the classification.

Because the Batch API is asynchronous, the pipeline is intentionally resumable:

1. First run prepares and submits pending batches.
2. Re-run later to hydrate completed batch outputs.
3. Once a dataset has no pending headline rows, downstream full-text and BERTopic stages continue.

## Commands

Run the full pipeline:

```bash
python3 src/run_pipeline.py --stage all
```

Run only RSS retrieval:

```bash
python3 src/run_pipeline.py --stage rss --pathogen-domains human,animal --languages en,fr,es,pt
```

Run only domain filtering + headline batch handling:

```bash
python3 src/run_pipeline.py --stage classify
```

Run full-text retrieval after headline batches are hydrated:

```bash
python3 src/run_pipeline.py --stage fulltext
```

Run BERTopic on finished full-text datasets:

```bash
python3 src/run_pipeline.py --stage bertopic
```

## Output layout

- RSS retrieval: `data/intermediate/rss/<dataset_key>.csv`
- RSS stepwise scan summary: `data/intermediate/rss/<dataset_key>_scan_summary.csv`
- Domain-filtered headlines: `data/intermediate/classification/<dataset_key>_headlines_prefiltered.csv`
- Headline classification state: `data/intermediate/classification/<dataset_key>_headlines_classified.csv`
- Batch registry: `data/intermediate/batches/<dataset_key>/`
- Full text: `data/final/fulltext/<dataset_key>.csv`
- BERTopic outputs: `data/final/bertopic/<dataset_key>/`

## Resume behavior

- RSS retrieval keeps a per-dataset manifest under `data/raw/rss/<dataset_key>/`.
- RSS retrieval manifests begin with one whole-period row per search string and locale. These can expand into `5-year` rows and then into hierarchical child windows at `30-day`, `14-day`, `7-day`, `3-day`, and `1-day`.
- Parent rows marked `split` are treated as complete for resume purposes and are excluded from the aggregated RSS output.
- If you already created a manifest with an older fixed-window strategy and want a clean adaptive run, remove that dataset's RSS manifest before rerunning retrieval.
- Domain filtering reuses its existing output unless `--force` is set.
- Headline filtering reuses the saved classification state and batch registry. Legacy `openai_*` columns in existing state files are auto-migrated to generic `batch_*` columns on load.
- Full-text retrieval reuses successful and failed records and checkpoints every few rows.
- BERTopic skips completed outputs unless `--force` is set.
