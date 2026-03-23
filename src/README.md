# Pathogen Pipeline in `src`

This folder contains the generalized news pipeline adapted from the Zika workflow for three pathogen domains:

- `human`
- `animal`
- `plant`

The pipeline keeps outputs separated by dataset key, where each dataset key is one domain-language combination:

- `human_en`, `human_fr`, `human_es`, `human_pt`
- `animal_en`, `animal_fr`, `animal_es`, `animal_pt`
- `plant_en`, `plant_fr`, `plant_es`, `plant_pt`

Within each language family bucket, RSS retrieval now runs multiple Google News locale variants and merges them back into the same output. For example, `pt` includes both `pt-BR` and `pt-PT` retrieval passes, while `en`, `fr`, and `es` also include multiple regional editions.

RSS retrieval now uses hierarchical adaptive windowing by default: it starts with `30-day` windows, and if a window returns at least `100` RSS items it recursively drills down to `14-day`, then `7-day`, then `3-day`, then `1-day` child windows only for that hot period. Split parent payloads are kept on disk for auditability, but only `success` and `empty` windows are aggregated into the normalized RSS CSV/Parquet outputs.

## Expected inputs

Create one CSV per pathogen domain with a `search_string` column:

- `data/inputs/human_diseases.csv`
- `data/inputs/animal_diseases.csv`
- `data/inputs/plant_diseases.csv`

The pipeline also expects `data/domains_removed.csv` with columns:

- `domain`
- `is_news_outlet`

Rows where `is_news_outlet == "No"` are treated as a blocklist before OpenAI headline filtering.

## OpenAI setup

Add your API key to the local `.env` file at repo root:

```dotenv
OPENAI_API_KEY=...
```

Headline filtering uses OpenAI Batch with `gpt-5-nano` by default. The batch request metadata and saved JSONL filenames include the dataset key so you can quickly see which species-language combination each batch belongs to. Because the Batch API is asynchronous, the pipeline is intentionally resumable:

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
python3 src/run_pipeline.py --stage rss --pathogen-domains human,animal --languages en,fr
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
- Domain-filtered headlines: `data/intermediate/classification/<dataset_key>_headlines_prefiltered.csv`
- Headline classification state: `data/intermediate/classification/<dataset_key>_headlines_classified.csv`
- Batch registry: `data/intermediate/openai_batches/<dataset_key>/`
- Full text: `data/final/fulltext/<dataset_key>.csv`
- BERTopic outputs: `data/final/bertopic/<dataset_key>/`

## Resume behavior

- RSS retrieval keeps a per-dataset manifest under `data/raw/rss/<dataset_key>/`.
- RSS retrieval manifests can contain base `30-day` windows plus hierarchical child windows at `14-day`, `7-day`, `3-day`, and `1-day`. Parent rows marked `split` are treated as complete for resume purposes and are excluded from the aggregated RSS output.
- If you already created a manifest with an older fixed-window strategy and want a clean adaptive run, remove that dataset's RSS manifest before rerunning retrieval.
- Domain filtering reuses its existing output unless `--force` is set.
- Headline filtering reuses the saved classification state and batch registry.
- Full-text retrieval reuses successful and failed records and checkpoints every few rows.
- BERTopic skips completed outputs unless `--force` is set.
