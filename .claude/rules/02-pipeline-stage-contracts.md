# Pipeline Stage Rules

## Stage Order

1. multilingual search string generation (`src/epp_namelist`)
2. RSS retrieval (`src/rss_retrieval`)
3. domain filtering (`src/classification/domain_filter.py`)
4. headline filtering via batch API (`src/classification/headline_batch.py`)
5. full-text retrieval (`src/fulltext_retrieval`)
6. BERTopic exploration (`src/extraction/bertopic_fulltext.py`)

## Core Runtime Defaults

- Search window: `2005-01-01` to `2025-12-31`
- Headline classification provider: set `classification.provider` to `openai` or `anthropic`
  - OpenAI default model: `gpt-5-nano`
  - Anthropic default model: `claude-haiku-4-5`
- Only one provider per run, no ensemble logic
- Domain blocklist file: `data/domains_removed.csv`
- Blocklist criterion: rows with `is_news_outlet == "No"`

## Headline Batch Architecture

- Provider-agnostic interface: `src/utils/batch_provider.py`
- Provider implementations: `src/utils/openai_batch.py`, `src/utils/anthropic_batch.py`
- Factory: `src/utils/config.py::create_batch_provider()`
- State columns use generic `batch_*` prefixes (not provider-specific)
- Legacy `openai_*` columns are auto-migrated on load

## RSS Rules

- Start with whole-period scan per search string and locale.
- If entries are high, split windows hierarchically using config:
  - coarse years: default `5`
  - day hierarchy: default `30,14,7,3,1`
- Keep manifest-driven resume behavior.
- Aggregate only completed `success` and `empty` windows.

## Headline Batch Rules

- Use strict JSON outputs with `label`, `confidence`, `rationale`.
- Batch names must include dataset key:
  - `<dataset_key>_headline_batch_<nnnn>`
- If batches are still pending in `--stage all`, downstream stages must wait.

## Full-Text Rules

- Retrieve only rows where `final_action` matches config (default `keep`).
- Resolve Google redirect wrappers before fetch when possible.
- Enforce minimum word count from config (default `120`).
- Deduplicate by URL and content hash.

## BERTopic Rules

- Skip modeling if prepared documents < `bertopic.min_documents`.
- Embedding model can vary by pathogen domain.
- Always write summary/timing artifacts, even for skipped runs.
