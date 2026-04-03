# EPP Name List Scripts

This folder contains scripts that build multilingual pathogen and disease name
lists for the Google RSS retrieval stage.

## Current Status

- `common_multilingual_names.R`: shared helper logic
- `human_diseases.R`: implemented
- `animal_diseases.R`: implemented
- `plant_diseases.R`: implemented with an EPPO workflow

## What These Scripts Do

The implemented scripts start from curated raw workbooks and produce
search-ready multilingual query workbooks.

For each row, the workflow expands:

- pathogen scientific names
- pathogen subtype names
- pathogen common names and aliases
- disease names
- multilingual equivalents in English, Spanish, French, and Portuguese
- Boolean search strings for each language

## Inputs

The current implemented scripts read:

- `data/inputs/human_diseases_raw.xlsx`
- `data/inputs/animal_diseases_raw.xlsx`
- `data/inputs/plant_diseases_raw.xlsx`

Expected seed columns for the shared human/animal helper:

- `pathogen_scientific_name_en`
- `pathogen_subtypes_names_en`
- `aka_en`
- `disease_name_en`
- `search_string_en`

Important import rule:

- the workbook may contain extra columns after `search_string_en`
- those extra columns are dropped immediately after import
- only the seed columns up to `search_string_en` are kept for name expansion

## Outputs

The scripts write:

- `data/inputs/human_diseases.xlsx`
- `data/inputs/animal_diseases.xlsx`
- `data/inputs/plant_diseases.xlsx`

Key output columns include:

- `search_string`
- `search_string_en`
- `search_string_es`
- `search_string_fr`
- `search_string_pt`
- `pathogen_uid_ncbi`
- `pathogen_common_names_taxize_en`
- `pathogen_wikidata_id`
- `disease_wikidata_id`
- `pathogen_terms_en`, `pathogen_terms_es`, `pathogen_terms_fr`, `pathogen_terms_pt`
- `disease_terms_en`, `disease_terms_es`, `disease_terms_fr`, `disease_terms_pt`

Compatibility note:

- `search_string` is currently set to the English query string
- the Python RSS pipeline now prefers `search_string_<language>` when available

## Plant Workflow 

The plant workbook uses a different seed strategy from the human and animal
workbooks.

### English seed columns

`plant_diseases.R` creates:

- `pathogen_scientific_name_en`
  - first nonblank value in this priority order:
    - `EPPO_Species`
    - `COL_scientificName`
    - `GBIF_species`
    - `IT IS_scientificName`
    - `NCBI_ScientificName`
- `aka_en`
  - merged from:
    - `IT IS_commonNames`
    - `NCBI_CommonName`
  - terms are lower-cased, deduplicated, and collapsed into one alias field

### Multilingual plant names

The plant script uses an EPPO-first strategy for multilingual common names.

Workflow:
#NOTE example link
- if `EPPO_taxonID` is present:
  - request `https://gd.eppo.int/taxon/<EPPO_taxonID>`
  - parse the `#tbcommon` table
  - collect common names by language
- map EPPO rows into pipeline languages:
  - `en`: `English`, `English (AU)`, `English (GB)`, `English (US)`
  - `es`: `Spanish`, `Spanish (HN)`
  - `fr`: `French`
  - `pt`: `Portuguese`
- always keep the Latin scientific name in all language search strings
- if EPPO is missing or a language has no EPPO names:
  - fall back to Wikidata labels and aliases using the scientific names and English aliases

This means the plant workflow is:

- EPPO for multilingual common names when an EPPO taxon anchor exists
- Wikidata for fallback multilingual aliases
- Latin scientific names as the language-stable base term

## Retrieval Strategy

The shared helper uses a two-source strategy.

### 1. NCBI taxonomy via `taxize`

Used for pathogen-side taxonomy resolution and English common names.

Workflow:

- take the English pathogen scientific name as the main taxonomic anchor
- resolve it to an NCBI taxon ID with `taxize::get_uid()`
- retrieve English common names with `taxize::sci2comm()`

Why this is used:

- NCBI taxonomy is a strong source for scientific pathogen naming
- `taxize` provides a convenient R interface for scientific and common names

### 2. Wikidata

Used for multilingual labels and aliases for both pathogens and diseases.
#NOTE example link
https://www.wikidata.org/wiki/Q20645236

Workflow:

- build English search candidates from:
  - scientific name
  - subtype names
  - English aliases
  - disease names
- search Wikidata items with the MediaWiki `wbsearchentities` API
- score candidate matches using:
  - exact label match
  - exact alias or match-text match
  - whether the entity description looks like a pathogen or a disease
- retrieve the chosen item from `Special:EntityData/<QID>.json`
- extract labels and aliases for:
  - `en`
  - `es`
  - `fr`
  - `pt`
  - `mul` for multilingual aliases

Why this is used:

- disease names are not handled well by taxonomic databases alone
- Wikidata is useful for multilingual synonyms and alternate labels

## How Search Strings Are Built

One search string is built per language.

For each row:

- start with pathogen base terms:
  - `pathogen_scientific_name_en`
  - `pathogen_subtypes_names_en`
- add English pathogen aliases from `aka_en`
- add English pathogen common names from NCBI when available
- add disease names from `disease_name_en`
- add Wikidata labels and aliases in the target language
- fall back to English when a target-language synonym is missing
- deduplicate case-insensitively
- quote each term
- join with ` OR `

Example pattern:

```text
"Vibrio cholerae" OR "cholera" OR "colera"
```

## Term Splitting Rules

Different columns use different splitting rules.

- `pathogen_scientific_name_en`: split on newline
- `pathogen_subtypes_names_en`: split on newline
- `aka_en`: split on newline or comma
- `disease_name_en`: split on newline or comma

This preserves subtype lines such as `H1, H3` when they are intentionally
grouped together on one line.

## Shared File Structure

The folder now uses a shared helper plus thin domain wrappers.

- `common_multilingual_names.R`
  - loads packages and `.env`
  - configures `taxize`
  - imports the raw workbook
  - drops columns after `search_string_en`
  - resolves names via NCBI and Wikidata
  - builds multilingual search strings
  - writes the final `.xlsx` file
- `human_diseases.R`
  - calls the shared helper with the human workbook paths
- `animal_diseases.R`
  - calls the shared helper with the animal workbook paths
- `plant_diseases.R`
  - builds English seed columns from plant taxonomy sources
  - scrapes EPPO common names by `EPPO_taxonID`
  - falls back to Wikidata when EPPO is unavailable
  - writes the final `.xlsx` file

## Caching and API Behavior

The shared helper caches repeated lookups during a run in in-memory
environments.

Cached lookups include:

- NCBI UID resolution
- NCBI common-name lookup
- Wikidata search results
- Wikidata entity payloads

This reduces repeated API calls when the same pathogen or disease appears more
than once.

## Entrez API Key

The shared helper looks for `ENTREZ_KEY` in the project `.env` file:

```env
ENTREZ_KEY=your_ncbi_key_here
```

Behavior:

- if `.env` exists, it is loaded with `readRenviron()`
- if `ENTREZ_KEY` is available, the script uses a faster NCBI sleep interval
- if no key is present, the script still runs, but more slowly

## Running the Scripts

## WOAH Animal Disease Retrieval

`retrieve_woah_animal_diseases.py` adds a live retrieval step for the WOAH
animal disease portal at:

- `https://www.woah.org/en/what-we-do/animal-health-and-welfare/animal-diseases/`

Implementation strategy:

- use Playwright to render the paginated WOAH list and save one HTML fragment
  per page under `data/raw_html/woah_animal_diseases/`
- use `ScrapeGraphAI` with an Ollama model to extract the visible disease cards
  from each rendered page fragment
- deduplicate cards across pages and write a tidy output table

### Why the script uses a side environment

This repository is pinned to Python `3.14`, while `ScrapeGraphAI` currently
imports cleanly in a Python `3.11` side environment on this machine.

Recommended setup from the repo root:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv venv --python /Users/luongnguyen/.local/share/uv/python/cpython-3.11-macos-aarch64-none/bin/python3.11 .venv-sga311
UV_CACHE_DIR=/tmp/uv-cache uv pip install --python .venv-sga311/bin/python scrapegraphai playwright pydantic
.venv-sga311/bin/playwright install chromium
```

### Ollama requirements

The script expects:

- an Ollama chat model, for example `llama3.2`
- an Ollama embedding model, default `nomic-embed-text`
- an Ollama base URL like `http://127.0.0.1:11434`

If local Ollama is unstable on Apple silicon in this repo, prefer pointing the
script at the patched or alternate Ollama server you are already using, for
example a local port such as `http://127.0.0.1:11437`.

### Example run

```bash
.venv-sga311/bin/python src/epp_namelist/retrieve_woah_animal_diseases.py \
  --ollama-model llama3.2 \
  --embedding-model nomic-embed-text \
  --ollama-base-url http://127.0.0.1:11434
```

Outputs:

- `data/inputs/animal_diseases_woah_raw.csv`
- `data/inputs/animal_diseases_woah_raw.parquet`
- `data/inputs/animal_diseases_woah_raw.json`
- `data/raw_html/woah_animal_diseases/page_*.html`
- `data/logs/woah_animal_diseases.log`

From the repository root:

```bash
Rscript src/epp_namelist/human_diseases.R
Rscript src/epp_namelist/animal_diseases.R
Rscript src/epp_namelist/plant_diseases.R
```

Recommended:

- run from the project root so `here::here()` resolves paths correctly
- ensure the required R packages are installed
- keep `.env` out of version control

## High-Level Flow

At a high level, the implemented scripts do this:

1. Load packages and `.env`
2. Configure `taxize`
3. Read the raw workbook
4. Normalize column names and whitespace
5. Drop extra columns after `search_string_en`
6. Resolve pathogen taxonomy with NCBI via `taxize`
7. Resolve pathogen and disease concepts in Wikidata
8. Extract multilingual labels and aliases
9. Build language-specific term sets
10. Build `search_string_en`, `search_string_es`, `search_string_fr`, `search_string_pt`
11. Export the final `.xlsx` workbook

## Limitations

- Wikidata matching is heuristic, not perfect
- some pathogens or diseases may have no multilingual labels
- some scientific updates or taxonomic revisions may not be captured uniformly
- the raw workbook is still manually curated and remains the main starting point
- EPPO coverage is strong but not complete, so some plant rows still depend on Wikidata fallback
- EPPO language coverage varies by taxon, so some `search_string_<language>` fields may contain only the Latin scientific name plus fallback aliases

## Plant Source Notes

The plant script intentionally does not mirror the human/animal helper exactly.

- plant rows are anchored first with existing taxonomy columns already present in the workbook
- EPPO provides the strongest multilingual common-name source because the `#tbcommon` table is curated per taxon
- Wikidata is still useful as a fallback when `EPPO_taxonID` is missing or when EPPO does not provide one of the target languages
