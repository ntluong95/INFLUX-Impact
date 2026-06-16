# API Documentation: common_multilingual_names.R

**Purpose**: Enrich pathogen and disease names with multilingual terms (English, Spanish, French, Portuguese) by querying external APIs (NCBI, taxize, Wikidata).

**Main Workflow**:
1. Read raw pathogen/disease data from Excel
2. For each row, enrich with:
   - NCBI Taxonomy UID
   - English common names (via taxize)
   - Wikidata entity IDs
   - Multilingual labels and aliases
3. Write enriched data to Excel

---

## Table of Contents

1. [Configuration & Globals](#configuration--globals)
2. [Utility Functions](#utility-functions)
3. [Caching & Rate Limiting](#caching--rate-limiting)
4. [HTTP & API Functions](#http--api-functions)
5. [NCBI/Taxize Functions](#ncbitaxize-functions)
6. [Wikidata Functions](#wikidata-functions)
7. [Enrichment Functions](#enrichment-functions)
8. [Main Pipeline Functions](#main-pipeline-functions)

---

## Configuration & Globals

### `languages`
**Type**: `character` vector  
**Value**: `c("en", "es", "fr", "pt")`  
**Purpose**: Supported languages for multilingual term extraction.

### `wikidata_languages`
**Type**: `character` vector  
**Value**: `c(languages, "mul")` (i.e., `c("en", "es", "fr", "pt", "mul")`)  
**Purpose**: Languages used for Wikidata queries, including "mul" (multilingual) for universal terms.

### `pathogen_description_regex`
**Type**: `regex` object  
**Pattern**: `"virus|bacteri|parasite|protozo|fung|microorganism|microbe|pathogen|species|genus|taxon"` (case-insensitive)  
**Purpose**: Identifies whether a Wikidata entity description refers to a pathogen (vs. a disease). Used for disambiguation and scoring.

### `disease_description_regex`
**Type**: `regex` object  
**Pattern**: `"disease|infection|syndrome|fever|illness|condition|pathology|disorder"` (case-insensitive)  
**Purpose**: Identifies whether a Wikidata entity description refers to a disease (vs. a pathogen). Used for disambiguation and scoring.

### `taxize_cache`
**Type**: `environment`  
**Purpose**: In-memory cache for NCBI/taxize API responses.

### `wikidata_search_cache`
**Type**: `environment`  
**Purpose**: In-memory cache for Wikidata search results.

### `wikidata_entity_cache`
**Type**: `environment`  
**Purpose**: In-memory cache for Wikidata entity details.

---

## Utility Functions

### `normalize_text(x)`
**Purpose**: Clean and standardize text by removing extra whitespace and control characters.

**Parameters**:
- `x`: Text to normalize (any type convertible to character)

**Returns**: Cleaned character string

**Example**:
```r
normalize_text("  Cholera  \n outbreak  ")  # "Cholera outbreak"
```

---

### `is_blank_text(x)`
**Purpose**: Check if text is empty, NA, or contains only whitespace.

**Parameters**:
- `x`: Text to check

**Returns**: `TRUE` if blank, `FALSE` otherwise

---

### `normalize_lookup_key(x)`
**Purpose**: Normalize text for comparison/lookup operations (lowercase, alphanumeric only).

**Parameters**:
- `x`: Text to normalize

**Returns**: Normalized key suitable for exact matching

**Example**:
```r
normalize_lookup_key("Vibrio cholerae!")  # "vibrio cholerae"
```

---

### `dedupe_terms(x)`
**Purpose**: Remove duplicate terms from a list/vector after normalization.

**Parameters**:
- `x`: Vector or list of terms

**Returns**: Character vector of unique terms

**Example**:
```r
dedupe_terms(c("cholera", "Cholera", "NA", ""))  # "cholera"
```

---

### `split_terms(x, pattern)`
**Purpose**: Split a delimited string into unique terms.

**Parameters**:
- `x`: String to split
- `pattern`: Regex pattern for splitting (e.g., `"\n|,"`)

**Returns**: Character vector of unique terms

**Example**:
```r
split_terms("cholera\nVibrio cholerae", "\n")  # c("cholera", "Vibrio cholerae")
```

---

### `collapse_terms(x)`
**Purpose**: Deduplicate terms and join into a single newline-separated string.

**Parameters**:
- `x`: Vector or list of terms

**Returns**: Single string (newline-separated) or `NA_character_` if empty

---

### `build_search_string(terms)`
**Purpose**: Create a Boolean search string for database queries.

**Parameters**:
- `terms`: Character vector of search terms

**Returns**: String in format `"term1" OR "term2" OR "term3"`

**Example**:
```r
build_search_string(c("cholera", "Vibrio cholerae"))
# '"cholera" OR "Vibrio cholerae"'
```

---

### `extract_first_id(x)`
**Purpose**: Extract the first non-empty ID from a list/vector.

**Parameters**:
- `x`: List or vector of IDs

**Returns**: First valid ID or `NA_character_`

---

### `extract_scalar_chr(x, default = NA_character_)`
**Purpose**: Extract a single scalar character value from complex data structures.

**Parameters**:
- `x`: Any data type (list, vector, etc.)
- `default`: Value to return if extraction fails

**Returns**: Single character value or default

---

### `first_non_blank_text(x, default = NA_character_)`
**Purpose**: Get the first non-blank value from a vector/list.

**Parameters**:
- `x`: Vector or list of values
- `default`: Value to return if all are blank

**Returns**: First non-blank normalized text or default

---

### `%||%` (null coalescing operator)
**Purpose**: Return left operand if not NULL/empty, otherwise return right operand.

**Parameters**:
- `x`: Value to check
- `y`: Default value

**Returns**: `x` if valid, otherwise `y`

**Example**:
```r
NULL %||% "default"  # "default"
"value" %||% "default"  # "value"
```

---

### `blank_term_map()`
**Purpose**: Create an empty named list for storing terms by language.

**Returns**: Named list with one empty character vector per language

**Structure**:
```r
list(en = character(), es = character(), fr = character(), pt = character(), mul = character())
```

---

## Caching & Rate Limiting

### `cache_get_or_set(cache_env, key, fn)`
**Purpose**: Retrieve cached value or compute and cache it.

**Parameters**:
- `cache_env`: Environment used as cache storage
- `key`: Cache key
- `fn`: Function to compute value if not cached

**Returns**: Cached or newly computed value

**Pattern**: Implements memoization for expensive API calls.

---

### `pause_for_api(seconds = 0.05)`
**Purpose**: Pause execution to respect API rate limits.

**Parameters**:
- `seconds`: Sleep duration (default: 50ms)

---

### `service_in_cooldown(service_key)`
**Purpose**: Check if an API service is currently in cooldown mode.

**Parameters**:
- `service_key`: Service identifier (e.g., "wikidata")

**Returns**: `TRUE` if service is in cooldown, `FALSE` otherwise

---

### `activate_service_cooldown(service_key, seconds, request_label, status)`
**Purpose**: Put an API service into cooldown after receiving error responses.

**Parameters**:
- `service_key`: Service identifier
- `seconds`: Cooldown duration
- `request_label`: Label for logging
- `status`: HTTP status code that triggered cooldown

**Side Effects**: Logs cooldown message, sets cooldown timestamp in `service_cooldown_cache`

---

### `timestamped_message(...)`
**Purpose**: Print a timestamped log message to console.

**Parameters**:
- `...`: Arguments passed to `message()`

**Output**: `[YYYY-MM-DD HH:MM:SS] <message>`

---

### `build_retry_policy(req, request_label, service_key = NULL, cooldown_statuses = integer())`
**Purpose**: Configure retry policy with exponential backoff for HTTP requests.

**Parameters**:
- `req`: httr2 request object
- `request_label`: Label for logging
- `service_key`: Service identifier for cooldown management
- `cooldown_statuses`: HTTP status codes that trigger cooldown

**Returns**: Modified request with retry policy

**Retry Behavior**:
- Maximum 3 attempts
- Exponential backoff (1-60 seconds)
- Activates cooldown for specified status codes (e.g., 429 Too Many Requests)

---

## HTTP & API Functions

### `request_json(url, query = list(), request_label = url, service_key = NULL, cooldown_statuses = integer())`
**Purpose**: Make HTTP GET request and parse JSON response with retry logic.

**Parameters**:
- `url`: API endpoint URL
- `query`: Query parameters (named list)
- `request_label`: Label for logging
- `service_key`: Service identifier for rate limiting
- `cooldown_statuses`: HTTP status codes triggering cooldown

**Returns**: Parsed JSON (list) or `NULL` on error

**Features**:
- Automatic retries (up to 3 attempts)
- Rate limiting via cooldown
- User-Agent header: "INFLUX-name-builder/1.0"
- 30-second timeout

---

## NCBI/Taxize Functions

### `fetch_ncbi_uid(scientific_name)`
**Purpose**: Look up NCBI Taxonomy UID for a scientific name.

**Parameters**:
- `scientific_name`: Scientific name (e.g., "Vibrio cholerae")

**Returns**: NCBI UID string (e.g., "666") or `NA_character_`

**Caching**: Results cached in `taxize_cache` with key `"uid::<name>"`

**API**: Uses `taxize::get_uid()` with NCBI Taxonomy database

**Example**:
```r
fetch_ncbi_uid("Vibrio cholerae")  # "666"
```

---

### `fetch_taxize_common_names(scientific_name)`
**Purpose**: Get English common names for a pathogen using NCBI UID.

**Parameters**:
- `scientific_name`: Scientific name (e.g., "Homo sapiens")

**Returns**: Character vector of common names (e.g., `c("human", "man")`) or empty vector

**Caching**: Results cached in `taxize_cache` with key `"common::<name>"`

**API**: Uses `taxize::sci2comm()` to convert scientific name to common names

**Note**: Many bacteria and pathogens don't have common names in NCBI.

**Example**:
```r
fetch_taxize_common_names("Homo sapiens")  # c("human", "man")
fetch_taxize_common_names("Vibrio cholerae")  # character() (no common names)
```

---

## Wikidata Functions

### `search_wikidata_entities(query_text, language = "en", limit = 5)`
**Purpose**: Search Wikidata for entities matching a query.

**Parameters**:
- `query_text`: Search term (e.g., "cholera")
- `language`: Language code for search (default: "en")
- `limit`: Maximum results to return (default: 5)

**Returns**: Tibble with columns:
- `id`: Wikidata entity ID (e.g., "Q12090")
- `label`: Primary label
- `description`: Entity description
- `alias`: Alternative name
- `match_text`: Text that matched the query
- `match_type`: Type of match ("label", "alias", etc.)

**Caching**: Results cached in `wikidata_search_cache`

**API**: Wikidata `wbsearchentities` endpoint

**Rate Limiting**: Respects cooldown, returns empty tibble if in cooldown

---

### `score_wikidata_hit(label, description, alias, match_text, match_type, query_text, kind)`
**Purpose**: Score a Wikidata search result for relevance.

**Parameters**:
- `label`, `description`, `alias`, `match_text`, `match_type`: Hit metadata
- `query_text`: Original search query
- `kind`: Entity type ("pathogen" or "disease")

**Returns**: Numeric score (higher = more relevant)

**Scoring Criteria**:
| Criterion | Points |
|-----------|--------|
| Exact label match | +10 |
| Exact match in `match_text` | +8 |
| Exact alias match | +6 |
| Partial label containment | +2 |
| Match type is "label" | +2 |
| Match type is "alias" | +1 |
| Description matches kind (pathogen/disease regex) | +3 |

**Purpose**: Helps disambiguate entities (e.g., "Ebola" virus vs. Ebola disease).

---

### `resolve_wikidata_id(candidate_terms, kind)`
**Purpose**: Find the best-matching Wikidata entity ID for a set of candidate terms.

**Parameters**:
- `candidate_terms`: Character vector of search terms
- `kind`: Entity type ("pathogen" or "disease")

**Returns**: Wikidata ID (e.g., "Q12090") or `NA_character_` if no match found

**Algorithm**:
1. Search Wikidata for each candidate term
2. Score all results using `score_wikidata_hit()`
3. Return highest-scoring ID (if score ≥ 3)

**Example**:
```r
resolve_wikidata_id("cholera", kind = "disease")  # "Q12090"
resolve_wikidata_id("Vibrio cholerae", kind = "pathogen")  # "Q160821"
```

---

### `fetch_wikidata_entity(entity_id)`
**Purpose**: Fetch full entity details from Wikidata.

**Parameters**:
- `entity_id`: Wikidata ID (e.g., "Q12090")

**Returns**: List containing entity data (labels, aliases, descriptions) or `NULL`

**Caching**: Results cached in `wikidata_entity_cache`

**API**: Wikidata `Special:EntityData/<id>.json` endpoint

**Example**:
```r
entity <- fetch_wikidata_entity("Q12090")
entity$labels$en$value  # "cholera"
```

---

### `extract_wikidata_terms(entity_id, language)`
**Purpose**: Extract all terms (label + aliases) for a Wikidata entity in a specific language.

**Parameters**:
- `entity_id`: Wikidata ID
- `language`: Language code (e.g., "en", "es")

**Returns**: Character vector of unique terms

**Example**:
```r
extract_wikidata_terms("Q12090", "en")
# c("cholera")
extract_wikidata_terms("Q12090", "es")
# c("cólera", "colera")
```

---

### `fetch_multilingual_terms(entity_id)`
**Purpose**: Fetch terms for a Wikidata entity in all supported languages.

**Parameters**:
- `entity_id`: Wikidata ID

**Returns**: Named list with one character vector per language

**Structure**:
```r
list(
  en = c("cholera"),
  es = c("cólera", "colera"),
  fr = c("choléra"),
  pt = c("cólera"),
  mul = character()
)
```

---

### `language_specific_terms(language, base_pathogen_terms, english_pathogen_terms, english_disease_terms_value, pathogen_term_map, disease_term_map)`
**Purpose**: Combine pathogen and disease terms for a specific language.

**Parameters**:
- `language`: Target language code
- `base_pathogen_terms`: Original scientific names
- `english_pathogen_terms`: English pathogen terms (including common names)
- `english_disease_terms_value`: English disease terms
- `pathogen_term_map`: Multilingual pathogen terms from Wikidata
- `disease_term_map`: Multilingual disease terms from Wikidata

**Returns**: Combined character vector of all terms for the target language

**Logic**:
- For English: Include base terms + English terms + Wikidata terms
- For other languages: Use Wikidata terms, fall back to English if unavailable
- Always deduplicate

---

## Enrichment Functions

### `enrich_multilingual_name_row(pathogen_scientific_name_en, pathogen_subtypes_names_en, aka_en, disease_name_en)`
**Purpose**: Enrich a single row of pathogen/disease data with multilingual terms.

**Parameters**:
- `pathogen_scientific_name_en`: Scientific name (e.g., "Vibrio cholerae")
- `pathogen_subtypes_names_en`: Subtype names (optional)
- `aka_en`: Alternative names/aliases (optional)
- `disease_name_en`: Disease name in English (e.g., "cholera")

**Returns**: Tibble with one row and the following columns:

| Column | Description |
|--------|-------------|
| `pathogen_uid_ncbi` | NCBI Taxonomy UID |
| `pathogen_common_names_taxize_en` | English common names from taxize |
| `pathogen_wikidata_id` | Wikidata ID for pathogen |
| `disease_wikidata_id` | Wikidata ID for disease |
| `pathogen_terms_en/es/fr/pt` | Pathogen terms by language |
| `disease_terms_en/es/fr/pt` | Disease terms by language |
| `search_string_en/es/fr/pt` | Boolean search strings for each language |

**Workflow**:
1. Extract candidate search terms from input
2. Fetch NCBI UID for pathogen
3. Fetch English common names via taxize
4. Resolve Wikidata IDs for pathogen and disease
5. Fetch multilingual terms from Wikidata
6. Combine terms by language
7. Build search strings

**Example**:
```r
enrich_multilingual_name_row(
  pathogen_scientific_name_en = "Vibrio cholerae",
  pathogen_subtypes_names_en = NA,
  aka_en = NA,
  disease_name_en = "cholera"
)
```

---

### `run_rowwise_enrichment(data, enrich_fn, label_fn, task_label)`
**Purpose**: Apply enrichment function to each row of a data frame with progress logging.

**Parameters**:
- `data`: Data frame to process
- `enrich_fn`: Function to apply to each row
- `label_fn`: Function to generate row labels for logging
- `task_label`: Task name for progress messages

**Returns**: Data frame with enrichment results (one row per input row)

**Features**:
- Progress logging with timestamps
- Error handling with descriptive messages
- Row-by-row processing (allows API rate limiting)

---

## Main Pipeline Functions

### `prepare_name_seed_workbook(raw_input_path)`
**Purpose**: Read and validate raw input Excel file.

**Parameters**:
- `raw_input_path`: Path to input Excel file

**Returns**: Cleaned data frame with required columns

**Required Input Columns**:
- `pathogen_scientific_name_en`
- `pathogen_subtypes_names_en`
- `aka_en`
- `disease_name_en`
- `search_string_en`

**Processing**:
1. Read Excel file
2. Normalize column names
3. Validate required columns exist
4. Select only required columns
5. Normalize all text values

**Errors**:
- Missing `search_string_en` column
- Missing any required columns

---

### `build_multilingual_name_list(raw_input_path, output_path)`
**Purpose**: Main entry point - build complete multilingual name list from raw input.

**Parameters**:
- `raw_input_path`: Path to input Excel file (`human_diseases_raw.xlsx`)
- `output_path`: Path for output Excel file (`human_diseases.xlsx`)

**Returns**: Enriched data frame (invisibly)

**Side Effects**: Writes Excel file to `output_path`

**Workflow**:
1. Load and validate input with `prepare_name_seed_workbook()`
2. Extract input columns for enrichment
3. Process each row with `run_rowwise_enrichment()`
4. Combine original and enriched data
5. Reorder columns for readability
6. Write output Excel file

**Output Columns** (in order):
1. Original columns from input
2. `search_string`, `search_string_en/es/fr/pt` (Boolean search strings)
3. `pathogen_uid_ncbi`
4. `pathogen_common_names_taxize_en`
5. `pathogen_wikidata_id`, `disease_wikidata_id`
6. `pathogen_terms_en/es/fr/pt`
7. `disease_terms_en/es/fr/pt`

**Example**:
```r
build_multilingual_name_list(
  raw_input_path = here::here("data/inputs/human_diseases_raw.xlsx"),
  output_path = here::here("data/inputs/human_diseases.xlsx")
)
```

---

## Helper Functions (Pathogen/Disease Term Extraction)

### `pathogen_search_candidates(pathogen_scientific_name_en, pathogen_subtypes_names_en, aka_en)`
**Purpose**: Extract all candidate terms for pathogen Wikidata search.

**Returns**: Deduplicated character vector of search terms

---

### `pathogen_base_terms(pathogen_scientific_name_en, pathogen_subtypes_names_en)`
**Purpose**: Extract base scientific names (without aliases).

**Returns**: Deduplicated character vector

---

### `pathogen_english_alias_terms(aka_en)`
**Purpose**: Extract English aliases from "also known as" field.

**Returns**: Character vector of aliases

---

### `disease_english_terms(disease_name_en)`
**Purpose**: Extract disease search terms.

**Returns**: Character vector of disease terms

---

## Configuration Constants

### `retry_wait_cap_seconds`
**Value**: `30`  
**Purpose**: Maximum wait time between API retries (seconds)

### `retry_max_tries`
**Value**: `3`  
**Purpose**: Maximum number of API request attempts

### `service_cooldown_cache`
**Type**: `environment`  
**Purpose**: Stores cooldown timestamps for API services

---

## Error Handling

- **API failures**: Return `NULL`, `NA`, or empty vector (graceful degradation)
- **Rate limiting**: Cooldown mode skips requests until cooldown expires
- **Invalid input**: Returns `NA` or empty results
- **Network errors**: Retries up to 3 times with exponential backoff

---

## Caching Strategy

Three separate caches (all in-memory environments):

1. **`taxize_cache`**: NCBI UIDs and common names
2. **`wikidata_search_cache`**: Wikidata search results
3. **`wikidata_entity_cache`**: Full Wikidata entity data

Cache keys are normalized to ensure consistency:
- NCBI: `"uid::<normalized_name>"`
- Common names: `"common::<normalized_name>"`
- Wikidata search: `"<language>::<normalized_query>"`
- Wikidata entity: `<normalized_entity_id>`

---

## API Rate Limiting

- **NCBI/taxize**: 50ms delay between requests (or 110ms without API key)
- **Wikidata**: 50ms delay, with cooldown on HTTP 429
- **Cooldown**: Activated on rate-limit responses, skips requests until cooldown expires

---

## Dependencies

Required R packages (loaded via `pacman::p_load`):
- `here`: Path management
- `httr2`: HTTP requests
- `jsonlite`: JSON parsing
- `readxl`: Excel input
- `taxize`: Taxonomic name resolution
- `tidyverse`: Data manipulation (dplyr, purrr, stringr, tibble)
- `writexl`: Excel output
