# Package loading --------------------------------------------------------
pacman::p_load(
  here,
  httr2,
  jsonlite,
  readxl,
  taxize,
  tidyverse,
  writexl
)

env_file <- here::here(".env")
if (file.exists(env_file)) {
  readRenviron(env_file)
}

entrez_key <- Sys.getenv("ENTREZ_KEY", unset = "")
if (nzchar(entrez_key)) {
  options(ENTREZ_KEY = entrez_key)
}

taxize::taxize_options(
  taxon_state_messages = FALSE,
  ncbi_sleep = if (nzchar(entrez_key)) 0.11 else 0.334,
  quiet = TRUE
)

languages <- c("en", "es", "fr", "pt")
wikidata_languages <- c(languages, "mul")

pathogen_description_regex <- stringr::regex(
  paste(
    c(
      "virus",
      "bacteri",
      "parasite",
      "protozo",
      "fung",
      "microorganism",
      "microbe",
      "pathogen",
      "species",
      "genus",
      "taxon"
    ),
    collapse = "|"
  ),
  ignore_case = TRUE
)
# NOTE
# This regex is used to **classify whether a Wikidata entity is a disease** (vs. a pathogen). When searching Wikidata, the code checks if the entity's description contains any of these keywords to determine if it's a disease entity, which helps with:

# 1. **Disambiguation** - e.g., "Ebola" could refer to the virus (pathogen) or Ebola disease
# 2. **Scoring** - Entities with disease-related descriptions get bonus points when searching for diseases
disease_description_regex <- stringr::regex(
  paste(
    c(
      "disease",
      "infection",
      "syndrome",
      "fever",
      "illness",
      "condition",
      "pathology",
      "disorder"
    ),
    collapse = "|"
  ),
  ignore_case = TRUE
)

#' Return a fallback value when the left side is missing
#'
#' Treats `NULL` and zero-length values as absent and returns `y` in those
#' cases. Otherwise it returns `x` unchanged.
#'
#' @param x Primary value.
#' @param y Fallback value.
#'
#' @return `x` when present, otherwise `y`.
`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0) {
    return(y)
  }
  x
}

#' Normalize text for downstream comparisons and output
#'
#' Converts input to character, replaces control whitespace with spaces, and
#' squishes repeated whitespace.
#'
#' @param x Value to normalize.
#'
#' @return A cleaned character vector.
normalize_text <- function(x) {
  x %>%
    as.character() %>%
    stringr::str_replace_all("[\r\n\t]+", " ") %>%
    stringr::str_squish()
}

#' Test whether text is effectively blank
#'
#' Uses `normalize_text()` so whitespace-only strings are treated as empty.
#'
#' @param x Value to check.
#'
#' @return `TRUE` when the value is empty, `NA`, or whitespace only.
is_blank_text <- function(x) {
  value <- normalize_text(x)
  length(value) == 0 || is.na(value) || identical(value, "")
}

#' Build a normalized lookup key
#'
#' Lowercases text, removes non-alphanumeric characters, and squishes
#' whitespace so values can be compared consistently.
#'
#' @param x Value to normalize for exact matching.
#'
#' @return A normalized lookup key.
normalize_lookup_key <- function(x) {
  value <- normalize_text(x)
  if (length(value) == 0 || is.na(value)) {
    return("")
  }

  value %>%
    stringr::str_to_lower() %>%
    stringr::str_replace_all("[^[:alnum:]]+", " ") %>%
    stringr::str_squish()
}

#' Deduplicate a set of terms case-insensitively
#'
#' Flattens list-like inputs, drops blank values, normalizes display
#' whitespace, and keeps the first case-insensitive occurrence of each term.
#'
#' @param x Vector or list of candidate terms.
#'
#' @return A character vector of unique normalized terms.
dedupe_terms <- function(x) {
  terms <- x %>%
    unlist(use.names = FALSE) %>%
    as.character() %>%
    purrr::discard(~ is.na(.x) || identical(normalize_text(.x), "")) %>%
    purrr::map_chr(normalize_text)

  if (length(terms) == 0) {
    return(character())
  }

  terms[!duplicated(stringr::str_to_lower(terms))]
}

#' Split a delimited text field into unique terms
#'
#' Blank input returns an empty character vector. Nonblank input is split on
#' `pattern` and deduplicated with `dedupe_terms()`.
#'
#' @param x String to split.
#' @param pattern Regular expression used for splitting.
#'
#' @return A character vector of unique terms.
split_terms <- function(x, pattern) {
  if (is.na(x) || is_blank_text(x)) {
    return(character())
  }

  stringr::str_split(as.character(x), pattern = pattern, simplify = FALSE)[[
    1
  ]] %>%
    dedupe_terms()
}

#' Collapse terms into a newline-delimited string
#'
#' Deduplicates the supplied terms and joins them with newline separators for
#' workbook output.
#'
#' @param x Vector or list of terms.
#'
#' @return A single newline-separated string or `NA_character_` if empty.
collapse_terms <- function(x) {
  values <- dedupe_terms(x)
  if (length(values) == 0) {
    return(NA_character_)
  }
  paste(values, collapse = "\n")
}

#' Build a Boolean search string from terms
#'
#' Each unique term is quoted and joined with ` OR ` so the result can be used
#' in downstream search systems.
#'
#' @param terms Character vector of search terms.
#'
#' @return A Boolean search string or `NA_character_` when no terms remain.
build_search_string <- function(terms) {
  values <- dedupe_terms(terms)
  if (length(values) == 0) {
    return(NA_character_)
  }

  values <- stringr::str_replace_all(values, '"', '\\"')
  paste(paste0('"', values, '"'), collapse = " OR ")
}

#' Create an empty multilingual term map
#'
#' Initializes one empty character vector per Wikidata language key.
#'
#' @return A named list keyed by `wikidata_languages`.
blank_term_map <- function() {
  purrr::set_names(
    vector("list", length(wikidata_languages)),
    wikidata_languages
  ) %>%
    purrr::map(~ character())
}

#' Read through a cache or populate it lazily
#'
#' Checks `cache_env` for `key` and returns the cached value when present.
#' Otherwise it evaluates `fn()`, stores the result, and returns it.
#'
#' @param cache_env Environment used as cache storage.
#' @param key Cache key.
#' @param fn Zero-argument function used to compute the value on a cache miss.
#'
#' @return The cached or newly computed value.
cache_get_or_set <- function(cache_env, key, fn) {
  if (exists(key, envir = cache_env, inherits = FALSE)) {
    return(get(key, envir = cache_env, inherits = FALSE))
  }

  value <- fn()
  assign(key, value, envir = cache_env)
  value
}

#' Pause briefly between API requests
#'
#' @param seconds Number of seconds to sleep.
#'
#' @return Invisibly returns the result of `Sys.sleep()`.
pause_for_api <- function(seconds = 0.05) {
  Sys.sleep(seconds)
}

retry_wait_cap_seconds <- 30
retry_max_tries <- 3
service_cooldown_cache <- new.env(parent = emptyenv())

#' Emit a timestamped progress message
#'
#' Concatenates the supplied values and writes a timestamped line to the
#' console.
#'
#' @param ... Message fragments passed to `paste0()`.
#'
#' @return Invisibly returns `NULL`.
timestamped_message <- function(...) {
  message(sprintf(
    "[%s] %s",
    format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    paste0(..., collapse = "")
  ))
  flush.console()
}

#' Return the first nonblank text value
#'
#' Flattens the input, drops blank values, and normalizes the first remaining
#' entry.
#'
#' @param x Vector or list of values.
#' @param default Value to return when no nonblank text is found.
#'
#' @return A single normalized character value or `default`.
first_non_blank_text <- function(x, default = NA_character_) {
  values <- x %>%
    unlist(use.names = FALSE) %>%
    as.character()

  values <- values[!is.na(values) & normalize_text(values) != ""]
  if (length(values) == 0) {
    return(default)
  }

  normalize_text(values[[1]])
}

#' Check whether a service is currently cooling down
#'
#' @param service_key Service identifier stored in `service_cooldown_cache`.
#'
#' @return `TRUE` when the current time is still before the stored cooldown
#'   timestamp.
service_in_cooldown <- function(service_key) {
  if (is.null(service_key) || is.na(service_key) || service_key == "") {
    return(FALSE)
  }

  cooldown_until <- get0(
    service_key,
    envir = service_cooldown_cache,
    inherits = FALSE,
    ifnotfound = NULL
  )

  !is.null(cooldown_until) && Sys.time() < cooldown_until
}

#' Activate cooldown for a rate-limited service
#'
#' Stores a future cooldown timestamp for `service_key` and logs the event.
#'
#' @param service_key Service identifier.
#' @param seconds Cooldown duration in seconds.
#' @param request_label Human-readable label for log messages.
#' @param status HTTP status code that triggered cooldown.
#'
#' @return Invisibly returns the cooldown timestamp, or `NULL` for blank keys.
activate_service_cooldown <- function(
  service_key,
  seconds,
  request_label,
  status
) {
  if (is.null(service_key) || is.na(service_key) || service_key == "") {
    return(invisible(NULL))
  }

  cooldown_seconds <- seconds
  if (
    is.na(cooldown_seconds) ||
      !is.finite(cooldown_seconds) ||
      cooldown_seconds <= 0
  ) {
    cooldown_seconds <- retry_wait_cap_seconds
  }

  cooldown_until <- Sys.time() + cooldown_seconds
  previous_until <- get0(
    service_key,
    envir = service_cooldown_cache,
    inherits = FALSE,
    ifnotfound = NULL
  )

  assign(service_key, cooldown_until, envir = service_cooldown_cache)

  if (
    is.null(previous_until) ||
      cooldown_until > previous_until ||
      Sys.time() >= previous_until
  ) {
    timestamped_message(
      request_label,
      " returned HTTP ",
      status,
      "; skipping ",
      service_key,
      " lookups until ",
      format(cooldown_until, "%Y-%m-%d %H:%M:%S"),
      "."
    )
  }

  invisible(cooldown_until)
}

#' Attach retry behavior to an httr2 request
#'
#' Configures bounded retries, exponential backoff, and optional cooldown
#' activation for specified HTTP statuses.
#'
#' @param req An `httr2` request object.
#' @param request_label Human-readable label for logging.
#' @param service_key Optional service identifier used for cooldown tracking.
#' @param cooldown_statuses Integer vector of statuses that should trigger
#'   cooldown.
#'
#' @return The request object with retry behavior configured.
build_retry_policy <- function(
  req,
  request_label,
  service_key = NULL,
  cooldown_statuses = integer()
) {
  httr2::req_retry(
    req,
    max_tries = retry_max_tries,
    is_transient = function(resp) {
      status <- httr2::resp_status(resp)

      if (!is.null(service_key) && status %in% cooldown_statuses) {
        activate_service_cooldown(
          service_key = service_key,
          seconds = httr2:::resp_retry_after(resp),
          request_label = request_label,
          status = status
        )
        return(FALSE)
      }

      status %in% c(429, 503)
    },
    backoff = function(i) {
      min(
        round(min(stats::runif(1, min = 1, max = 2^i), 60), 1),
        retry_wait_cap_seconds
      )
    },
    after = function(resp) {
      delay <- httr2:::resp_retry_after(resp)
      if (is.na(delay)) {
        return(NA_real_)
      }

      clipped_delay <- max(0, min(delay, retry_wait_cap_seconds))
      if (is.finite(delay) && delay > retry_wait_cap_seconds) {
        timestamped_message(
          request_label,
          " requested Retry-After=",
          delay,
          "s; capping wait to ",
          clipped_delay,
          "s."
        )
      }

      clipped_delay
    }
  )
}

#' Enrich a data frame row by row with progress logging
#'
#' Applies `enrich_fn` to each row individually so API-backed work can proceed
#' with clear row-level logging and error context.
#'
#' @param data Data frame of rows to enrich.
#' @param enrich_fn Function applied to each row via `purrr::pmap_dfr()`.
#' @param label_fn Function that returns row label candidates for logging.
#' @param task_label Task name shown in progress messages.
#'
#' @return A tibble formed by binding the per-row enrichment results.
run_rowwise_enrichment <- function(data, enrich_fn, label_fn, task_label) {
  n_rows <- nrow(data)
  if (n_rows == 0) {
    return(tibble::tibble())
  }

  timestamped_message(task_label, ": starting ", n_rows, " rows.")

  results <- vector("list", n_rows)
  for (i in seq_len(n_rows)) {
    row <- data[i, , drop = FALSE]
    row_label <- first_non_blank_text(label_fn(row), default = "<unnamed>")

    timestamped_message(task_label, ": row ", i, "/", n_rows, " - ", row_label)

    results[[i]] <- tryCatch(
      purrr::pmap_dfr(row, enrich_fn),
      error = function(e) {
        stop(
          sprintf(
            "%s failed on row %s/%s (%s): %s",
            task_label,
            i,
            n_rows,
            row_label,
            conditionMessage(e)
          ),
          call. = FALSE
        )
      }
    )
  }

  timestamped_message(task_label, ": completed ", n_rows, " rows.")

  dplyr::bind_rows(results)
}

#' Extract the first non-empty identifier
#'
#' @param x Vector or list of candidate IDs.
#'
#' @return The first nonblank ID or `NA_character_`.
extract_first_id <- function(x) {
  values <- x %>%
    unlist(use.names = FALSE) %>%
    as.character()

  values <- values[!is.na(values) & values != "" & values != "NA"]
  if (length(values) == 0) {
    return(NA_character_)
  }
  values[[1]]
}

#' Extract a single scalar character value
#'
#' Flattens list-like objects, drops blank values, and returns the first
#' remaining string.
#'
#' @param x Value to simplify.
#' @param default Value to return when no scalar text can be extracted.
#'
#' @return A single character value or `default`.
extract_scalar_chr <- function(x, default = NA_character_) {
  if (is.null(x) || length(x) == 0) {
    return(default)
  }

  if (is.list(x) && !is.data.frame(x)) {
    x <- unlist(x, recursive = TRUE, use.names = FALSE)
  }

  values <- as.character(x)
  values <- values[!is.na(values) & values != ""]
  if (length(values) == 0) {
    return(default)
  }

  values[[1]]
}

#' Request JSON from an HTTP endpoint
#'
#' Builds a GET request with retry, timeout, and user-agent settings, then
#' parses the response body as JSON.
#'
#' @param url Endpoint URL.
#' @param query Named list of query parameters.
#' @param request_label Human-readable label for logging.
#' @param service_key Optional service identifier used for cooldown tracking.
#' @param cooldown_statuses Integer vector of statuses that should trigger
#'   cooldown.
#'
#' @return Parsed JSON as a list, or `NULL` if the request fails.
request_json <- function(
  url,
  query = list(),
  request_label = url,
  service_key = NULL,
  cooldown_statuses = integer()
) {
  tryCatch(
    {
      response <- httr2::request(url) %>%
        httr2::req_url_query(!!!query) %>%
        httr2::req_user_agent("INFLUX-name-builder/1.0") %>%
        httr2::req_timeout(30) %>%
        build_retry_policy(
          request_label = request_label,
          service_key = service_key,
          cooldown_statuses = cooldown_statuses
        ) %>%
        httr2::req_perform()

      jsonlite::fromJSON(
        httr2::resp_body_string(response),
        simplifyVector = FALSE
      )
    },
    error = function(e) {
      NULL
    }
  )
}

#' Build pathogen-side Wikidata search candidates
#'
#' Combines scientific names, subtype names, and English aliases into a single
#' deduplicated search term vector.
#'
#' @param pathogen_scientific_name_en Pathogen scientific name field.
#' @param pathogen_subtypes_names_en Pathogen subtype names field.
#' @param aka_en English alias field.
#'
#' @return A character vector of candidate search terms.
pathogen_search_candidates <- function(
  pathogen_scientific_name_en,
  pathogen_subtypes_names_en,
  aka_en
) {
  dedupe_terms(c(
    split_terms(pathogen_scientific_name_en, "\n"),
    split_terms(pathogen_subtypes_names_en, "\n"),
    split_terms(aka_en, "\n|,")
  ))
}

#' Extract base pathogen terms
#'
#' Keeps the scientific name and subtype terms without adding alias fields.
#'
#' @param pathogen_scientific_name_en Pathogen scientific name field.
#' @param pathogen_subtypes_names_en Pathogen subtype names field.
#'
#' @return A character vector of deduplicated base pathogen terms.
pathogen_base_terms <- function(
  pathogen_scientific_name_en,
  pathogen_subtypes_names_en
) {
  dedupe_terms(c(
    split_terms(pathogen_scientific_name_en, "\n"),
    split_terms(pathogen_subtypes_names_en, "\n")
  ))
}

#' Extract English pathogen aliases
#'
#' @param aka_en Alias field split on newline or comma.
#'
#' @return A character vector of alias terms.
pathogen_english_alias_terms <- function(aka_en) {
  split_terms(aka_en, "\n|,")
}

#' Extract English disease terms
#'
#' @param disease_name_en Disease name field split on newline or comma.
#'
#' @return A character vector of disease search terms.
disease_english_terms <- function(disease_name_en) {
  split_terms(disease_name_en, "\n|,")
}

taxize_cache <- new.env(parent = emptyenv())
wikidata_search_cache <- new.env(parent = emptyenv())
wikidata_entity_cache <- new.env(parent = emptyenv())

#' Resolve an NCBI taxonomy UID
#'
#' Looks up the first matching NCBI Taxonomy identifier for a scientific name
#' and caches the result.
#'
#' @param scientific_name Scientific name to resolve.
#'
#' @return An NCBI UID as a character string, or `NA_character_`.
fetch_ncbi_uid <- function(scientific_name) {
  key <- paste0("uid::", normalize_text(scientific_name))
  cache_get_or_set(
    taxize_cache,
    key,
    function() {
      if (is_blank_text(scientific_name)) {
        return(NA_character_)
      }

      uid <- tryCatch(
        taxize::get_uid(
          scientific_name,
          ask = FALSE,
          messages = FALSE,
          rows = 1,
          modifier = "Scientific Name"
        ),
        error = function(e) {
          NA_character_
        }
      )

      extract_first_id(uid)
    }
  )
}

#' Fetch English common names via taxize
#'
#' Uses the NCBI UID when available and falls back to scientific-name lookup
#' when it is not.
#'
#' @param scientific_name Scientific name to look up.
#'
#' @return A character vector of unique common names.
fetch_taxize_common_names <- function(scientific_name) {
  key <- paste0("common::", normalize_text(scientific_name))
  cache_get_or_set(
    taxize_cache,
    key,
    function() {
      if (is_blank_text(scientific_name)) {
        return(character())
      }

      pause_for_api()
      uid <- fetch_ncbi_uid(scientific_name)

      common_names <- tryCatch(
        {
          if (!is.na(uid)) {
            taxize::sci2comm(taxize::as.uid(uid, check = FALSE))
          } else {
            taxize::sci2comm(scientific_name, db = "ncbi")
          }
        },
        error = function(e) {
          character()
        }
      )

      dedupe_terms(unname(unlist(common_names, use.names = FALSE)))
    }
  )
}

#' Search Wikidata entities for a query string
#'
#' Calls the `wbsearchentities` API, applies cooldown protection, and returns a
#' tidy hit table.
#'
#' @param query_text Search term.
#' @param language Language code used for search and labels.
#' @param limit Maximum number of hits to request.
#'
#' @return A tibble of Wikidata search hits, or an empty tibble on failure or
#'   cooldown.
search_wikidata_entities <- function(query_text, language = "en", limit = 5) {
  if (service_in_cooldown("wikidata")) {
    return(tibble::tibble())
  }

  key <- paste(language, normalize_text(query_text), sep = "::")
  cache_get_or_set(
    wikidata_search_cache,
    key,
    function() {
      if (is_blank_text(query_text)) {
        return(tibble::tibble())
      }

      pause_for_api()
      payload <- request_json(
        "https://www.wikidata.org/w/api.php",
        query = list(
          action = "wbsearchentities",
          format = "json",
          search = query_text,
          language = language,
          uselang = language,
          type = "item",
          limit = limit
        ),
        request_label = "Wikidata search API",
        service_key = "wikidata",
        cooldown_statuses = 429
      )

      hits <- payload$search %||% list()
      if (length(hits) == 0) {
        return(tibble::tibble())
      }

      tibble::tibble(
        id = purrr::map_chr(hits, ~ extract_scalar_chr(.x$id)),
        label = purrr::map_chr(hits, ~ extract_scalar_chr(.x$label)),
        description = purrr::map_chr(
          hits,
          ~ extract_scalar_chr(.x$description)
        ),
        alias = purrr::map_chr(hits, ~ extract_scalar_chr(.x$alias)),
        match_text = purrr::map_chr(hits, ~ extract_scalar_chr(.x$match$text)),
        match_type = purrr::map_chr(hits, ~ extract_scalar_chr(.x$match$type))
      )
    }
  )
}

#' Score a Wikidata hit for relevance
#'
#' Rewards exact label and alias matches, preferred match types, and
#' descriptions that look like the requested entity kind.
#'
#' @param label Wikidata label.
#' @param description Wikidata description.
#' @param alias Wikidata alias matched by the API.
#' @param match_text Text reported by the API as the match.
#' @param match_type Match type reported by the API.
#' @param query_text Original query term.
#' @param kind Expected entity kind, typically `"pathogen"` or `"disease"`.
#'
#' @return A numeric relevance score where larger values are better.
score_wikidata_hit <- function(
  label,
  description,
  alias,
  match_text,
  match_type,
  query_text,
  kind
) {
  label_key <- normalize_lookup_key(label)
  description_key <- normalize_lookup_key(description)
  alias_key <- normalize_lookup_key(alias)
  match_key <- normalize_lookup_key(match_text)
  query_key <- normalize_lookup_key(query_text)

  kind_bonus <- dplyr::case_when(
    kind == "pathogen" &&
      stringr::str_detect(description_key, pathogen_description_regex) ~ 3,
    kind == "disease" &&
      stringr::str_detect(description_key, disease_description_regex) ~ 3,
    TRUE ~ 0
  )

  exact_label_bonus <- ifelse(label_key == query_key, 10, 0)
  exact_match_bonus <- ifelse(match_key == query_key, 8, 0)
  alias_bonus <- ifelse(alias_key == query_key, 6, 0)
  partial_label_bonus <- ifelse(
    label_key != "" &&
      query_key != "" &&
      (stringr::str_detect(label_key, stringr::fixed(query_key)) ||
        stringr::str_detect(query_key, stringr::fixed(label_key))),
    2,
    0
  )
  match_type_bonus <- ifelse(
    match_type == "label",
    2,
    ifelse(match_type == "alias", 1, 0)
  )

  exact_label_bonus +
    exact_match_bonus +
    alias_bonus +
    partial_label_bonus +
    match_type_bonus +
    kind_bonus
}

#' Resolve the best Wikidata entity ID from candidate terms
#'
#' Searches each candidate term, scores the hits, and returns the best-scoring
#' entity when it clears the minimum score threshold.
#'
#' @param candidate_terms Character vector of candidate search terms.
#' @param kind Expected entity kind, typically `"pathogen"` or `"disease"`.
#'
#' @return A Wikidata entity ID or `NA_character_` when no suitable match is
#'   found.
resolve_wikidata_id <- function(candidate_terms, kind) {
  candidates <- dedupe_terms(candidate_terms)
  if (length(candidates) == 0) {
    return(NA_character_)
  }

  scored_hits <- purrr::map_dfr(
    candidates,
    function(candidate) {
      hits <- search_wikidata_entities(candidate, language = "en", limit = 5)
      if (nrow(hits) == 0) {
        return(tibble::tibble())
      }

      hits %>%
        dplyr::mutate(
          query_text = candidate,
          score = purrr::pmap_dbl(
            list(label, description, alias, match_text, match_type),
            ~ score_wikidata_hit(..1, ..2, ..3, ..4, ..5, candidate, kind)
          )
        )
    }
  )

  if (nrow(scored_hits) == 0) {
    return(NA_character_)
  }

  best_hit <- scored_hits %>%
    dplyr::arrange(dplyr::desc(score), dplyr::desc(match_type == "label")) %>%
    dplyr::slice(1)

  if (best_hit$score[[1]] < 3) {
    return(NA_character_)
  }

  best_hit$id[[1]]
}

#' Fetch and cache a Wikidata entity record
#'
#' Retrieves a full entity payload from `Special:EntityData/<id>.json` unless
#' the Wikidata service is cooling down.
#'
#' @param entity_id Wikidata entity ID.
#'
#' @return A parsed entity list or `NULL`.
fetch_wikidata_entity <- function(entity_id) {
  if (is_blank_text(entity_id)) {
    return(NULL)
  }

  if (service_in_cooldown("wikidata")) {
    return(NULL)
  }

  cache_get_or_set(
    wikidata_entity_cache,
    normalize_text(entity_id),
    function() {
      pause_for_api()
      payload <- request_json(
        sprintf(
          "https://www.wikidata.org/wiki/Special:EntityData/%s.json",
          entity_id
        ),
        request_label = "Wikidata entity API",
        service_key = "wikidata",
        cooldown_statuses = 429
      )

      payload$entities[[entity_id]] %||% NULL
    }
  )
}

#' Extract labels and aliases for one Wikidata language
#'
#' @param entity_id Wikidata entity ID.
#' @param language Target language code.
#'
#' @return A deduplicated character vector of label and alias terms.
extract_wikidata_terms <- function(entity_id, language) {
  entity <- fetch_wikidata_entity(entity_id)
  if (is.null(entity)) {
    return(character())
  }

  label_value <- entity$labels[[language]]$value %||% NA_character_
  alias_values <- entity$aliases[[language]] %||% list()
  alias_terms <- purrr::map_chr(alias_values, ~ .x$value %||% NA_character_)

  dedupe_terms(c(label_value, alias_terms))
}

#' Fetch multilingual terms for all configured languages
#'
#' @param entity_id Wikidata entity ID.
#'
#' @return A named list with one character vector per value in
#'   `wikidata_languages`.
fetch_multilingual_terms <- function(entity_id) {
  if (is_blank_text(entity_id)) {
    return(blank_term_map())
  }

  purrr::set_names(wikidata_languages, wikidata_languages) %>%
    purrr::map(~ extract_wikidata_terms(entity_id, .x))
}

#' Combine pathogen and disease terms for one language
#'
#' Uses language-specific Wikidata terms when available and falls back to the
#' English term sets when target-language terms are missing.
#'
#' @param language Target language code.
#' @param base_pathogen_terms Base pathogen scientific and subtype terms.
#' @param english_pathogen_terms English pathogen terms including aliases and
#'   common names.
#' @param english_disease_terms_value English disease terms.
#' @param pathogen_term_map Multilingual pathogen terms keyed by language.
#' @param disease_term_map Multilingual disease terms keyed by language.
#'
#' @return A deduplicated character vector of search terms for the target
#'   language.
language_specific_terms <- function(
  language,
  base_pathogen_terms,
  english_pathogen_terms,
  english_disease_terms_value,
  pathogen_term_map,
  disease_term_map
) {
  pathogen_local_terms <- dedupe_terms(c(
    pathogen_term_map[[language]],
    pathogen_term_map[["mul"]]
  ))
  disease_local_terms <- dedupe_terms(c(
    disease_term_map[[language]],
    disease_term_map[["mul"]]
  ))

  if (language == "en" || length(pathogen_local_terms) == 0) {
    pathogen_local_terms <- dedupe_terms(c(
      pathogen_local_terms,
      english_pathogen_terms
    ))
  }

  if (language == "en" || length(disease_local_terms) == 0) {
    disease_local_terms <- dedupe_terms(c(
      disease_local_terms,
      english_disease_terms_value
    ))
  }

  dedupe_terms(c(
    base_pathogen_terms,
    pathogen_local_terms,
    disease_local_terms
  ))
}

#' Enrich one pathogen or disease row with multilingual search data
#'
#' Resolves NCBI and Wikidata identifiers, expands multilingual term sets, and
#' builds language-specific Boolean search strings.
#'
#' @param pathogen_scientific_name_en English pathogen scientific name field.
#' @param pathogen_subtypes_names_en English pathogen subtype field.
#' @param aka_en English alias field.
#' @param disease_name_en English disease name field.
#'
#' @return A one-row tibble containing identifiers, multilingual term columns,
#'   and search strings.
enrich_multilingual_name_row <- function(
  pathogen_scientific_name_en,
  pathogen_subtypes_names_en,
  aka_en,
  disease_name_en
) {
  base_pathogen_terms <- pathogen_base_terms(
    pathogen_scientific_name_en = pathogen_scientific_name_en,
    pathogen_subtypes_names_en = pathogen_subtypes_names_en
  )
  pathogen_alias_en <- pathogen_english_alias_terms(aka_en)
  disease_terms_en_value <- disease_english_terms(disease_name_en)

  pathogen_candidates <- pathogen_search_candidates(
    pathogen_scientific_name_en = pathogen_scientific_name_en,
    pathogen_subtypes_names_en = pathogen_subtypes_names_en,
    aka_en = aka_en
  )
  disease_candidates <- disease_english_terms(disease_name_en)

  pathogen_uid_ncbi <- fetch_ncbi_uid(pathogen_scientific_name_en)
  pathogen_common_names_taxize_en <- fetch_taxize_common_names(
    pathogen_scientific_name_en
  )

  pathogen_wikidata_id <- resolve_wikidata_id(
    pathogen_candidates,
    kind = "pathogen"
  )
  disease_wikidata_id <- resolve_wikidata_id(
    disease_candidates,
    kind = "disease"
  )

  pathogen_term_map <- fetch_multilingual_terms(pathogen_wikidata_id)
  disease_term_map <- fetch_multilingual_terms(disease_wikidata_id)

  pathogen_terms_en_value <- dedupe_terms(c(
    base_pathogen_terms,
    pathogen_alias_en,
    pathogen_common_names_taxize_en,
    pathogen_term_map[["en"]],
    pathogen_term_map[["mul"]]
  ))

  disease_terms_en_value <- dedupe_terms(c(
    disease_terms_en_value,
    disease_term_map[["en"]],
    disease_term_map[["mul"]]
  ))

  search_terms_en <- language_specific_terms(
    language = "en",
    base_pathogen_terms = base_pathogen_terms,
    english_pathogen_terms = pathogen_terms_en_value,
    english_disease_terms_value = disease_terms_en_value,
    pathogen_term_map = pathogen_term_map,
    disease_term_map = disease_term_map
  )
  search_terms_es <- language_specific_terms(
    language = "es",
    base_pathogen_terms = base_pathogen_terms,
    english_pathogen_terms = pathogen_terms_en_value,
    english_disease_terms_value = disease_terms_en_value,
    pathogen_term_map = pathogen_term_map,
    disease_term_map = disease_term_map
  )
  search_terms_fr <- language_specific_terms(
    language = "fr",
    base_pathogen_terms = base_pathogen_terms,
    english_pathogen_terms = pathogen_terms_en_value,
    english_disease_terms_value = disease_terms_en_value,
    pathogen_term_map = pathogen_term_map,
    disease_term_map = disease_term_map
  )
  search_terms_pt <- language_specific_terms(
    language = "pt",
    base_pathogen_terms = base_pathogen_terms,
    english_pathogen_terms = pathogen_terms_en_value,
    english_disease_terms_value = disease_terms_en_value,
    pathogen_term_map = pathogen_term_map,
    disease_term_map = disease_term_map
  )

  tibble::tibble(
    pathogen_uid_ncbi = pathogen_uid_ncbi,
    pathogen_common_names_taxize_en = collapse_terms(
      pathogen_common_names_taxize_en
    ),
    pathogen_wikidata_id = pathogen_wikidata_id,
    disease_wikidata_id = disease_wikidata_id,
    pathogen_terms_en = collapse_terms(pathogen_terms_en_value),
    pathogen_terms_es = collapse_terms(dedupe_terms(c(
      base_pathogen_terms,
      pathogen_term_map[["es"]],
      pathogen_term_map[["mul"]]
    ))),
    pathogen_terms_fr = collapse_terms(dedupe_terms(c(
      base_pathogen_terms,
      pathogen_term_map[["fr"]],
      pathogen_term_map[["mul"]]
    ))),
    pathogen_terms_pt = collapse_terms(dedupe_terms(c(
      base_pathogen_terms,
      pathogen_term_map[["pt"]],
      pathogen_term_map[["mul"]]
    ))),
    disease_terms_en = collapse_terms(disease_terms_en_value),
    disease_terms_es = collapse_terms(dedupe_terms(c(
      disease_term_map[["es"]],
      disease_term_map[["mul"]]
    ))),
    disease_terms_fr = collapse_terms(dedupe_terms(c(
      disease_term_map[["fr"]],
      disease_term_map[["mul"]]
    ))),
    disease_terms_pt = collapse_terms(dedupe_terms(c(
      disease_term_map[["pt"]],
      disease_term_map[["mul"]]
    ))),
    search_string = build_search_string(search_terms_en),
    search_string_en = build_search_string(search_terms_en),
    search_string_es = build_search_string(search_terms_es),
    search_string_fr = build_search_string(search_terms_fr),
    search_string_pt = build_search_string(search_terms_pt)
  )
}

#' Read and validate the seed workbook
#'
#' Imports the raw Excel workbook, normalizes column names and cell text, keeps
#' only the seed input section up to `search_string_en`, validates the required
#' fields, and removes any existing derived search or term columns.
#'
#' @param raw_input_path Path to the raw input workbook.
#'
#' @return A normalized tibble containing the name seed columns used for
#'   enrichment.
prepare_name_seed_workbook <- function(raw_input_path) {
  raw_df <- readxl::read_xlsx(raw_input_path, col_types = "text")
  names(raw_df) <- names(raw_df) %>%
    as.character() %>%
    normalize_text()

  if (!"search_string_en" %in% names(raw_df)) {
    stop(sprintf(
      "%s must contain a search_string_en column.",
      raw_input_path
    ))
  }

  end_idx <- match("search_string_en", names(raw_df))
  raw_df <- raw_df[, seq_len(end_idx), drop = FALSE]

  required_columns <- c(
    "pathogen_scientific_name_en",
    "pathogen_subtypes_names_en",
    "aka_en",
    "disease_name_en",
    "search_string_en"
  )
  missing_columns <- setdiff(required_columns, names(raw_df))
  if (length(missing_columns) > 0) {
    stop(sprintf(
      "%s is missing required columns: %s",
      raw_input_path,
      paste(missing_columns, collapse = ", ")
    ))
  }

  raw_df %>%
    dplyr::mutate(dplyr::across(dplyr::everything(), normalize_text)) %>%
    dplyr::select(dplyr::all_of(required_columns)) %>%
    dplyr::select(
      -dplyr::any_of(c(
        "search_string",
        "search_string_en",
        "search_string_es",
        "search_string_fr",
        "search_string_pt",
        "pathogen_uid_ncbi",
        "pathogen_common_names_taxize_en",
        "pathogen_wikidata_id",
        "disease_wikidata_id",
        "pathogen_terms_en",
        "pathogen_terms_es",
        "pathogen_terms_fr",
        "pathogen_terms_pt",
        "disease_terms_en",
        "disease_terms_es",
        "disease_terms_fr",
        "disease_terms_pt"
      ))
    )
}

#' Build and write the multilingual name workbook
#'
#' Runs the full enrichment workflow on the seed workbook, binds the results to
#' the original data, reorders the search string columns, and writes the final
#' `.xlsx` output.
#'
#' @param raw_input_path Path to the raw input workbook.
#' @param output_path Path where the enriched workbook should be written.
#'
#' @return Invisibly returns the enriched output tibble.
build_multilingual_name_list <- function(raw_input_path, output_path) {
  raw_df <- prepare_name_seed_workbook(raw_input_path)

  input_df <- raw_df %>%
    dplyr::select(
      pathogen_scientific_name_en,
      pathogen_subtypes_names_en,
      aka_en,
      disease_name_en
    )

  enriched_df <- run_rowwise_enrichment(
    data = input_df,
    enrich_fn = enrich_multilingual_name_row,
    label_fn = function(row) {
      c(
        row$disease_name_en,
        row$pathogen_scientific_name_en,
        row$pathogen_subtypes_names_en,
        row$aka_en
      )
    },
    task_label = "Multilingual name build"
  )

  output_df <- dplyr::bind_cols(raw_df, enriched_df) %>%
    dplyr::relocate(
      search_string,
      search_string_en,
      search_string_es,
      search_string_fr,
      search_string_pt,
      .after = disease_name_en
    )

  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  writexl::write_xlsx(output_df, output_path)

  invisible(output_df)
}
