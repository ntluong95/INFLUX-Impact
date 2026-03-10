suppressPackageStartupMessages({
  library(httr2)
  library(xml2)
  library(rvest)
  library(jsonlite)
  library(lubridate)
  library(stringr)
  library(dplyr)
  library(purrr)
  library(readr)
  library(glue)
  library(yaml)
  library(digest)
  library(tibble)
  library(tidyr)
})

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0) {
    return(y)
  }
  x
}

get_script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- args[grepl("^--file=", args)]
  if (length(file_arg) == 0) {
    stop("Unable to determine script path from commandArgs().")
  }
  normalizePath(sub("^--file=", "", file_arg[[1]]), winslash = "/", mustWork = TRUE)
}

get_zika_root <- function() {
  normalizePath(file.path(dirname(get_script_path()), ".."), winslash = "/", mustWork = TRUE)
}

ensure_dir <- function(path) {
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
}

timestamp_utc <- function() {
  format(with_tz(Sys.time(), tzone = "UTC"), "%Y-%m-%dT%H:%M:%SZ")
}

normalize_whitespace <- function(x) {
  ifelse(is.na(x), NA_character_, str_squish(as.character(x)))
}

stable_hash <- function(x) {
  digest(as.character(x), algo = "sha256", serialize = FALSE)
}

load_zika_config <- function(zika_root) {
  env_path <- file.path(zika_root, "config", ".env")
  if (file.exists(env_path)) {
    env_lines <- readLines(env_path, warn = FALSE, encoding = "UTF-8")
    env_lines <- env_lines[!grepl("^\\s*(#|$)", env_lines)]
    for (line in env_lines) {
      if (!grepl("=", line, fixed = TRUE)) {
        next
      }
      parts <- strsplit(line, "=", fixed = TRUE)[[1]]
      key <- trimws(parts[[1]])
      value <- trimws(paste(parts[-1], collapse = "="))
      if (!nzchar(key)) {
        next
      }
      if (!nzchar(Sys.getenv(key, unset = ""))) {
        do.call(Sys.setenv, stats::setNames(list(value), key))
      }
    }
  }

  cfg <- yaml::read_yaml(file.path(zika_root, "config", "zika.yaml"))

  cfg$rss$chunk_size_days <- as.integer(Sys.getenv("RSS_CHUNK_SIZE", unset = cfg$rss$chunk_size_days))
  cfg$validation$sample_size <- as.integer(Sys.getenv("VALIDATION_SAMPLE_SIZE", unset = cfg$validation$sample_size))
  cfg$fulltext$rate_limit_per_second <- as.numeric(Sys.getenv("FULLTEXT_RATE_LIMIT", unset = cfg$fulltext$rate_limit_per_second))

  timeout_override <- Sys.getenv("TIMEOUT_SECONDS", unset = "")
  if (nzchar(timeout_override)) {
    cfg$rss$timeout_seconds <- as.integer(timeout_override)
    cfg$headline_filter$openai$timeout_seconds <- as.integer(timeout_override)
    cfg$headline_filter$local$timeout_seconds <- as.integer(timeout_override)
    cfg$fulltext$timeout_seconds <- as.integer(timeout_override)
  }

  cfg$headline_filter$openai$api_key <- Sys.getenv("OPENAI_API_KEY", unset = "")
  cfg$headline_filter$openai$model <- Sys.getenv("OPENAI_MODEL", unset = cfg$headline_filter$openai$model %||% "")
  cfg$headline_filter$openai$base_url <- Sys.getenv("OPENAI_BASE_URL", unset = cfg$headline_filter$openai$base_url %||% "")
  cfg$headline_filter$local$base_url <- Sys.getenv("LOCAL_LLM_BASE_URL", unset = cfg$headline_filter$local$base_url %||% "")
  cfg$headline_filter$local$model <- Sys.getenv("LOCAL_LLM_MODEL", unset = cfg$headline_filter$local$model %||% "")
  cfg$runtime <- list(env_path = env_path, loaded_at = timestamp_utc())
  cfg
}

make_logger <- function(log_path) {
  ensure_dir(dirname(log_path))
  if (!file.exists(log_path)) {
    file.create(log_path)
  }

  write_line <- function(level, message) {
    line <- sprintf("%s | %s | %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S %z"), level, as.character(message))
    cat(line, "\n")
    cat(line, "\n", file = log_path, append = TRUE)
  }

  list(
    info = function(message) write_line("INFO", message),
    warn = function(message) write_line("WARN", message),
    error = function(message) write_line("ERROR", message),
    path = log_path
  )
}

timed_step <- function(logger, label, expr) {
  start_time <- Sys.time()
  logger$info(glue("{label} started"))
  result <- force(expr)
  elapsed <- round(as.numeric(difftime(Sys.time(), start_time, units = "secs")), 2)
  logger$info(glue("{label} finished in {elapsed}s"))
  result
}

build_date_windows <- function(start_date, end_date, chunk_size_days) {
  chunk_size_days <- max(1L, as.integer(chunk_size_days))
  start_date <- as.Date(start_date)
  end_date <- as.Date(end_date)

  starts <- seq(start_date, end_date, by = sprintf("%d days", chunk_size_days))
  tibble(
    date_window_start = starts,
    date_window_end = pmin(starts + days(chunk_size_days - 1L), end_date),
    query_before_date = date_window_end + days(1L)
  ) %>%
    mutate(
      window_id = glue("{date_window_start}_{date_window_end}")
    )
}

build_google_rss_url <- function(query_term, window_start, query_before_date, rss_cfg) {
  q_value <- glue(
    "{query_term} after:{window_start} before:{query_before_date}"
  )
  base_url <- rss_cfg$base_url %||% "https://news.google.com/rss/search"
  paste0(
    base_url,
    "?q=",
    utils::URLencode(q_value, reserved = TRUE),
    "&hl=",
    utils::URLencode(rss_cfg$hl %||% "en-US", reserved = TRUE),
    "&gl=",
    utils::URLencode(rss_cfg$gl %||% "US", reserved = TRUE),
    "&ceid=",
    utils::URLencode(rss_cfg$ceid %||% "US:en", reserved = TRUE)
  )
}

fetch_text_with_retry <- function(url, timeout_seconds, user_agent, max_retries, retry_backoff_seconds) {
  last_error <- NULL
  for (attempt in seq_len(max_retries)) {
    response <- tryCatch(
      {
        req <- request(url) |>
          req_user_agent(user_agent) |>
          req_timeout(timeout_seconds)
        req_perform(req)
      },
      error = function(e) e
    )

    if (!inherits(response, "error")) {
      return(list(ok = TRUE, text = resp_body_string(response), status = resp_status(response), error = NULL))
    }

    last_error <- conditionMessage(response)
    if (attempt < max_retries) {
      Sys.sleep(retry_backoff_seconds * (2 ^ (attempt - 1)))
    }
  }

  list(ok = FALSE, text = "", status = NA_integer_, error = last_error %||% "Unknown request error")
}

parse_description_text <- function(description_html) {
  if (is.na(description_html) || !nzchar(description_html)) {
    return("")
  }
  tryCatch(
    {
      xml2::read_html(paste0("<div>", description_html, "</div>")) |>
        rvest::html_text2()
    },
    error = function(e) ""
  ) |>
    normalize_whitespace()
}

parse_rss_datetime <- function(x) {
  parsed <- suppressWarnings(parse_date_time(
    x,
    orders = c("a, d b Y H:M:S z", "d b Y H:M:S z"),
    tz = "UTC"
  ))
  ifelse(is.na(parsed), NA_character_, format(with_tz(parsed, "UTC"), "%Y-%m-%dT%H:%M:%SZ"))
}

empty_rss_tbl <- function() {
  tibble(
    record_id = character(),
    query = character(),
    date_window_start = character(),
    date_window_end = character(),
    language = character(),
    rss_title = character(),
    rss_pubdate = character(),
    rss_pubdate_utc = character(),
    rss_description = character(),
    rss_description_text = character(),
    google_rss_url = character(),
    google_news_redirect_url = character(),
    guid = character(),
    source_hint = character(),
    source_url = character(),
    feed_channel_title = character(),
    feed_last_build_date = character(),
    retrieved_at = character(),
    raw_xml_path = character()
  )
}

parse_rss_feed_file <- function(xml_path, window_row, query_term, language_label) {
  doc <- tryCatch(xml2::read_xml(xml_path), error = function(e) NULL)
  if (is.null(doc)) {
    return(empty_rss_tbl())
  }

  channel <- xml2::xml_find_first(doc, "//channel")
  feed_channel_title <- xml2::xml_text(xml2::xml_find_first(channel, "./title"))
  feed_last_build_date <- xml2::xml_text(xml2::xml_find_first(channel, "./lastBuildDate"))
  feed_language <- xml2::xml_text(xml2::xml_find_first(channel, "./language"))
  items <- xml2::xml_find_all(channel, "./item")

  if (length(items) == 0) {
    return(empty_rss_tbl())
  }

  retrieved_at <- format(with_tz(file.info(xml_path)$mtime, "UTC"), "%Y-%m-%dT%H:%M:%SZ")

  map_dfr(items, function(item) {
    rss_title <- xml2::xml_text(xml2::xml_find_first(item, "./title"))
    rss_pubdate <- xml2::xml_text(xml2::xml_find_first(item, "./pubDate"))
    google_news_redirect_url <- xml2::xml_text(xml2::xml_find_first(item, "./link"))
    guid <- xml2::xml_text(xml2::xml_find_first(item, "./guid"))
    description <- xml2::xml_text(xml2::xml_find_first(item, "./description"))
    source_node <- xml2::xml_find_first(item, "./source")
    source_hint <- xml2::xml_text(source_node)
    source_url <- xml2::xml_attr(source_node, "url")

    tibble(
      query = query_term,
      date_window_start = as.character(window_row$date_window_start),
      date_window_end = as.character(window_row$date_window_end),
      language = coalesce(feed_language, language_label),
      rss_title = normalize_whitespace(rss_title),
      rss_pubdate = normalize_whitespace(rss_pubdate),
      rss_pubdate_utc = parse_rss_datetime(rss_pubdate),
      rss_description = description %||% "",
      rss_description_text = parse_description_text(description),
      google_rss_url = window_row$google_rss_url,
      google_news_redirect_url = normalize_whitespace(google_news_redirect_url),
      guid = normalize_whitespace(guid),
      source_hint = normalize_whitespace(source_hint %||% ""),
      source_url = normalize_whitespace(source_url %||% ""),
      feed_channel_title = normalize_whitespace(feed_channel_title),
      feed_last_build_date = normalize_whitespace(feed_last_build_date),
      retrieved_at = retrieved_at,
      raw_xml_path = normalizePath(xml_path, winslash = "/", mustWork = FALSE)
    )
  }) %>%
    mutate(
      record_id = stable_hash(paste(
        query,
        date_window_start,
        date_window_end,
        google_news_redirect_url,
        guid,
        rss_title,
        rss_pubdate_utc,
        sep = "||"
      ))
    ) %>%
    select(record_id, everything())
}

normalize_title_for_dedupe <- function(x) {
  x %>%
    str_to_lower() %>%
    str_replace_all("[^[:alnum:] ]+", " ") %>%
    str_squish()
}

dedupe_rss_records <- function(df) {
  if (nrow(df) == 0) {
    return(list(
      data = df,
      counts = tibble(
        metric = c("input_rows", "duplicate_redirect_url", "duplicate_guid", "duplicate_title_pubdate", "kept_rows"),
        value = c(0, 0, 0, 0, 0)
      )
    ))
  }

  prepared <- df %>%
    mutate(
      dedupe_redirect_key = na_if(google_news_redirect_url, ""),
      dedupe_guid_key = na_if(guid, ""),
      dedupe_title_pubdate_key = if_else(
        !is.na(rss_pubdate_utc) & nzchar(rss_pubdate_utc),
        paste(normalize_title_for_dedupe(rss_title), rss_pubdate_utc, sep = "||"),
        NA_character_
      )
    ) %>%
    arrange(date_window_start, date_window_end, retrieved_at, google_news_redirect_url, guid, rss_title)

  duplicate_redirect <- !is.na(prepared$dedupe_redirect_key) & duplicated(prepared$dedupe_redirect_key)
  remaining_after_redirect <- prepared[!duplicate_redirect, , drop = FALSE]
  duplicate_guid_remaining <- !is.na(remaining_after_redirect$dedupe_guid_key) & duplicated(remaining_after_redirect$dedupe_guid_key)
  duplicate_guid <- rep(FALSE, nrow(prepared))
  duplicate_guid[which(!duplicate_redirect)] <- duplicate_guid_remaining

  remaining_after_guid <- prepared[!(duplicate_redirect | duplicate_guid), , drop = FALSE]
  duplicate_title_pubdate_remaining <- !is.na(remaining_after_guid$dedupe_title_pubdate_key) & duplicated(remaining_after_guid$dedupe_title_pubdate_key)
  duplicate_title_pubdate <- rep(FALSE, nrow(prepared))
  duplicate_title_pubdate[which(!(duplicate_redirect | duplicate_guid))] <- duplicate_title_pubdate_remaining

  kept <- prepared[!(duplicate_redirect | duplicate_guid | duplicate_title_pubdate), , drop = FALSE] %>%
    select(-starts_with("dedupe_"))

  counts <- tibble(
    metric = c("input_rows", "duplicate_redirect_url", "duplicate_guid", "duplicate_title_pubdate", "kept_rows"),
    value = c(
      nrow(prepared),
      sum(duplicate_redirect),
      sum(duplicate_guid),
      sum(duplicate_title_pubdate),
      nrow(kept)
    )
  )

  list(data = kept, counts = counts)
}

write_csv_and_parquet <- function(df, csv_path, parquet_path) {
  ensure_dir(dirname(csv_path))
  readr::write_csv(df, csv_path, na = "")

  if (requireNamespace("arrow", quietly = TRUE)) {
    arrow::write_parquet(df, parquet_path)
    return(invisible(parquet_path))
  }

  script_path <- tempfile(fileext = ".py")
  writeLines(
    c(
      "import sys",
      "import pandas as pd",
      "df = pd.read_csv(sys.argv[1])",
      "df.to_parquet(sys.argv[2], index=False)"
    ),
    script_path
  )
  cmd <- paste(
    "uv run python",
    shQuote(script_path),
    shQuote(csv_path),
    shQuote(parquet_path)
  )
  result <- tryCatch(
    system2("bash", args = c("-lc", shQuote(cmd)), stdout = TRUE, stderr = TRUE),
    warning = function(w) character(),
    error = function(e) stop(glue("Failed to write parquet via uv: {conditionMessage(e)}"))
  )
  unlink(script_path)

  if (!file.exists(parquet_path)) {
    stop(glue("Parquet file was not created: {parquet_path}\n{paste(result, collapse = '\n')}"))
  }

  invisible(parquet_path)
}

extract_json_object <- function(raw_text) {
  cleaned <- raw_text %>%
    str_replace_all("```json|```", "") %>%
    str_trim()

  json_candidate <- str_extract(cleaned, "(?s)\\{.*\\}")
  if (is.na(json_candidate) || !nzchar(json_candidate)) {
    return(NULL)
  }

  tryCatch(jsonlite::fromJSON(json_candidate, simplifyVector = TRUE), error = function(e) NULL)
}

parse_llm_label <- function(raw_text, confidence_default = 0.5) {
  parsed <- extract_json_object(raw_text)
  if (is.null(parsed)) {
    return(list(
      label = "unsure",
      confidence = as.numeric(confidence_default),
      rationale = "",
      parse_error = "invalid_json"
    ))
  }

  label <- tolower(parsed$label %||% "unsure")
  if (!label %in% c("relevant", "irrelevant", "unsure")) {
    label <- "unsure"
    parse_error <- "invalid_label"
  } else {
    parse_error <- ""
  }

  confidence <- suppressWarnings(as.numeric(parsed$confidence %||% confidence_default))
  if (is.na(confidence)) {
    confidence <- as.numeric(confidence_default)
  }
  confidence <- min(1, max(0, confidence))

  rationale <- normalize_whitespace(substr(parsed$rationale %||% "", 1, 240))

  list(
    label = label,
    confidence = confidence,
    rationale = rationale,
    parse_error = parse_error
  )
}

build_headline_prompt <- function(title, source_hint, pubdate) {
  glue(
"Classify this Google News headline for relevance to Zika.

Return JSON only with keys:
- label: relevant | irrelevant | unsure
- confidence: number from 0 to 1
- rationale: short rationale under 20 words

Rules:
- relevant: clearly about Zika virus, Zika disease, ZIKV, outbreaks, cases, spread, risk, policy, or health impacts related to Zika
- irrelevant: Zika is incidental, metaphorical, spammy, ambiguous, or not about the pathogen or disease
- unsure: not enough information from the headline alone

Headline: {title %||% '[missing]'}
Source: {source_hint %||% '[missing]'}
Published: {pubdate %||% '[missing]'}"
  )
}

openai_endpoint <- function(base_url, path) {
  if (is.null(base_url) || !nzchar(base_url)) {
    return(paste0("https://api.openai.com/v1", path))
  }

  trimmed <- sub("/+$", "", base_url)
  if (grepl("/v1$", trimmed)) {
    paste0(trimmed, path)
  } else {
    paste0(trimmed, "/v1", path)
  }
}

call_openai_chat <- function(system_prompt, user_prompt, model_cfg) {
  api_key <- model_cfg$api_key %||% ""
  if (!nzchar(api_key)) {
    stop("Missing OPENAI_API_KEY. Set it in zika/config/.env or the shell environment.")
  }

  body <- list(
    model = model_cfg$model,
    temperature = model_cfg$temperature %||% 0,
    max_tokens = model_cfg$max_tokens %||% 120,
    response_format = list(type = "json_object"),
    messages = list(
      list(role = "system", content = system_prompt),
      list(role = "user", content = user_prompt)
    )
  )

  endpoint <- openai_endpoint(model_cfg$base_url %||% "", "/chat/completions")
  resp <- request(endpoint) |>
    req_headers(
      Authorization = paste("Bearer", api_key),
      `Content-Type` = "application/json"
    ) |>
    req_timeout(as.integer(model_cfg$timeout_seconds %||% 60)) |>
    req_body_json(body, auto_unbox = TRUE) |>
    req_perform()

  raw <- resp_body_string(resp)
  payload <- resp_body_json(resp, simplifyVector = FALSE)
  content <- payload$choices[[1]]$message$content %||% ""
  list(raw_http = raw, content = content)
}

call_local_llm <- function(system_prompt, user_prompt, model_cfg) {
  base_url <- sub("/+$", "", model_cfg$base_url %||% "")
  if (!nzchar(base_url)) {
    stop("Missing LOCAL_LLM_BASE_URL. Set it in zika/config/.env or the shell environment.")
  }

  provider <- tolower(model_cfg$provider %||% "ollama")
  if (provider != "ollama") {
    stop(glue("Unsupported local provider in this MVP: {provider}"))
  }

  body <- list(
    model = model_cfg$model,
    system = system_prompt,
    prompt = user_prompt,
    stream = FALSE,
    format = "json",
    options = list(
      temperature = model_cfg$temperature %||% 0,
      num_predict = model_cfg$max_tokens %||% 120
    )
  )

  resp <- request(paste0(base_url, "/api/generate")) |>
    req_timeout(as.integer(model_cfg$timeout_seconds %||% 120)) |>
    req_body_json(body, auto_unbox = TRUE) |>
    req_perform()

  raw <- resp_body_string(resp)
  payload <- resp_body_json(resp, simplifyVector = FALSE)
  content <- payload$response %||% ""
  list(raw_http = raw, content = content)
}

get_openai_embedding <- function(text, model_cfg, cache_env) {
  key <- stable_hash(text)
  if (exists(key, envir = cache_env, inherits = FALSE)) {
    return(get(key, envir = cache_env, inherits = FALSE))
  }

  endpoint <- openai_endpoint(model_cfg$base_url %||% "", "/embeddings")
  resp <- request(endpoint) |>
    req_headers(
      Authorization = paste("Bearer", model_cfg$api_key),
      `Content-Type` = "application/json"
    ) |>
    req_timeout(as.integer(model_cfg$timeout_seconds %||% 60)) |>
    req_body_json(
      list(
        model = model_cfg$embedding_model,
        input = text
      ),
      auto_unbox = TRUE
    ) |>
    req_perform()

  payload <- resp_body_json(resp, simplifyVector = FALSE)
  vector <- unlist(payload$data[[1]]$embedding)
  assign(key, vector, envir = cache_env)
  vector
}

cosine_similarity <- function(a, b) {
  denom <- sqrt(sum(a * a)) * sqrt(sum(b * b))
  if (identical(denom, 0) || is.na(denom) || denom == 0) {
    return(NA_real_)
  }
  as.numeric(sum(a * b) / denom)
}

compute_rationale_similarity <- function(rationale_a, rationale_b, model_cfg, cache_env) {
  if (!nzchar(rationale_a) || !nzchar(rationale_b)) {
    return(list(score = NA_real_, method = "missing_rationale"))
  }

  emb_a <- get_openai_embedding(rationale_a, model_cfg = model_cfg, cache_env = cache_env)
  emb_b <- get_openai_embedding(rationale_b, model_cfg = model_cfg, cache_env = cache_env)
  list(score = cosine_similarity(emb_a, emb_b), method = "openai_embedding_cosine")
}

derive_ensemble_action <- function(openai_label, local_label, rationale_similarity, similarity_threshold) {
  if (identical(openai_label, "irrelevant") && identical(local_label, "irrelevant")) {
    return("drop")
  }

  if (
    identical(openai_label, "relevant") &&
      identical(local_label, "relevant") &&
      !is.na(rationale_similarity) &&
      rationale_similarity >= similarity_threshold
  ) {
    return("keep")
  }

  "review"
}

ensemble_label_from_action <- function(action) {
  dplyr::case_when(
    action == "keep" ~ "relevant",
    action == "drop" ~ "irrelevant",
    TRUE ~ "unsure"
  )
}

deterministic_validation_sample <- function(scored_df, sample_size, seed = 42L) {
  if (nrow(scored_df) == 0) {
    return(scored_df[0, , drop = FALSE])
  }

  candidate_df <- scored_df %>%
    mutate(
      agreement_bucket = if_else(openai_label == local_label, "agree", "disagree"),
      sample_stratum = paste(final_action, agreement_bucket, sep = "__"),
      sample_order_key = map_chr(record_id, ~ digest(.x, algo = "xxhash64", serialize = FALSE))
    )

  total_n <- nrow(candidate_df)
  sample_size <- min(sample_size, total_n)
  if (sample_size <= 0) {
    return(candidate_df[0, , drop = FALSE])
  }

  strata_counts <- candidate_df %>%
    count(sample_stratum, name = "available_n") %>%
    mutate(
      raw_target = sample_size * available_n / sum(available_n),
      target_n = floor(raw_target),
      fractional = raw_target - target_n
    )

  remaining <- sample_size - sum(strata_counts$target_n)
  if (remaining > 0) {
    strata_counts <- strata_counts %>%
      arrange(desc(fractional), sample_stratum) %>%
      mutate(target_n = target_n + if_else(row_number() <= remaining, 1L, 0L)) %>%
      arrange(sample_stratum)
  }

  set.seed(seed)
  candidate_with_targets <- candidate_df %>%
    left_join(strata_counts %>% select(sample_stratum, target_n), by = "sample_stratum")

  sampled <- purrr::map_dfr(split(candidate_with_targets, candidate_with_targets$sample_stratum), function(group_df) {
    target_n <- unique(group_df$target_n)[1]
    group_df %>%
      arrange(sample_order_key) %>%
      slice_head(n = target_n)
  })

  sampled %>%
    select(-target_n) %>%
    arrange(sample_stratum, sample_order_key)
}

metric_row <- function(group, metric, value, subgroup = "") {
  tibble(metric_group = group, metric_name = metric, subgroup = subgroup, value = value)
}

classification_metrics <- function(truth, pred, labels = c("relevant", "irrelevant", "unsure")) {
  truth <- as.character(truth)
  pred <- as.character(pred)
  valid <- truth %in% labels & pred %in% labels
  truth <- truth[valid]
  pred <- pred[valid]

  if (length(truth) == 0) {
    return(list(n = 0, accuracy = NA_real_, macro_f1 = NA_real_))
  }

  accuracy <- mean(truth == pred)

  per_label_f1 <- map_dbl(labels, function(label) {
    tp <- sum(truth == label & pred == label)
    fp <- sum(truth != label & pred == label)
    fn <- sum(truth == label & pred != label)
    precision <- if ((tp + fp) == 0) NA_real_ else tp / (tp + fp)
    recall <- if ((tp + fn) == 0) NA_real_ else tp / (tp + fn)
    if (is.na(precision) || is.na(recall) || (precision + recall) == 0) {
      return(NA_real_)
    }
    2 * precision * recall / (precision + recall)
  })

  list(
    n = length(truth),
    accuracy = accuracy,
    macro_f1 = mean(per_label_f1, na.rm = TRUE)
  )
}
