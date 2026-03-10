args_all <- commandArgs(trailingOnly = FALSE)
script_path <- normalizePath(
  sub("^--file=", "", args_all[grepl("^--file=", args_all)][[1]]),
  winslash = "/",
  mustWork = TRUE
)
source(file.path(dirname(script_path), "utils.R"))

zika_root <- get_zika_root()
cfg <- load_zika_config(zika_root)

log_path <- file.path(zika_root, "logs", "01_retrieve_google_rss.log")
logger <- make_logger(log_path)

raw_rss_dir <- file.path(zika_root, "data", "raw", "rss")
intermediate_dir <- file.path(zika_root, "data", "intermediate")
ensure_dir(raw_rss_dir)
ensure_dir(intermediate_dir)

output_csv <- file.path(intermediate_dir, "zika_rss_raw.csv")
output_parquet <- file.path(intermediate_dir, "zika_rss_raw.parquet")

query_term <- cfg$query$term
language_label <- cfg$query$language
windows <- build_date_windows(
  start_date = cfg$query$start_date,
  end_date = cfg$query$end_date,
  chunk_size_days = cfg$rss$chunk_size_days
) %>%
  mutate(
    google_rss_url = pmap_chr(
      list(date_window_start, query_before_date),
      function(date_window_start, query_before_date) {
        build_google_rss_url(
          query_term = query_term,
          window_start = as.character(date_window_start),
          query_before_date = as.character(query_before_date),
          rss_cfg = cfg$rss
        )
      }
    ),
    raw_xml_path = file.path(raw_rss_dir, glue("{query_term}_{window_id}.xml"))
  )

logger$info(glue("Stage 1 configured for {nrow(windows)} windows from {cfg$query$start_date} to {cfg$query$end_date}"))

rss_records <- timed_step(logger, "RSS retrieval and parsing", {
  all_rows <- vector("list", length = nrow(windows))

  for (i in seq_len(nrow(windows))) {
    window_row <- windows[i, ]
    xml_path <- window_row$raw_xml_path[[1]]
    google_rss_url <- window_row$google_rss_url[[1]]
    fetch_mode <- "cached"

    if (!file.exists(xml_path) || isTRUE(cfg$rss$force_refresh)) {
      fetch_mode <- "fetched"
      response <- fetch_text_with_retry(
        url = google_rss_url,
        timeout_seconds = as.integer(cfg$rss$timeout_seconds),
        user_agent = cfg$rss$user_agent,
        max_retries = as.integer(cfg$rss$max_retries),
        retry_backoff_seconds = as.numeric(cfg$rss$retry_backoff_seconds)
      )

      if (!isTRUE(response$ok)) {
        logger$warn(glue("Failed {window_row$window_id}: {response$error}"))
        next
      }

      writeLines(response$text, xml_path, useBytes = TRUE)
      Sys.sleep(as.numeric(cfg$rss$request_sleep_seconds %||% 0))
    }

    parsed_rows <- parse_rss_feed_file(
      xml_path = xml_path,
      window_row = window_row,
      query_term = query_term,
      language_label = language_label
    )
    all_rows[[i]] <- parsed_rows

    logger$info(glue(
      "Window {i}/{nrow(windows)} {window_row$date_window_start}..{window_row$date_window_end} mode={fetch_mode} rows={nrow(parsed_rows)}"
    ))
  }

  bind_rows(all_rows)
})

dedupe_result <- timed_step(logger, "Deterministic deduplication", {
  dedupe_rss_records(rss_records)
})

rss_deduped <- dedupe_result$data
write_csv_and_parquet(rss_deduped, output_csv, output_parquet)

for (i in seq_len(nrow(dedupe_result$counts))) {
  logger$info(glue("Metric {dedupe_result$counts$metric[[i]]}={dedupe_result$counts$value[[i]]}"))
}

logger$info(glue("Wrote {nrow(rss_deduped)} deduplicated RSS rows to {output_csv} and {output_parquet}"))
