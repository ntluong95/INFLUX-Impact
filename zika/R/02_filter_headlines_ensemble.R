args_all <- commandArgs(trailingOnly = FALSE)
script_path <- normalizePath(
  sub("^--file=", "", args_all[grepl("^--file=", args_all)][[1]]),
  winslash = "/",
  mustWork = TRUE
)
source(file.path(dirname(script_path), "utils.R"))

zika_root <- get_zika_root()
cfg <- load_zika_config(zika_root)

log_path <- file.path(zika_root, "logs", "02_filter_headlines_ensemble.log")
logger <- make_logger(log_path)

intermediate_dir <- file.path(zika_root, "data", "intermediate")
validation_dir <- file.path(zika_root, "data", "validation")
ensure_dir(intermediate_dir)
ensure_dir(validation_dir)

input_csv <- file.path(intermediate_dir, "zika_rss_raw.csv")
output_csv <- file.path(intermediate_dir, "zika_headlines_scored.csv")
output_parquet <- file.path(intermediate_dir, "zika_headlines_scored.parquet")
validation_csv <- file.path(
  validation_dir,
  "zika_headline_validation_sample.csv"
)
metrics_json <- file.path(validation_dir, "zika_headline_metrics.json")
metrics_csv <- file.path(validation_dir, "zika_headline_metrics.csv")

if (!file.exists(input_csv)) {
  stop(glue("Missing input CSV: {input_csv}. Run stage 1 first."))
}

scored_base <- readr::read_csv(input_csv, show_col_types = FALSE) %>%
  mutate(
    openai_model = cfg$headline_filter$openai$model,
    openai_raw_response = NA_character_,
    openai_label = NA_character_,
    openai_confidence = NA_real_,
    openai_rationale = NA_character_,
    openai_error = NA_character_,
    local_model = cfg$headline_filter$local$model,
    local_raw_response = NA_character_,
    local_label = NA_character_,
    local_confidence = NA_real_,
    local_rationale = NA_character_,
    local_error = NA_character_,
    rationale_similarity = NA_real_,
    rationale_similarity_method = NA_character_,
    final_action = NA_character_,
    ensemble_label = NA_character_,
    processed_at = NA_character_
  )

if (file.exists(output_csv)) {
  existing <- readr::read_csv(output_csv, show_col_types = FALSE)
  extra_cols <- setdiff(names(existing), names(scored_base))
  if (length(extra_cols) > 0) {
    scored_base <- scored_base %>%
      left_join(
        existing %>% select(record_id, all_of(extra_cols)),
        by = "record_id"
      )
  } else {
    scored_base <- scored_base %>%
      select(names(scored_base)) %>%
      left_join(
        existing %>%
          select(record_id, setdiff(names(existing), names(scored_base))),
        by = "record_id"
      )
  }

  overlap_cols <- intersect(
    setdiff(
      names(existing),
      names(readr::read_csv(input_csv, show_col_types = FALSE))
    ),
    names(scored_base)
  )
  for (col_name in overlap_cols) {
    scored_base[[col_name]] <- existing[[col_name]][match(
      scored_base$record_id,
      existing$record_id
    )]
  }
}

incomplete_idx <- which(
  is.na(scored_base$final_action) | scored_base$final_action == ""
)
logger$info(glue(
  "Stage 2 starting with {nrow(scored_base)} headlines; {length(incomplete_idx)} rows need scoring"
))

if (
  length(incomplete_idx) > 0 &&
    !nzchar(cfg$headline_filter$openai$api_key %||% "")
) {
  stop(
    "Missing OPENAI_API_KEY. Stage 2 requires OpenAI + local LLM scoring for incomplete rows."
  )
}

if (
  length(incomplete_idx) > 0 &&
    (!nzchar(cfg$headline_filter$local$base_url %||% "") ||
      !nzchar(cfg$headline_filter$local$model %||% ""))
) {
  stop(
    "Missing LOCAL_LLM_BASE_URL or LOCAL_LLM_MODEL. Stage 2 requires a configured local LLM for incomplete rows."
  )
}

system_prompt <- paste(
  "You are a strict headline classifier for Zika relevance.",
  'Return JSON only with keys "label", "confidence", and "rationale".',
  'Use label values only: "relevant", "irrelevant", "unsure".',
  "Confidence must be between 0 and 1.",
  "Rationale must be under 20 words.",
  sep = " "
)

embedding_cache <- new.env(parent = emptyenv())

score_single_row <- function(row) {
  prompt <- build_headline_prompt(
    title = row$rss_title,
    source_hint = row$source_hint,
    pubdate = row$rss_pubdate
  )

  openai_result <- tryCatch(
    call_openai_chat(system_prompt, prompt, cfg$headline_filter$openai),
    error = function(e) {
      list(raw_http = "", content = "", error = conditionMessage(e))
    }
  )
  openai_parsed <- parse_llm_label(
    openai_result$content %||% "",
    cfg$headline_filter$confidence_default
  )
  openai_error <- openai_result$error %||% openai_parsed$parse_error

  local_result <- tryCatch(
    call_local_llm(system_prompt, prompt, cfg$headline_filter$local),
    error = function(e) {
      list(raw_http = "", content = "", error = conditionMessage(e))
    }
  )
  local_parsed <- parse_llm_label(
    local_result$content %||% "",
    cfg$headline_filter$confidence_default
  )
  local_error <- local_result$error %||% local_parsed$parse_error

  rationale_similarity <- NA_real_
  rationale_similarity_method <- "not_applicable"

  if (
    identical(openai_parsed$label, "relevant") &&
      identical(local_parsed$label, "relevant")
  ) {
    sim_result <- tryCatch(
      compute_rationale_similarity(
        rationale_a = openai_parsed$rationale,
        rationale_b = local_parsed$rationale,
        model_cfg = cfg$headline_filter$openai,
        cache_env = embedding_cache
      ),
      error = function(e) {
        list(
          score = NA_real_,
          method = paste0("embedding_error:", conditionMessage(e))
        )
      }
    )
    rationale_similarity <- sim_result$score
    rationale_similarity_method <- sim_result$method
  }

  final_action <- derive_ensemble_action(
    openai_label = openai_parsed$label,
    local_label = local_parsed$label,
    rationale_similarity = rationale_similarity,
    similarity_threshold = as.numeric(cfg$headline_filter$similarity_threshold)
  )

  list(
    openai_raw_response = openai_result$raw_http %||%
      openai_result$content %||%
      "",
    openai_label = openai_parsed$label,
    openai_confidence = openai_parsed$confidence,
    openai_rationale = openai_parsed$rationale,
    openai_error = openai_error %||% "",
    local_raw_response = local_result$raw_http %||%
      local_result$content %||%
      "",
    local_label = local_parsed$label,
    local_confidence = local_parsed$confidence,
    local_rationale = local_parsed$rationale,
    local_error = local_error %||% "",
    rationale_similarity = rationale_similarity,
    rationale_similarity_method = rationale_similarity_method,
    final_action = final_action,
    ensemble_label = ensemble_label_from_action(final_action),
    processed_at = timestamp_utc()
  )
}

timed_step(logger, "Headline scoring", {
  if (length(incomplete_idx) == 0) {
    logger$info(
      "No incomplete rows detected; refreshing diagnostics and validation assets only."
    )
    invisible(NULL)
  }

  checkpoint_every <- max(
    1L,
    as.integer(cfg$headline_filter$checkpoint_every %||% 10L)
  )

  for (offset in seq_along(incomplete_idx)) {
    idx <- incomplete_idx[[offset]]
    row <- scored_base[idx, , drop = FALSE]

    scored <- score_single_row(row)
    for (name in names(scored)) {
      scored_base[[name]][idx] <- scored[[name]]
    }

    logger$info(glue(
      "Scored {offset}/{length(incomplete_idx)} record_id={row$record_id[[1]]} openai={scored$openai_label} local={scored$local_label} action={scored$final_action}"
    ))

    if ((offset %% checkpoint_every) == 0 || offset == length(incomplete_idx)) {
      write_csv_and_parquet(scored_base, output_csv, output_parquet)
      logger$info(glue("Checkpoint written after {offset} newly scored rows"))
    }
  }
})

if (!file.exists(output_csv)) {
  write_csv_and_parquet(scored_base, output_csv, output_parquet)
}

scored_df <- readr::read_csv(output_csv, show_col_types = FALSE)

metrics_rows <- bind_rows(
  metric_row("counts", "total_rows", nrow(scored_df)),
  metric_row(
    "agreement",
    "raw_agreement_rate",
    mean(scored_df$openai_label == scored_df$local_label, na.rm = TRUE)
  ),
  metric_row(
    "agreement",
    "disagreement_rate",
    mean(scored_df$openai_label != scored_df$local_label, na.rm = TRUE)
  ),
  metric_row(
    "agreement",
    "share_any_unsure",
    mean(
      scored_df$openai_label == "unsure" | scored_df$local_label == "unsure",
      na.rm = TRUE
    )
  )
) %>%
  bind_rows(
    scored_df %>%
      count(openai_label, name = "value") %>%
      transmute(
        metric_group = "openai_distribution",
        metric_name = "label_share",
        subgroup = openai_label,
        value = value / sum(value)
      ),
    scored_df %>%
      count(local_label, name = "value") %>%
      transmute(
        metric_group = "local_distribution",
        metric_name = "label_share",
        subgroup = local_label,
        value = value / sum(value)
      ),
    scored_df %>%
      count(final_action, name = "value") %>%
      transmute(
        metric_group = "ensemble_distribution",
        metric_name = "action_share",
        subgroup = final_action,
        value = value / sum(value)
      )
  )

existing_validation <- if (file.exists(validation_csv)) {
  readr::read_csv(validation_csv, show_col_types = FALSE)
} else {
  tibble(record_id = character())
}

for (column_name in c("gold_label", "review_notes", "validated_at")) {
  if (!column_name %in% names(existing_validation)) {
    existing_validation[[column_name]] <- NA_character_
  }
}

validation_sample <- deterministic_validation_sample(
  scored_df = scored_df,
  sample_size = as.integer(cfg$validation$sample_size),
  seed = as.integer(cfg$validation$seed %||% 42L)
) %>%
  select(
    record_id,
    rss_title,
    source_hint,
    rss_pubdate,
    google_news_redirect_url,
    openai_label,
    openai_confidence,
    openai_rationale,
    local_label,
    local_confidence,
    local_rationale,
    rationale_similarity,
    final_action,
    ensemble_label
  ) %>%
  left_join(
    existing_validation %>%
      select(record_id, gold_label, review_notes, validated_at),
    by = "record_id"
  ) %>%
  mutate(
    gold_label = coalesce(gold_label, NA_character_),
    review_notes = coalesce(review_notes, NA_character_),
    validated_at = coalesce(validated_at, NA_character_)
  )

readr::write_csv(validation_sample, validation_csv, na = "")

labeled_validation <- validation_sample %>%
  filter(gold_label %in% c("relevant", "irrelevant", "unsure"))

if (nrow(labeled_validation) > 0) {
  openai_eval <- classification_metrics(
    labeled_validation$gold_label,
    labeled_validation$openai_label
  )
  local_eval <- classification_metrics(
    labeled_validation$gold_label,
    labeled_validation$local_label
  )
  ensemble_eval <- classification_metrics(
    labeled_validation$gold_label,
    labeled_validation$ensemble_label
  )

  metrics_rows <- metrics_rows %>%
    bind_rows(
      metric_row("validation", "labeled_rows", nrow(labeled_validation)),
      metric_row("validation_openai", "accuracy", openai_eval$accuracy),
      metric_row("validation_openai", "macro_f1", openai_eval$macro_f1),
      metric_row("validation_local", "accuracy", local_eval$accuracy),
      metric_row("validation_local", "macro_f1", local_eval$macro_f1),
      metric_row("validation_ensemble", "accuracy", ensemble_eval$accuracy),
      metric_row("validation_ensemble", "macro_f1", ensemble_eval$macro_f1)
    )
} else {
  metrics_rows <- metrics_rows %>%
    bind_rows(metric_row("validation", "labeled_rows", 0))
}

readr::write_csv(metrics_rows, metrics_csv, na = "")

metrics_payload <- list(
  generated_at = timestamp_utc(),
  total_rows = nrow(scored_df),
  agreement = list(
    raw_agreement_rate = mean(
      scored_df$openai_label == scored_df$local_label,
      na.rm = TRUE
    ),
    disagreement_rate = mean(
      scored_df$openai_label != scored_df$local_label,
      na.rm = TRUE
    ),
    share_any_unsure = mean(
      scored_df$openai_label == "unsure" | scored_df$local_label == "unsure",
      na.rm = TRUE
    )
  ),
  distributions = list(
    openai = scored_df %>%
      count(openai_label, name = "n") %>%
      arrange(openai_label),
    local = scored_df %>%
      count(local_label, name = "n") %>%
      arrange(local_label),
    ensemble_action = scored_df %>%
      count(final_action, name = "n") %>%
      arrange(final_action)
  ),
  validation = list(
    sample_path = validation_csv,
    labeled_rows = nrow(labeled_validation)
  )
)

write(
  jsonlite::toJSON(
    metrics_payload,
    pretty = TRUE,
    auto_unbox = TRUE,
    null = "null"
  ),
  metrics_json
)

logger$info(glue("Scored output written to {output_csv} and {output_parquet}"))
logger$info(glue("Validation sample written to {validation_csv}"))
logger$info(glue("Metrics written to {metrics_json} and {metrics_csv}"))
