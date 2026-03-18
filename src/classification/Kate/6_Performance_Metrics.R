# 6) Performance Metrics (Recall, Precision, F1)

# Compare 1 gold CSV vs 5 prediction CSVs for "News outlet (Yes/No)"

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(stringr)
  library(yardstick)  # precision, recall, f_meas
})

# ---------- SET YOUR FILES ----------
gold_file <- "Gold.csv"  
pred_files <- c(          
  "Domains_Sampled_100.csv",
  "Domains_Sampled_100_2.csv",
  "Domains_Sampled_100_3.csv",
  "Domains_Sampled_100_4.csv",
  "Domains_Sampled_100_5.csv"
)
target_col <- "News outlet (Yes/No)"  # column to evaluate
positive_label <- "Yes"               # positive class
out_file <- "Per_Run_Metrics.csv"
# -----------------------------------

norm_names <- function(x) trimws(x)
norm_domain <- function(x) {
  x %>% as.character() %>% str_trim() %>% str_to_lower() %>% str_replace("^\ufeff", "")
}
has_domain <- function(df) "Domain" %in% names(df)

read_frame <- function(path, target) {
  df <- read_csv(path, show_col_types = FALSE)
  names(df) <- norm_names(names(df))
  if (!(target %in% names(df))) {
    stop(sprintf("File '%s' is missing the '%s' column.", path, target))
  }
  if (has_domain(df)) df$Domain <- norm_domain(df$Domain)
  df
}

eval_binary <- function(truth, estimate, positive = "Yes") {
  # Yardstick expects factors with event_level = "first"
  lvls <- c(positive, setdiff(unique(c(truth, estimate)), positive))
  tib <- tibble(
    truth = factor(truth, levels = lvls),
    estimate = factor(estimate, levels = lvls)
  )
  tibble(
    precision = precision(tib, truth, estimate, event_level = "first")$.estimate,
    recall    = recall(   tib, truth, estimate, event_level = "first")$.estimate,
    f1        = f_meas(   tib, truth, estimate, event_level = "first")$.estimate
  )
}

# ---- load gold ----
gold <- read_frame(gold_file, target_col) %>%
  transmute(
    Domain = if (has_domain(.)) Domain else row_number(),  # row index fallback
    truth  = as.character(.data[[target_col]]) %>% str_trim()
  )

# ---- evaluate each prediction ----
rows <- list()
for (pf in pred_files) {
  pred <- read_frame(pf, target_col) %>%
    transmute(
      Domain = if (has_domain(.)) Domain else row_number(),
      pred   = as.character(.data[[target_col]]) %>% str_trim()
    )
  
  merged <- gold %>% inner_join(pred, by = "Domain")
  n_overlap <- nrow(merged)
  
  if (n_overlap == 0) {
    rows[[pf]] <- tibble(run = pf, overlap = 0, precision = NA_real_, recall = NA_real_, f1 = NA_real_)
    next
  }
  
  mets <- eval_binary(merged$truth, merged$pred, positive = positive_label)
  rows[[pf]] <- tibble(run = pf, overlap = n_overlap) %>% bind_cols(mets)
}

summary_tbl <- bind_rows(rows) %>% arrange(desc(f1))

# ---- output ----
print(summary_tbl, n = nrow(summary_tbl))
write_csv(summary_tbl, out_file)
cat("\nSaved →", out_file, "\n")





