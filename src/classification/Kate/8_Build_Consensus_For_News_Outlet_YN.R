# 8) Consensus of 5 runs of GPT Classified News Outlets Y/N  

pkgs <- c("readr","dplyr","stringr","purrr")
to_install <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(to_install)) install.packages(to_install, repos = "https://cloud.r-project.org")
invisible(lapply(pkgs, library, character.only = TRUE))

files <- c(
  "Run_1.csv",
  "Run_2.csv",
  "Run_3.csv",
  "Run_4.csv",
  "Run_5.csv"
)
out_file <- "Consensus_Outlets.csv"

norm_domain <- function(x) {
  x %>% as.character() %>% str_trim() %>% str_to_lower() %>% str_replace("^\ufeff", "")
}

majority_vote <- function(values) {
  vals <- values[!is.na(values) & nzchar(values)]
  if (length(vals) == 0) return(NA_character_)
  tab <- sort(table(vals), decreasing = TRUE)
  winners <- names(tab)[tab == max(tab)]
  sort(winners)[1]  # deterministic tie-break
}

load_run <- function(path) {
  df <- read_csv(path, show_col_types = FALSE)
  names(df) <- trimws(names(df))
  required <- c("Domain","News outlet (Yes/No)")
  missing <- setdiff(required, names(df))
  if (length(missing)) stop(sprintf("%s is missing columns: %s", path, paste(missing, collapse=", ")))
  
  df %>%
    transmute(
      Domain = norm_domain(.data$Domain),
      News   = as.character(`News outlet (Yes/No)`) %>% str_trim()
    ) %>%
    distinct(Domain, .keep_all = TRUE)
}

if (length(files) != 5) stop("Please provide exactly 5 CSV paths in `files`.")
runs <- map(files, load_run)
names(runs) <- paste0("r", seq_along(runs))

wide <- runs %>%
  imap(~ rename(.x, !!paste0("News_", .y) := News)) %>%
  reduce(full_join, by = "Domain")

news_cols <- grep("^News_", names(wide), value = TRUE)

consensus <- wide %>%
  rowwise() %>%
  mutate(
    `News outlet (Yes/No)` = majority_vote(c_across(all_of(news_cols)))
  ) %>%
  ungroup() %>%
  select(Domain, `News outlet (Yes/No)`) %>%
  arrange(Domain)

print(head(consensus, 12))
cat("\nRows:", nrow(consensus), "\nSaving to:", out_file, "\n")
write_csv(consensus, out_file)
cat("Done.\n")
