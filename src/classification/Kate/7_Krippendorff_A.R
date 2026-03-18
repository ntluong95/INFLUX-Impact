# 7) Krippendorff A 

# install.packages(c("readr","dplyr","tidyr","stringr","purrr","irr"))
library(readr); library(dplyr); library(tidyr); library(stringr); library(purrr); library(irr)

# ==== CONFIG ====
INPUT_CSV <- "Krippendorffs.csv"      # <- change to your file
RUN_COLS  <- c("run1","run2","run3","run4","run5")  # <- change to your column names
LEVEL     <- "nominal"   # "nominal" or "ordinal"
B         <- 2000        # bootstrap reps for 95% CI

# ==== LOAD ====
df <- read_csv(INPUT_CSV, show_col_types = FALSE) %>%
  mutate(across(all_of(RUN_COLS), ~na_if(trimws(as.character(.x)), "")))

# ==== SUMMARY OF LABELS & MISSINGNESS ====
label_counts <- df %>%
  pivot_longer(all_of(RUN_COLS), names_to="run", values_to="label") %>%
  count(label, sort = TRUE)

missing_by_item <- df %>%
  transmute(missing = rowSums(is.na(across(all_of(RUN_COLS))))) %>%
  pull(missing)

cat("\nLabel distribution across all runs:\n")
print(label_counts)
cat("\nItems with k missing labels (k=0..5):\n")
print(table(missing_by_item))

# ==== KRIPPENDORFF'S ALPHA ====
# irr::kripp.alpha expects a matrix: coders x items
ratings <- t(as.matrix(df[, RUN_COLS]))

# Nominal/ordinal selection
method <- match.arg(LEVEL, c("nominal","ordinal"))

alpha_obj <- irr::kripp.alpha(ratings, method = method)
alpha_hat <- alpha_obj$value

cat(sprintf("\nKrippendorff's alpha (%s): %.4f\n", method, alpha_hat))

# ==== BOOTSTRAP 95% CI (resample items with replacement) ====
set.seed(42)
boot_vals <- replicate(B, {
  idx <- sample.int(n = ncol(ratings), size = ncol(ratings), replace = TRUE)
  suppressWarnings(irr::kripp.alpha(ratings[, idx, drop=FALSE], method = method)$value)
})

ci <- quantile(boot_vals, c(0.025, 0.975), na.rm = TRUE)
cat(sprintf("95%% bootstrap CI: [%.4f, %.4f]  (B=%d)\n", ci[1], ci[2], B))

# ==== OPTIONAL: AGREEMENT MATRIX (how often coders agree per item) ====
agreement_rate <- df %>%
  rowwise() %>%
  mutate(agr = {
    x <- c_across(all_of(RUN_COLS)); x <- x[!is.na(x)]
    if(length(x) <= 1) NA_real_ else max(table(x)) / length(x)
  }) %>% ungroup() %>% pull(agr)

cat("\nAgreement rate (majority proportion) summary:\n")
print(summary(agreement_rate))

# ==== SAVE A SMALL REPORT ====
summary_out <- tibble(
  alpha_level = method,
  alpha = alpha_hat,
  ci_low = ci[1],
  ci_high = ci[2],
  n_items = ncol(ratings),
  n_coders = nrow(ratings)
)
write_csv(summary_out, "Alpha_Summary.csv")
cat('\nWrote Alpha_Summary.csv\n')
