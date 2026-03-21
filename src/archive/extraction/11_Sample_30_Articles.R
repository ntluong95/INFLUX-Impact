# 11) 5 Randomly Sampled Articles of Each EPP for Manual Classification

library(readr)
library(dplyr)
library(stringr)

in_file  <- "Relevant.csv"
out_samp <- "30_Sampled.csv"
out_rest <- "Leftover.csv"
set.seed(14)

df <- read_csv(in_file, show_col_types = FALSE)

df <- df %>%
  mutate(
    feed_title = as.character(feed_title),
    feed_title = str_replace(feed_title, "^[\"'\\u201C\\u201D\\u2018\\u2019]+", ""),
    feed_title = str_trim(feed_title)
  )

first_token <- function(x) {
  str_to_lower(str_split_fixed(str_trim(as.character(x)), "\\s+", 2)[, 1])
}

df <- df %>%
  mutate(
    .rowid = row_number(),
    first  = first_token(feed_title),
    species = case_when(
      first == "dengue" ~ "Dengue",
      first == "zika"   ~ "Zika",
      first == "ebola"  ~ "Ebola",
      first %in% c("screw","screw-worm","screwworm","cochliomyia-hominivorax") ~ "Screwworm",
      first %in% c("schistocerca","desert-locust","desert")                    ~ "Desert locust",
      TRUE ~ NA_character_
    )
  )

# Shuffle within each species, then take first 5 rows from each
sampled <- df %>%
  filter(!is.na(species)) %>%
  group_by(species) %>%
  mutate(.rand = runif(n())) %>%
  arrange(species, .rand) %>%
  slice_head(n = 5) %>%
  ungroup() %>%
  select(-.rand)

remaining <- df %>% filter(!.rowid %in% sampled$.rowid)

sampled  <- sampled  %>% select(-.rowid, -first)
remaining <- remaining %>% select(-.rowid, -first)

write_csv(sampled,  out_samp)
write_csv(remaining, out_rest)

cat("Sampled", nrow(sampled),  "rows →", out_samp,  "\n")
cat("Remaining", nrow(remaining), "rows →", out_rest, "\n")


