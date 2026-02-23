# Collects Google News RSS results for zika day-by-day in 2015 (after/before
# windows of 1 day), using Brazil Portuguese edition (ceid=BR:pt, hl=pt, gl=BR).

# 1) Collect Google News
pacman::p_load(tidyRSS, lubridate)

# Define search term and base URL for the RSS feed
keyword_base <- "https://news.google.com/rss/search?q=zika+after:"

# Define search date ranges
start_dates <- seq(ymd("2015-01-01"), ymd("2015-12-30"), by = "days")
end_dates <- seq(ymd("2015-01-02"), ymd("2015-12-31"), by = "days")

# Initialize an empty list to store data frames
results_list <- list()

# Loop through the date ranges and fetch the RSS feeds
for (i in seq_along(start_dates)) {
  start_date <- start_dates[i]
  end_date <- end_dates[i]
  keyword <- paste0(
    keyword_base,
    start_date,
    "+before:",
    end_date,
    "&ceid=BR:pt&hl=pt&gl=BR"
  )

  # Fetch the RSS feed and handle any errors
  google_news <- try(
    tidyfeed(keyword, clean_tags = TRUE, parse_dates = TRUE),
    silent = TRUE
  )

  # Short delay to avoid rate limiting
  Sys.sleep(1)

  # Check if the feed was fetched successfully
  if (inherits(google_news, "try-error")) {
    message(paste(
      "Failed to fetch feed for date range:",
      start_date,
      "to",
      end_date
    ))
    next
  }

  # Convert the feed to a data frame and check for validity
  if (nrow(google_news) > 0) {
    df <- apply(google_news, 2, as.character)
    results_list[[i]] <- df
  } else {
    message(paste("No results for date range:", start_date, "to", end_date))
  }
}

# Combine all data frames into one, if there are any results
if (length(results_list) > 0) {
  combined_df <- do.call(rbind, results_list)

  # Write the combined data frame to a CSV file
  write.csv(combined_df, "~/data/Zika_News_2015.csv", row.names = FALSE)
} else {
  message("No data collected.")
}
