# Builds one RSS query per ISO country code for dengue (2000-01-01 to 2025-03-22)
# in English edition (hl=en, ceid=<country>:en)

# 2) Collect Google News ISO Codes
pacman::p_load(tidyRSS, dplyr, purrr, readr)

#fmt: skip
country_codes <- c(
  "AF","AX","AL","DZ","AS","AD","AO","AI","AQ","AG","AR","AM","AW","AU","AT","AZ",
  "BS","BH","BD","BB","BY","BE","BZ","BJ","BM","BT","BO","BQ","BA","BW","BV","BR",
  "IO","BN","BG","BF","BI","CV","KH","CM","CA","KY","CF","TD","CL","CN","CX","CC",
  "CO","KM","CG","CD","CK","CR","CI","HR","CU","CW","CY","CZ","DK","DJ","DM","DO",
  "EC","EG","SV","GQ","ER","EE","SZ","ET","FK","FO","FJ","FI","FR","GF","PF","TF",
  "GA","GM","GE","DE","GH","GI","GR","GL","GD","GP","GU","GT","GG","GN","GW","GY",
  "HT","HM","VA","HN","HK","HU","IS","IN","ID","IR","IQ","IE","IM","IL","IT","JM",
  "JP","JE","JO","KZ","KE","KI","KP","KR","KW","KG","LA","LV","LB","LS","LR","LY",
  "LI","LT","LU","MO","MG","MW","MY","MV","ML","MT","MH","MQ","MR","MU","YT","MX",
  "FM","MD","MC","MN","ME","MS","MA","MZ","MM","NA","NR","NP","NL","NC","NZ","NI",
  "NE","NG","NU","NF","MK","MP","NO","OM","PK","PW","PS","PA","PG","PY","PE","PH",
  "PN","PL","PT","PR","QA","RE","RO","RU","RW","BL","SH","KN","LC","MF","PM","VC",
  "WS","SM","ST","SA","SN","RS","SC","SL","SG","SX","SK","SI","SB","SO","ZA","GS",
  "SS","ES","LK","SD","SR","SJ","SE","CH","SY","TW","TJ","TZ","TH","TL","TG","TK",
  "TO","TT","TN","TR","TM","TC","TV","UG","UA","AE","GB","UM","UY","UZ","VU",
  "VE","VN","VG","VI","WF","EH","YE","ZM","ZW"
)

base_url <- "https://news.google.com/rss/search?q=dengue+after:2000-01-01+before:2025-03-22"
urls <- paste0(
  base_url,
  "&hl=en&gl=",
  country_codes,
  "&ceid=",
  country_codes,
  ":en"
)

pb <- txtProgressBar(min = 0, max = length(urls), style = 3)

feeds <- map2(
  urls,
  seq_along(urls),
  ~ {
    setTxtProgressBar(pb, .y)
    tryCatch(
      tidyfeed(.x, clean_tags = TRUE, parse_dates = TRUE),
      error = function(e) NULL
    )
  }
)
close(pb)

valid_feeds <- compact(feeds)
combined_df <- bind_rows(valid_feeds)

link_col <- intersect(
  c("item_link", "link", "guid", "item_guid_id"),
  names(combined_df)
)[1]
if (is.na(link_col)) {
  stop("No link-like column found; check names(combined_df).")
}

combined_df <- combined_df %>% distinct(.data[[link_col]], .keep_all = TRUE)

chr_cols <- names(combined_df)[vapply(combined_df, is.character, logical(1))]
combined_df[chr_cols] <- lapply(combined_df[chr_cols], as.character)


purrr::map_chr(combined_df, ~ class(.x)[1])
combined_df_clean <- combined_df %>%
  mutate(across(where(is.list), ~ map_chr(.x, ~ paste(.x, collapse = "; "))))

# write_csv(combined_df, "~/data/Dengue_ISO_EN.csv")
rio::export(combined_df_clean, here::here("data", "Dengue_ISO_EN.csv"))
