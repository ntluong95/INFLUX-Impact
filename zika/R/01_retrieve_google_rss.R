args_all <- commandArgs(trailingOnly = FALSE)
script_path <- normalizePath(
  sub("^--file=", "", args_all[grepl("^--file=", args_all)][[1]]),
  winslash = "/",
  mustWork = TRUE
)
zika_root <- normalizePath(file.path(dirname(script_path), ".."), winslash = "/", mustWork = TRUE)
repo_root <- normalizePath(file.path(zika_root, ".."), winslash = "/", mustWork = TRUE)

message("Deprecated: Stage 1 retrieval moved to Python using pygooglenews.")
message("Delegating to zika/python/01_retrieve_google_rss.py and zika/python/01_parse_google_rss.py.")

setwd(repo_root)

status_fetch <- system2(
  "uv",
  args = c("run", "python", "zika/python/01_retrieve_google_rss.py"),
  stdout = "",
  stderr = "",
  wait = TRUE
)

if (!identical(status_fetch, 0L)) {
  quit(save = "no", status = status_fetch)
}

status_parse <- system2(
  "uv",
  args = c("run", "python", "zika/python/01_parse_google_rss.py"),
  stdout = "",
  stderr = "",
  wait = TRUE
)

quit(save = "no", status = status_parse)
