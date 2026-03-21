# 3) Scrape Google News URLs

library(chromote)
library(rvest)
library(httr)
library(jsonlite)
library(stringr)
library(urltools)   
library(dplyr)
library(future.apply)

#Read CSV with column named "url" 
urls_df <- read.csv("Zika.csv", stringsAsFactors = FALSE)

# Specify rows want to process
rows_to_process <- 1:4000

# Filter out troublesome URLs from the original list 
# These were creating formatting issues in final csv
# Full texts were likely exceeding cell character limit causing spillover 
all_urls <- urls_df$url[rows_to_process]
all_urls <- all_urls[ !grepl("philstar\\.com|nature\\.com|biomedcentral\\.com|reliefweb\\.int", all_urls, ignore.case = TRUE) ]

# Fallback Logic for Final URL Resolution
# URLs retrieved from Google News RSS initially go to intermediary URLs or landing pages 
# Which we want to bypass to get to the desired content 

# 3.1 The main driver: process_url()
process_url <- function(url, max_attempts = 3) {
  attempt <- 1
  final_url <- url
  nav_success <- FALSE
  
  while (attempt <= max_attempts) {
    cat(sprintf("[process_url] Attempt %d to navigate: %s\n", attempt, url))
    
    session <- ChromoteSession$new()
    
    # Try to navigate
    tryCatch({
      session$Page$navigate(url)
      Sys.sleep(5)  # wait for page load
      nav_success <- TRUE
    }, error = function(e) {
      cat("    Navigation error on attempt", attempt, "for URL:", url, "\n    Message:", e$message, "\n")
    })
    
    if (nav_success) {
      # If navigation worked, run fallback logic
      final_url <- fallback_logic(session, url)
      session$close()
      break
    } else {
      session$close()
      attempt <- attempt + 1
      Sys.sleep(2)
    }
  }
  
  if (!nav_success) {
    cat("[process_url] All attempts failed for:", url, "\nReturning original URL.\n")
  }
  
  return(final_url)
}

# 3.2 The fallback_logic() function
fallback_logic <- function(session, original_url) {
  # 1) Check final navigation history URL
  final_url <- tryCatch({
    nav_hist <- session$Page$getNavigationHistory()
    if (length(nav_hist$entries) > 0) {
      nav_hist$entries[[length(nav_hist$entries)]]$url
    } else {
      original_url
    }
  }, error = function(e) {
    cat("Error retrieving navigation history for:", original_url, "\n", e$message, "\n")
    return(original_url)
  })
  
  # If final_url is still Google News, do fallback checks
  if (grepl("news\\.google\\.com", final_url, ignore.case = TRUE)) {
    cat("[fallback_logic] Google News URL detected =>", final_url, "\n")
    
    final_url <- fallback_canonical(session, final_url)
    if (grepl("news\\.google\\.com", final_url, ignore.case = TRUE)) {
      final_url <- fallback_meta_refresh(session, final_url)
    }
    if (grepl("news\\.google\\.com", final_url, ignore.case = TRUE)) {
      final_url <- fallback_anchor_scan(session, final_url)
    }
    if (grepl("news\\.google\\.com", final_url, ignore.case = TRUE)) {
      final_url <- fallback_query_param(session, final_url)
    }
    if (grepl("news\\.google\\.com", final_url, ignore.case = TRUE)) {
      final_url <- fallback_jsonld(session, final_url)
    }
    if (grepl("news\\.google\\.com", final_url, ignore.case = TRUE)) {
      final_url <- fallback_html_regex(session, final_url)
    }
  }
  
  cat("[fallback_logic] Final URL =>", final_url, "\n")
  return(final_url)
}

# 3.3 Fallback helper: fallback_canonical
fallback_canonical <- function(session, current_url) {
  # Attempt <link rel="canonical">
  can_url <- tryCatch({
    res <- session$Runtime$evaluate('
      (function() {
        let link = document.querySelector("link[rel=\'canonical\']");
        return link ? link.href : "";
      })();
    ')
    res$result$value
  }, error = function(e) "")
  
  if (nzchar(can_url) && !grepl("news\\.google\\.com", can_url, ignore.case=TRUE)) {
    cat("    [fallback_canonical] Found canonical link =>", can_url, "\n")
    return(can_url)
  } else if (nzchar(can_url)) {
    cat("    [fallback_canonical] Canonical is Google News =>", can_url, "\n")
  } else {
    cat("    [fallback_canonical] No canonical link found.\n")
  }
  return(current_url)
}

# 3.4 Fallback helper: fallback_meta_refresh
fallback_meta_refresh <- function(session, current_url) {
  meta_refresh <- tryCatch({
    res <- session$Runtime$evaluate('
      (function(){
        let meta = document.querySelector("meta[http-equiv=\'refresh\']");
        return meta ? meta.content : "";
      })();
    ')
    res$result$value
  }, error = function(e) "")
  
  if (nzchar(meta_refresh)) {
    cat("    [fallback_meta_refresh] Found meta refresh =>", meta_refresh, "\n")
    refresh_url <- sub(".*url=([^;]+).*", "\\1", meta_refresh)
    if (nzchar(refresh_url) && grepl("^http", refresh_url)) {
      cat("    Navigating to meta refresh URL =>", refresh_url, "\n")
      # Attempt navigation
      tryCatch({
        session$Page$navigate(refresh_url)
        Sys.sleep(3)
        new_final <- tryCatch({
          hist <- session$Page$getNavigationHistory()
          if (length(hist$entries) > 0) hist$entries[[length(hist$entries)]]$url else refresh_url
        }, error = function(e) refresh_url)
        
        if (!grepl("news\\.google\\.com", new_final, ignore.case=TRUE)) {
          return(new_final)
        } else {
          cat("    Meta refresh => still Google News =>", new_final, "\n")
          return(new_final)
        }
      }, error = function(e) {
        cat("    [fallback_meta_refresh] Error navigating =>", refresh_url, "\n", e$message, "\n")
      })
    }
  }
  return(current_url)
}

# 3.5 Fallback helper: fallback_anchor_scan
fallback_anchor_scan <- function(session, current_url) {
  anchors <- tryCatch({
    session$Runtime$evaluate('
      (function(){
        let as = document.querySelectorAll("a");
        let links = [];
        for (let i=0; i<as.length; i++){
          links.push(as[i].href);
        }
        return links;
      })();
    ')
  }, error = function(e) { NULL })
  
  if (!is.null(anchors)) {
    all_links <- anchors$result$value
    external_links <- all_links[ !grepl("google\\.", all_links, ignore.case = TRUE) ]
    if (length(external_links) > 0) {
      cat("    [fallback_anchor_scan] External link =>", external_links[1], "\n")
      return(external_links[1])
    } else {
      cat("    [fallback_anchor_scan] No external anchors found.\n")
    }
  }
  return(current_url)
}

# 3.6 Fallback helper: fallback_query_param
fallback_query_param <- function(session, current_url) {
  parsed <- parse_url(current_url)
  check_params <- c("continue", "url", "u")
  found_param <- NA
  
  for (nm in check_params) {
    if (!is.null(parsed$query[[nm]])) {
      found_param <- parsed$query[[nm]]
      if (nzchar(found_param)) {
        break
      }
    }
  }
  
  if (!is.na(found_param) && grepl("^http", found_param)) {
    cat("    [fallback_query_param] Param =>", nm, "=", found_param, "\n")
    tryCatch({
      session$Page$navigate(found_param)
      Sys.sleep(3)
      new_final <- tryCatch({
        hist <- session$Page$getNavigationHistory()
        if (length(hist$entries) > 0) hist$entries[[length(hist$entries)]]$url else found_param
      }, error = function(e) found_param)
      
      if (!grepl("news\\.google\\.com", new_final, ignore.case=TRUE)) {
        return(new_final)
      } else {
        cat("    Param link => still Google News =>", new_final, "\n")
        return(new_final)
      }
    }, error = function(e) {
      cat("    [fallback_query_param] Error =>", found_param, "\n", e$message, "\n")
    })
  } else {
    cat("    [fallback_query_param] No param (continue/url/u) with valid link.\n")
  }
  return(current_url)
}

# 3.7 Fallback helper: fallback_jsonld
fallback_jsonld <- function(session, current_url) {
  scripts <- tryCatch({
    session$Runtime$evaluate('
      (function(){
        let out = [];
        let s = document.querySelectorAll("script[type=\'application/ld+json\']");
        for (let i=0; i<s.length; i++) {
          out.push(s[i].innerText);
        }
        return out;
      })();
    ')
  }, error = function(e) { NULL })
  
  if (!is.null(scripts)) {
    all_json <- scripts$result$value
    for (json_str in all_json) {
      possible_obj <- NULL
      # parse each JSON-LD block
      tryCatch({
        possible_obj <- fromJSON(json_str)
      }, error = function(e) {})
      
      if (!is.null(possible_obj) && is.list(possible_obj)) {
        link_candidates <- find_url_fields(possible_obj)
        # filter out google
        link_candidates <- link_candidates[ !grepl("google\\.", link_candidates, ignore.case=TRUE) ]
        if (length(link_candidates) > 0) {
          cat("    [fallback_jsonld] Found external URL =>", link_candidates[1], "\n")
          return(link_candidates[1])
        }
      }
    }
    cat("    [fallback_jsonld] No external URLs found in JSON-LD.\n")
  }
  return(current_url)
}

# Recursively find 'url' or 'mainEntityOfPage' fields in JSON-LD
find_url_fields <- function(x) {
  results <- character(0)
  if (is.list(x)) {
    for (nm in names(x)) {
      val <- x[[nm]]
      if (nm %in% c("url","mainEntityOfPage") && is.character(val)) {
        results <- c(results, val)
      }
      # Recurse
      if (is.list(val) || is.data.frame(val)) {
        results <- c(results, find_url_fields(val))
      }
    }
  }
  return(results)
}

# 3.8 Fallback helper: fallback_html_regex
fallback_html_regex <- function(session, current_url) {
  html_res <- tryCatch({
    session$Runtime$evaluate('document.documentElement.outerHTML')
  }, error = function(e) NULL)
  
  if (!is.null(html_res) && !is.null(html_res$result$value)) {
    html <- html_res$result$value
    # Regex for "http(s)://(not google) until space or quote"
    pattern <- "(https?://(?![^/]*google\\.).+?)(?=[\"' <])"
    matches <- str_match_all(html, pattern)[[1]]
    if (!is.null(matches) && nrow(matches) > 0) {
      cat("    [fallback_html_regex] Found external link =>", matches[1,1], "\n")
      return(matches[1,1])
    } else {
      cat("    [fallback_html_regex] No external link found.\n")
    }
  }
  return(current_url)
}

##################################################
#   Parsing Date, Text, and Domain from HTML     #
##################################################

extract_publication_date <- function(page) {
  # 1) Try <meta property="article:published_time">
  meta_time <- page %>%
    html_nodes("meta[property='article:published_time']") %>%
    html_attr("content")
  if (length(meta_time) > 0 && nzchar(meta_time[1])) {
    return(meta_time[1])
  }
  
  # 2) Try <meta name="date"> or <meta name="pubdate">
  meta_date <- page %>%
    html_nodes("meta[name='date'], meta[name='pubdate']") %>%
    html_attr("content")
  if (length(meta_date) > 0 && nzchar(meta_date[1])) {
    return(meta_date[1])
  }
  
  # 3) Sometimes there's a <time> element
  time_text <- page %>%
    html_node("time") %>%
    html_attr("datetime")
  if (!is.na(time_text) && nzchar(time_text)) {
    return(time_text)
  }
  
  # 4) Attempt JSON-LD scanning for datePublished or similar
  ld_json <- page %>%
    html_nodes("script[type='application/ld+json']") %>%
    html_text(trim = TRUE)
  if (length(ld_json) > 0) {
    for (txt in ld_json) {
      # parse JSON safely
      tmp <- NULL
      tryCatch({
        tmp <- fromJSON(txt)
      }, error=function(e){})
      if (!is.null(tmp)) {
        # If this is a list with "datePublished"
        if (!is.null(tmp$datePublished) && nzchar(tmp$datePublished[1])) {
          return(tmp$datePublished[1])
        }
        
        
        # Some sites nest inside "article" or "newsArticle"
        if (is.list(tmp$article)) {
          if (!is.null(tmp$article$datePublished)) {
            return(tmp$article$datePublished)
          }
        }
        if (is.list(tmp$newsArticle)) {
          if (!is.null(tmp$newsArticle$datePublished)) {
            return(tmp$newsArticle$datePublished)
          }
        }
      }
    }
  }
  
  # 5) Fallback: Just NA
  return(NA_character_)
}

extract_full_text <- function(page) {
  # 1) If the site uses <article> as main container
  article_p <- page %>% 
    html_nodes("article p") %>%
    html_text(trim = TRUE)
  
  # If found text in article>p
  if (length(article_p) > 0) {
    return(paste(article_p, collapse = "\n\n"))
  }
  
  # 2) Fallback: gather all <p> (be aware this might be messy)
  all_p <- page %>%
    html_nodes("p") %>%
    html_text(trim = TRUE)
  
  if (length(all_p) > 0) {
    return(paste(all_p, collapse = "\n\n"))
  }
  
  return(NA_character_)
}

extract_domain <- function(url) {
  parsed <- urltools::url_parse(url)
  return(parsed$domain)
}

get_article_data <- function(original_url) {
  final_url <- process_url(original_url)
  
  # Skip non-HTML resources by file extension
  if (grepl("\\.(ico|jpg|jpeg|png|gif)$", final_url, ignore.case = TRUE)) {
    message("Skipping non-HTML resource: ", final_url)
    return(data.frame(
      original_url = original_url,
      final_url    = final_url,
      domain       = extract_domain(final_url),
      pub_date     = NA_character_,
      full_text    = NA_character_,
      stringsAsFactors = FALSE
    ))
  }
  
  # Try to GET the URL with a timeout, catching errors like timeouts or connection issues
  response <- tryCatch({
    httr::GET(final_url, httr::user_agent("Mozilla/5.0"), httr::timeout(60))
  }, error = function(e) {
    message("Timeout or connection error for URL: ", final_url, "\n  Message: ", e$message)
    return(NULL)
  })
  
  if (is.null(response)) {
    message("Skipping URL due to GET error: ", final_url)
    return(data.frame(
      original_url = original_url,
      final_url    = final_url,
      domain       = extract_domain(final_url),
      pub_date     = NA_character_,
      full_text    = NA_character_,
      stringsAsFactors = FALSE
    ))
  }
  
  # Check for non-200 status codes
  if (httr::status_code(response) != 200) {
    message("Skipping URL due to non-200 status: ", final_url)
    return(data.frame(
      original_url = original_url,
      final_url    = final_url,
      domain       = extract_domain(final_url),
      pub_date     = NA_character_,
      full_text    = NA_character_,
      stringsAsFactors = FALSE
    ))
  }
  
  # Check the Content-Type header and also extract the raw HTML content
  content_type <- httr::headers(response)$`content-type`
  if (is.null(content_type) || length(content_type) == 0 || 
      !grepl("text/html", content_type, ignore.case = TRUE)) {
    message("Skipping URL due to non-HTML or missing content type: ", final_url, 
            " (Content-Type: ", content_type, ")")
    return(data.frame(
      original_url = original_url,
      final_url    = final_url,
      domain       = extract_domain(final_url),
      pub_date     = NA_character_,
      full_text    = NA_character_,
      stringsAsFactors = FALSE
    ))
  }
  
  # Get the raw HTML as text
  html_text <- httr::content(response, "text", encoding = "UTF-8")
  
  # Check if the page is essentially blank (you can adjust the threshold as needed)
  if (nchar(trimws(html_text)) < 100) {
    message("Skipping URL due to blank or near-blank page (possible authentication requirement): ", final_url)
    return(data.frame(
      original_url = original_url,
      final_url    = final_url,
      domain       = extract_domain(final_url),
      pub_date     = NA_character_,
      full_text    = NA_character_,
      stringsAsFactors = FALSE
    ))
  }
  
  # Attempt to parse the HTML
  page <- tryCatch({
    read_html(html_text)
  }, error = function(e) {
    message("[get_article_data] Error reading HTML => ", final_url, "\n  Message: ", e$message)
    NULL
  })
  
  if (is.null(page) || !inherits(page, "xml_document")) {
    message("Skipping URL due to invalid page: ", final_url)
    return(data.frame(
      original_url = original_url,
      final_url    = final_url,
      domain       = extract_domain(final_url),
      pub_date     = NA_character_,
      full_text    = NA_character_,
      stringsAsFactors = FALSE
    ))
  }
  
  # If we get here, the page appears valid. Extract data.
  dom <- extract_domain(final_url)
  date_found <- extract_publication_date(page)
  text_found <- extract_full_text(page)
  
  data.frame(
    original_url = original_url,
    final_url    = final_url,
    domain       = dom,
    pub_date     = date_found,
    full_text    = text_found,
    stringsAsFactors = FALSE
  )
}

# Number of parallel workers
plan(multisession, workers = 4) 

urls_to_process <- all_urls
num_urls <- length(urls_to_process)
batch_size <- 100
num_batches <- ceiling(num_urls / batch_size)

all_results <- list()

for (batch in seq_len(num_batches)) {
  start_idx <- (batch - 1) * batch_size + 1
  end_idx   <- min(batch * batch_size, num_urls)
  
  cat("\n=== Processing batch", batch, "of", num_batches,
      " => URLs [", start_idx, ":", end_idx, "] ===\n")
  
  batch_urls <- urls_to_process[start_idx:end_idx]
  
  # Parallel-lapply
  batch_out <- future_lapply(batch_urls, get_article_data, future.seed = TRUE)
  
  # Combine each list of data frames into a single DF
  batch_df <- do.call(rbind, batch_out)
  
  # Store results
  all_results[[batch]] <- batch_df
  
  # Write partial CSV for each batch
  out_file <- paste0("Zika_Batch_", batch, ".csv")
  write.csv(batch_df, out_file, row.names = FALSE)
  cat("   => Batch", batch, "written to", out_file, "\n")
}



