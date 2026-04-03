source(here::here("src/epp_namelist/common_multilingual_names.R"))

pacman::p_load(rvest)

eppo_common_name_cache <- new.env(parent = emptyenv())

eppo_language_groups <- list(
  en = c("English", "English (AU)", "English (GB)", "English (US)"),
  es = c("Spanish", "Spanish (HN)"),
  fr = c("French"),
  pt = c("Portuguese")
)

split_plant_common_terms <- function(x, lowercase = FALSE) {
  values <- split_terms(x, "\n|,|;|\\|")
  if (lowercase) {
    values <- stringr::str_to_lower(values)
  }
  dedupe_terms(values)
}

extract_scientific_candidates <- function(
  eppo_species,
  col_scientific_name,
  gbif_species,
  itis_scientific_name,
  ncbi_scientific_name
) {
  dedupe_terms(c(
    eppo_species,
    col_scientific_name,
    gbif_species,
    itis_scientific_name,
    ncbi_scientific_name
  ))
}

fetch_eppo_common_terms <- function(eppo_taxon_id) {
  key <- paste0("eppo::", normalize_text(eppo_taxon_id))
  cache_get_or_set(
    eppo_common_name_cache,
    key,
    function() {
      term_map <- blank_term_map()
      if (is_blank_text(eppo_taxon_id)) {
        return(term_map)
      }

      page <- tryCatch(
        {
          pause_for_api()
          response <- httr2::request(
            sprintf("https://gd.eppo.int/taxon/%s", normalize_text(eppo_taxon_id))
          ) %>%
            httr2::req_user_agent("INFLUX-plant-diseases-name-builder/1.0") %>%
            httr2::req_timeout(30) %>%
            build_retry_policy(
              request_label = paste0("EPPO taxon ", normalize_text(eppo_taxon_id))
            ) %>%
            httr2::req_perform()

          rvest::read_html(httr2::resp_body_string(response))
        },
        error = function(e) {
          NULL
        }
      )

      if (is.null(page)) {
        return(term_map)
      }

      table_node <- rvest::html_element(page, "#tbcommon")
      if (length(table_node) == 0 || inherits(table_node, "xml_missing")) {
        return(term_map)
      }

      table_df <- tryCatch(
        {
          rvest::html_table(table_node, trim = TRUE)
        },
        error = function(e) {
          tibble::tibble()
        }
      )

      if (nrow(table_df) == 0) {
        return(term_map)
      }

      names(table_df) <- names(table_df) %>%
        as.character() %>%
        normalize_text() %>%
        stringr::str_to_lower()

      if (!all(c("name", "language") %in% names(table_df))) {
        return(term_map)
      }

      common_df <- table_df %>%
        dplyr::transmute(
          name = normalize_text(.data$name),
          language = normalize_text(.data$language)
        ) %>%
        dplyr::filter(
          !is.na(.data$name),
          .data$name != "",
          !is.na(.data$language),
          .data$language != ""
        )

      if (nrow(common_df) == 0) {
        return(term_map)
      }

      normalized_groups <- purrr::map(
        eppo_language_groups,
        ~ stringr::str_to_lower(.x)
      )

      for (lang_key in names(normalized_groups)) {
        target_languages <- normalized_groups[[lang_key]]

        term_map[[lang_key]] <- common_df %>%
          dplyr::filter(
            stringr::str_to_lower(.data$language) %in% target_languages
          ) %>%
          dplyr::pull(.data$name) %>%
          dedupe_terms()
      }

      term_map
    }
  )
}

build_plant_language_terms <- function(
  language,
  scientific_terms,
  english_alias_terms,
  eppo_term_map,
  wikidata_term_map
) {
  local_terms <- dedupe_terms(c(
    eppo_term_map[[language]],
    wikidata_term_map[[language]],
    wikidata_term_map[["mul"]]
  ))

  if (language == "en") {
    local_terms <- dedupe_terms(c(local_terms, english_alias_terms))
  }

  dedupe_terms(c(scientific_terms, local_terms))
}

enrich_plant_name_row <- function(
  eppo_taxon_id,
  eppo_species,
  col_scientific_name,
  gbif_species,
  itis_scientific_name,
  ncbi_scientific_name,
  pathogen_scientific_name_en,
  aka_en
) {
  scientific_candidates <- extract_scientific_candidates(
    eppo_species = eppo_species,
    col_scientific_name = col_scientific_name,
    gbif_species = gbif_species,
    itis_scientific_name = itis_scientific_name,
    ncbi_scientific_name = ncbi_scientific_name
  )

  scientific_terms <- dedupe_terms(c(
    pathogen_scientific_name_en,
    scientific_candidates
  ))
  english_alias_terms <- split_terms(aka_en, "\n")

  eppo_term_map <- fetch_eppo_common_terms(eppo_taxon_id)
  pathogen_wikidata_id <- resolve_wikidata_id(
    c(scientific_candidates, english_alias_terms),
    kind = "pathogen"
  )
  wikidata_term_map <- fetch_multilingual_terms(pathogen_wikidata_id)

  search_terms_en <- build_plant_language_terms(
    language = "en",
    scientific_terms = scientific_terms,
    english_alias_terms = english_alias_terms,
    eppo_term_map = eppo_term_map,
    wikidata_term_map = wikidata_term_map
  )
  search_terms_es <- build_plant_language_terms(
    language = "es",
    scientific_terms = scientific_terms,
    english_alias_terms = english_alias_terms,
    eppo_term_map = eppo_term_map,
    wikidata_term_map = wikidata_term_map
  )
  search_terms_fr <- build_plant_language_terms(
    language = "fr",
    scientific_terms = scientific_terms,
    english_alias_terms = english_alias_terms,
    eppo_term_map = eppo_term_map,
    wikidata_term_map = wikidata_term_map
  )
  search_terms_pt <- build_plant_language_terms(
    language = "pt",
    scientific_terms = scientific_terms,
    english_alias_terms = english_alias_terms,
    eppo_term_map = eppo_term_map,
    wikidata_term_map = wikidata_term_map
  )

  tibble::tibble(
    pathogen_wikidata_id = pathogen_wikidata_id,
    eppo_common_names_en = collapse_terms(eppo_term_map[["en"]]),
    eppo_common_names_es = collapse_terms(eppo_term_map[["es"]]),
    eppo_common_names_fr = collapse_terms(eppo_term_map[["fr"]]),
    eppo_common_names_pt = collapse_terms(eppo_term_map[["pt"]]),
    pathogen_terms_en = collapse_terms(search_terms_en),
    pathogen_terms_es = collapse_terms(search_terms_es),
    pathogen_terms_fr = collapse_terms(search_terms_fr),
    pathogen_terms_pt = collapse_terms(search_terms_pt),
    search_string = build_search_string(search_terms_en),
    search_string_en = build_search_string(search_terms_en),
    search_string_es = build_search_string(search_terms_es),
    search_string_fr = build_search_string(search_terms_fr),
    search_string_pt = build_search_string(search_terms_pt)
  )
}

prepare_plant_seed_workbook <- function(raw_input_path) {
  raw_df <- readxl::read_xlsx(raw_input_path, col_types = "text") %>%
    dplyr::mutate(dplyr::across(dplyr::everything(), normalize_text)) %>%
    dplyr::select(-dplyr::any_of(c(
      "pathogen_scientific_name_en",
      "aka_en",
      "pathogen_wikidata_id",
      "eppo_common_names_en",
      "eppo_common_names_es",
      "eppo_common_names_fr",
      "eppo_common_names_pt",
      "pathogen_terms_en",
      "pathogen_terms_es",
      "pathogen_terms_fr",
      "pathogen_terms_pt",
      "search_string",
      "search_string_en",
      "search_string_es",
      "search_string_fr",
      "search_string_pt"
    )))

  work_df <- raw_df %>%
    dplyr::rename(
      eppo_species = EPPO_Species,
      col_scientific_name = COL_scientificName,
      gbif_species = GBIF_species,
      itis_scientific_name = `IT IS_scientificName`,
      ncbi_scientific_name = NCBI_ScientificName,
      itis_common_names = `IT IS_commonNames`,
      ncbi_common_name = NCBI_CommonName,
      eppo_taxon_id = EPPO_taxonID
    )

  raw_df %>%
    dplyr::mutate(
      pathogen_scientific_name_en = dplyr::coalesce(
        dplyr::na_if(work_df$eppo_species, ""),
        dplyr::na_if(work_df$col_scientific_name, ""),
        dplyr::na_if(work_df$gbif_species, ""),
        dplyr::na_if(work_df$itis_scientific_name, ""),
        dplyr::na_if(work_df$ncbi_scientific_name, "")
      ),
      aka_en = purrr::map2_chr(
        work_df$itis_common_names,
        work_df$ncbi_common_name,
        ~ collapse_terms(dedupe_terms(c(
          split_plant_common_terms(.x, lowercase = TRUE),
          split_plant_common_terms(.y, lowercase = TRUE)
        )))
      )
    ) %>%
    dplyr::relocate(
      pathogen_scientific_name_en,
      aka_en,
      .after = NCBI_CommonName
    )
}

build_plant_multilingual_name_list <- function(raw_input_path, output_path) {
  raw_df <- prepare_plant_seed_workbook(raw_input_path)

  input_df <- raw_df %>%
    dplyr::transmute(
      eppo_taxon_id = .data$EPPO_taxonID,
      eppo_species = .data$EPPO_Species,
      col_scientific_name = .data$COL_scientificName,
      gbif_species = .data$GBIF_species,
      itis_scientific_name = .data$`IT IS_scientificName`,
      ncbi_scientific_name = .data$NCBI_ScientificName,
      pathogen_scientific_name_en = .data$pathogen_scientific_name_en,
      aka_en = .data$aka_en
    )

  enriched_df <- run_rowwise_enrichment(
    data = input_df,
    enrich_fn = enrich_plant_name_row,
    label_fn = function(row) {
      c(
        row$pathogen_scientific_name_en,
        row$eppo_species,
        row$col_scientific_name,
        row$gbif_species,
        row$itis_scientific_name,
        row$ncbi_scientific_name,
        row$eppo_taxon_id
      )
    },
    task_label = "Plant multilingual name build"
  )

  output_df <- dplyr::bind_cols(raw_df, enriched_df) %>%
    dplyr::relocate(
      pathogen_scientific_name_en,
      aka_en,
      pathogen_wikidata_id,
      eppo_common_names_en,
      eppo_common_names_es,
      eppo_common_names_fr,
      eppo_common_names_pt,
      pathogen_terms_en,
      pathogen_terms_es,
      pathogen_terms_fr,
      pathogen_terms_pt,
      search_string,
      search_string_en,
      search_string_es,
      search_string_fr,
      search_string_pt,
      .after = NCBI_CommonName
    )

  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  writexl::write_xlsx(output_df, output_path)

  invisible(output_df)
}

build_plant_multilingual_name_list(
  raw_input_path = here::here("data/inputs/plant_diseases_raw.xlsx"),
  output_path = here::here("data/inputs/plant_diseases.xlsx")
)
