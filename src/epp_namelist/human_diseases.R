source(here::here("src/epp_namelist/common_multilingual_names.R"))

build_multilingual_name_list(
  raw_input_path = here::here("data/inputs/human_diseases_raw.xlsx"),
  output_path = here::here("data/inputs/human_diseases.xlsx")
)

# raw_df <- prepare_name_seed_workbook("data/inputs/human_diseases_raw.xlsx")
# input_df <- raw_df %>%
#   dplyr::select(
#     pathogen_scientific_name_en,
#     pathogen_subtypes_names_en,
#     aka_en,
#     disease_name_en
#   )

# enriched_df <- run_rowwise_enrichment(
#   data = input_df,
#   #NOTE enrich_multilingual_name_row is the most important one
#   enrich_fn = enrich_multilingual_name_row,
#   label_fn = function(row) {
#     c(
#       row$disease_name_en,
#       row$pathogen_scientific_name_en,
#       row$pathogen_subtypes_names_en,
#       row$aka_en
#     )
#   },
#   task_label = "Multilingual name build"
# )
