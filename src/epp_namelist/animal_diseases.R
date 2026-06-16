source(here::here("src/epp_namelist/common_multilingual_names.R"))

build_multilingual_name_list(
  raw_input_path = here::here("data/inputs/animal_diseases_raw.xlsx"),
  output_path = here::here("data/inputs/animal_diseases.xlsx")
)
