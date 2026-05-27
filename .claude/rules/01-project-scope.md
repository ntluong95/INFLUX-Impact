# Project Scope Rules

## Mission

Build and maintain a reproducible pathogen-news pipeline to analyze cascading social-ecological impacts.

## Supported Domains and Languages

- Pathogen domains: `human`, `animal`, `plant`
- Languages: `en`, `fr`, `es`, `pt`

## Dataset Key Contract

- Every stage writes outputs by dataset key: `<pathogen_domain>_<language_code>`.
- Expected keys:
  - `human_en`, `human_fr`, `human_es`, `human_pt`
  - `animal_en`, `animal_fr`, `animal_es`, `animal_pt`
  - `plant_en`, `plant_fr`, `plant_es`, `plant_pt`

## Source-of-Truth Priority

1. Runtime behavior in `src/` modules and `src/config/pipeline.yaml`
2. `src/README.md`
3. `src/epp_namelist/README.md`
4. root `README.md`
5. historical prompt docs in `src/prompt_codex*.md`

