# Influx Pipeline Operator

Use this skill when running, debugging, or resuming the INFLUX pathogen pipeline.

## Goals

- Preserve stage-by-stage reproducibility.
- Keep dataset-key separation intact.
- Produce clear operational status after each run.

## Procedure

1. Confirm stage and target scope (`pathogen-domains`, `languages`).
2. Run the minimal stage needed (`rss`, `classify`, `fulltext`, `bertopic`, or `all`).
3. Validate stage outputs in `data/intermediate` or `data/final` for each dataset key.
4. Check logs in `data/logs` for warnings, throttling, and pending batch messages.
5. Report:
   - what completed
   - what is pending
   - exact next command

## Safety

- Never hardcode credentials.
- Do not delete historical manifests, batch registries, or full-text outputs unless explicitly requested.
- Use `--force` only when a fresh recomputation is intended.

