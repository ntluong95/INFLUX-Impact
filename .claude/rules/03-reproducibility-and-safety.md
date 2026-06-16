# Reproducibility and Safety Rules

## Secrets and Credentials

- Never hardcode secrets.
- Use local `.env` for credentials.
- Required for headline filtering: `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` (depending on `classification.provider` config)
- Optional for faster NCBI API calls in R scripts: `ENTREZ_KEY`

## Reproducibility

- Preserve resumable behavior for all stages.
- Reuse stage state artifacts unless `--force` is explicitly set.
- Prefer config changes in `src/config/pipeline.yaml` over hardcoded constants.

## Output and Logging Discipline

- Keep dataset-key-specific outputs in established `data/` paths.
- Keep logs in `data/logs/`.
- Keep manifest, batch registry, and checkpoint artifacts for auditability.

## Change Discipline

- Do not rewrite unrelated files.
- Avoid destructive cleanup of existing data by default.
- For new behavior, include clear stage-local comments where non-obvious decisions are made.

