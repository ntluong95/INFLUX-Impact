# Run RSS for Subset

Run RSS retrieval for selected domain-language slices:

```bash
python3 src/run_pipeline.py --stage rss --pathogen-domains human,animal --languages en,fr,es,pt
```

Guidance:

1. Keep outputs separated by dataset key.
2. Summarize manifest status (`success`, `empty`, `split`, `failed`, `throttled`) from each dataset manifest.
3. Mention whether the run paused due to throttling and what to rerun next.

