# Resume Headline Filtering

Resume domain filtering + OpenAI Batch hydration/submission:

```bash
python3 src/run_pipeline.py --stage classify
```

After running:

1. Report pending row count per dataset from `*_headlines_classified.csv`.
2. Report batch status summary from each dataset registry under `data/intermediate/openai_batches/`.
3. If all datasets are ready, suggest continuing with:

```bash
python3 src/run_pipeline.py --stage fulltext
```

