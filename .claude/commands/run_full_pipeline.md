# Run Full Pipeline

Run the full pathogen pipeline from repo root:

```bash
python3 src/run_pipeline.py --stage all
```

Then:

1. Summarize per-dataset status from logs and key output files.
2. Explicitly report if any datasets are paused because headline batches are still pending.
3. Provide the next command to resume safely.

