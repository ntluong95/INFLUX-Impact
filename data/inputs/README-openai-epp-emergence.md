# OpenAI EPP emergence-event scripts

These scripts sit beside `EPPs input file.xlsx` and read:

- `Main list`
- `Country`

They load `OPENAI_API_KEY` from the project-root `.env` file and call the OpenAI Responses API with `gpt-5.1` by default.

Default workflow:

1. Search each selected EPP independently using `ORIGINAL_USER_PROMPT`.
2. Save intermediate search JSON named by standardized scientific name.
3. Peer-review that JSON with `VERIFICATION_USER_PROMPT`.
4. Save reviewed JSON named by standardized scientific name.
5. Write one `curated_batch.json` and `curated_batch.xlsx` per batch.

The API instruction prompt is stored verbatim in `epp_emergence_analysis_prompt.py`. Runtime row and `Country` sheet data are added as attached Excel context.

If imports are missing in a fresh environment, install the project dependencies or run:

```bash
python3 -m pip install openai python-dotenv pandas openpyxl
```

## Run by standardized scientific names

```bash
python3 data/inputs/epp_emergence_analysis_query_openai_events.py \
  --names "Vibrio cholerae" "Xylella fastidiosa"
```

Outputs are written in batch folders:

```text
data/inputs/openai_outputs/epp_emergence_run_YYYYMMDD_HHMMSS/
├── batch_manifest.json
└── batch_001/
    ├── search_json/
    ├── reviewed_json/
    ├── curated_batch.json
    └── curated_batch.xlsx
```

Each batch defaults to 30 EPPs:

```bash
python3 data/inputs/epp_emergence_analysis_query_openai_events.py \
  --all \
  --batch-size 30
```

Preview the input workbook, selected rows, and exact batch membership without
calling the API:

```bash
python3 data/inputs/epp_emergence_analysis_query_openai_events.py \
  --all \
  --batch-size 30 \
  --dry-run \
  --show-batches
```

Real runs also save the same plan to `batch_manifest.json` in the run folder.
Each `curated_batch.json` also contains the `selected_rows` used for that
batch.

If a run stops partway through, rerun against the same folder. Existing per-EPP JSON files are reused:

```bash
python3 data/inputs/epp_emergence_analysis_query_openai_events.py \
  --all \
  --run-dir data/inputs/openai_outputs/epp_emergence_run_YYYYMMDD_HHMMSS
```

Use a stronger review model or stronger review reasoning settings:

```bash
python3 data/inputs/epp_emergence_analysis_query_openai_events.py \
  --names "Orthoflavivirus denguei" \
  --model gpt-5.1 \
  --review-model gpt-5.1 \
  --review-reasoning-effort high
```

If a row has many events and the script reports incomplete JSON, rerun with a larger output cap:

```bash
python3 data/inputs/epp_emergence_analysis_query_openai_events.py \
  --names "Orthoflavivirus denguei" \
  --model gpt-5.1 \
  --max-output-tokens 50000 \
  --review-max-output-tokens 50000
```

## Run by row selectors

`--row-indices` uses zero-based row positions in `Main list`. `--input-rows`/`--excel-rows` uses Excel worksheet row numbers, where the first data row is `2`.

```bash
python3 data/inputs/epp_emergence_analysis_query_openai_events.py --row-indices 0 1 2
python3 data/inputs/epp_emergence_analysis_query_openai_events.py --input-rows 2 3 4
```

## Use a file of names

Line-delimited text:

```bash
python3 data/inputs/epp_emergence_analysis_query_openai_events.py --names-file names.txt
```

JSON list:

```bash
python3 data/inputs/epp_emergence_analysis_query_openai_events.py --names-file names.json
```

## Dry run selector check

```bash
python3 data/inputs/epp_emergence_analysis_query_openai_events.py \
  --names "Vibrio cholerae" \
  --dry-run
```

## Check available OpenAI models

If `gpt-5.1` returns `model not found`, list the model IDs available to your API key:

```bash
python3 data/inputs/epp_emergence_analysis_query_openai_events.py --list-models
```

The script defaults to the official OpenAI API base URL:

```text
https://api.openai.com/v1
```

If you intentionally use an OpenAI-compatible gateway, pass it explicitly:

```bash
python3 data/inputs/epp_emergence_analysis_query_openai_events.py \
  --list-models \
  --base-url "https://your-gateway.example/v1"
```

Then rerun with one of those IDs:

```bash
python3 data/inputs/epp_emergence_analysis_query_openai_events.py \
  --names "Orthoflavivirus denguei" \
  --model gpt-5
```

## Convert saved JSON to Excel without another API call

```bash
python3 data/inputs/epp_emergence_analysis_convert_json_to_excel.py \
  data/inputs/openai_outputs/openai_epp_emergence_events_YYYYMMDD_HHMMSS.json
```

Outputs are written to `data/inputs/openai_outputs/`.
