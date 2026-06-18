# EPP emergence-event LLM scripts

These scripts sit beside `EPPs input file.xlsx` and read:

- `Main list`
- `Country`

They load API keys from the project-root `.env` file.

```dotenv
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

By default, both roles use OpenAI with `gpt-5.1`. You can opt in by role:

- Search prompt provider: `--search-provider openai|claude`
- Verification prompt provider: `--review-provider openai|claude`
- Both roles at once: `--provider openai|claude`

Both execution modes use the same scientific workflow:

1. Search each selected EPP independently using `ORIGINAL_USER_PROMPT`.
2. Save intermediate search JSON named by standardized scientific name.
3. Peer-review that JSON with `VERIFICATION_USER_PROMPT`.
4. Save reviewed JSON named by standardized scientific name.
5. Write one `curated_batch.json` and `curated_batch.xlsx` per batch.

There are two ways to run the code:

1. **Run directly, not using OpenAI Batch API:** uses
   `epp_emergence_analysis_query_openai_events.py`. This calls the selected
   provider APIs from your machine and can mix OpenAI and Claude by role.
2. **Run through OpenAI Batch API:** uses
   `epp_emergence_analysis_openai_batch_api.py`. This submits asynchronous
   OpenAI batch jobs. The current Batch API workflow is OpenAI-only.

The API instruction prompt is stored verbatim in `epp_emergence_analysis_prompt.py`. Runtime row and `Country` sheet data are added as attached Excel context.

If imports are missing in a fresh environment, install the project dependencies or run:

```bash
python3 -m pip install openai python-dotenv pandas openpyxl requests
```

If macOS refuses to install into the system Python, create a virtual
environment and use that interpreter:

```bash
python3 -m venv .venv-epp
. .venv-epp/bin/activate
python -m pip install openai python-dotenv pandas openpyxl requests
```

Then either keep the virtual environment active, or replace `python3` in the
examples below with the virtual-environment interpreter. For example, if you
are using the temporary environment created earlier in this project:

```bash
PYTHON="/tmp/influx-epp-venv/bin/python"
```

## Way 1: Run directly, not using Batch API

Use this mode for quick experiments, immediate responses, resume/retry behavior
from local JSON files, or mixed-provider runs such as OpenAI search plus Claude
review.

Important distinction: `--batch-size` in this mode only controls how outputs
are grouped into local folders and Excel files. It is not the OpenAI Batch API.

In direct mode, each selected EPP is processed separately:

- 1 search API request for the EPP
- 1 verification API request for the EPP

### Run by standardized scientific names

```bash
python3 data/inputs/epp_emergence_analysis_query_openai_events.py \
  --names "Vibrio cholerae" "Xylella fastidiosa"
```

Outputs are written in batch folders. The default output folder name is kept for backward compatibility:

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

Use OpenAI for the search prompt and Claude for the verification prompt:

```bash
python3 data/inputs/epp_emergence_analysis_query_openai_events.py \
  --names "Orthoflavivirus denguei" \
  --search-provider openai \
  --review-provider claude \
  --model gpt-5.1 \
  --review-model claude-opus-4-8
```

Use Claude for both prompts:

```bash
python3 data/inputs/epp_emergence_analysis_query_openai_events.py \
  --names "Orthoflavivirus denguei" \
  --provider claude \
  --model claude-sonnet-4-6 \
  --review-model claude-opus-4-8
```

Run all rows in batches with OpenAI search and Claude verification:

```bash
python3 data/inputs/epp_emergence_analysis_query_openai_events.py \
  --all \
  --batch-size 30 \
  --search-provider openai \
  --review-provider claude \
  --model gpt-5.1 \
  --review-model claude-opus-4-8
```

If a row has many events and the script reports incomplete JSON, rerun with a larger output cap:

```bash
python3 data/inputs/epp_emergence_analysis_query_openai_events.py \
  --names "Orthoflavivirus denguei" \
  --model gpt-5.1 \
  --max-output-tokens 50000 \
  --review-max-output-tokens 50000
```

## Way 2: Run through OpenAI Batch API

Use this mode for large OpenAI-only runs that do not need immediate responses.
The script uses the OpenAI Batch API with the `/v1/responses` endpoint.

This workflow sends one async OpenAI batch for the search prompt, then a second
async OpenAI batch for the verification prompt after the search JSON has been
downloaded. The review batch cannot be created before search completes because
the verification prompt includes the search JSON.

### How EPPs count as Batch API requests

Yes: in this code, each selected EPP counts as 1 request in the search batch.
The same EPP then counts as 1 request in the review batch.

Examples:

- 7 EPPs = 7 requests in `search_requests.jsonl`, then 7 requests in
  `review_requests.jsonl`.
- 36 EPPs = 36 search requests, then 36 review requests.
- Total OpenAI requests across the full search+review workflow = selected EPP
  count x 2.

The OpenAI per-batch request limit applies to each submitted batch file. For
example, 36 EPPs is far below the 50,000-request limit for the search batch and
also far below the 50,000-request limit for the review batch.

### Batch API rate limits

OpenAI documents separate Batch API limits:

- A single batch can include up to 50,000 requests.
- A batch input file can be up to 200 MB.
- Each model has an enqueued prompt-token limit for batch processing. Check
  your organization limit at
  `https://platform.openai.com/settings/organization/limits`.
- You can create up to 2,000 batches per hour.
- Output tokens are not currently limited by the Batch API rate-limit pool, but
  `--max-output-tokens` and `--review-max-output-tokens` still cap each model
  response.

Because this script submits search first and review second, you normally enqueue
only one prompt batch at a time for a given model. If you manually submit
multiple pending batches for the same model, their prompt tokens can count
together against that model's enqueued prompt-token limit.

OpenAI guide:
`https://developers.openai.com/api/docs/guides/batch`

### Prepare request JSONL

This validates selected rows and writes the Batch API JSONL file without
submitting a paid job. This step does not call OpenAI, but later `submit-*` and
`poll-*` steps require the `openai` Python package.

```bash
RUN_DIR="data/inputs/openai_outputs/epp_openai_batch_$(date -u +%Y%m%d_%H%M%S)"
PYTHON="/tmp/influx-epp-venv/bin/python"

"$PYTHON" data/inputs/epp_emergence_analysis_openai_batch_api.py prepare-search \
  --run-dir "$RUN_DIR" \
  --input-rows 6 482 19 4 497 206 9 \
  --model gpt-5.1 \
  --review-model gpt-5.1 \
  --max-output-tokens 50000 \
  --review-max-output-tokens 50000 \
  --search-context-size high
```

### Submit the search batch

```bash
PYTHON="/tmp/influx-epp-venv/bin/python"

"$PYTHON" data/inputs/epp_emergence_analysis_openai_batch_api.py submit-search \
  --run-dir "$RUN_DIR" \
  --input-rows 6 482 19 4 497 206 9 \
  --model gpt-5.1 \
  --review-model gpt-5.1 \
  --max-output-tokens 50000 \
  --review-max-output-tokens 50000 \
  --search-context-size high
```

### Poll the search batch

Poll until the search batch status is `completed`. When completed, this also
downloads per-EPP search JSON files:

```bash
PYTHON="/tmp/influx-epp-venv/bin/python"
RUN_DIR="data/inputs/openai_outputs/epp_openai_batch_20260618_141226"
"$PYTHON" data/inputs/epp_emergence_analysis_openai_batch_api.py poll-search \
  --run-dir "$RUN_DIR"
```

### Submit and poll the verification batch

```bash
"$PYTHON" data/inputs/epp_emergence_analysis_openai_batch_api.py submit-review \
  --run-dir "$RUN_DIR" \
  --review-model gpt-5.1 \
  --review-max-output-tokens 50000 \
  --search-context-size high

"$PYTHON" data/inputs/epp_emergence_analysis_openai_batch_api.py poll-review \
  --run-dir "$RUN_DIR"
```

### Finalize JSON and Excel

After the review batch completes, write curated JSON and Excel outputs:

```bash
"$PYTHON" data/inputs/epp_emergence_analysis_openai_batch_api.py finalize \
  --run-dir "$RUN_DIR" \
  --batch-size 30
```

OpenAI Batch API runs are written under:

```text
$RUN_DIR/
├── batch_manifest.json
├── openai_batch/
│   ├── search_requests.jsonl
│   ├── search_batch.json
│   ├── review_requests.jsonl
│   └── review_batch.json
├── batch_search_json/
├── batch_reviewed_json/
└── batch_001/
    ├── curated_batch.json
    └── curated_batch.xlsx
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

## Check available models

If `gpt-5.1` returns `model not found`, list the model IDs available to your API key:

```bash
python3 data/inputs/epp_emergence_analysis_query_openai_events.py --list-models
```

For Claude:

```bash
python3 data/inputs/epp_emergence_analysis_query_openai_events.py \
  --list-models \
  --list-model-provider claude
```

The script defaults to the official OpenAI API base URL:

```text
https://api.openai.com/v1
```

If you intentionally use an OpenAI-compatible gateway, pass it explicitly:

```bash
python3 data/inputs/epp_emergence_analysis_query_openai_events.py \
  --list-models \
  --openai-base-url "https://your-gateway.example/v1"
```

For a Claude-compatible gateway, pass `--claude-base-url`.

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

Outputs are written to `data/inputs/openai_outputs/` unless you pass `--output-dir`.
