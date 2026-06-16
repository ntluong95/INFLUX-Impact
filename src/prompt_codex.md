You are an expert Python ML/data engineer. I’m on a MacBook Pro (Apple silicon) with 16GB RAM. I have an OpenAI API key stored in a local .env file (do NOT hardcode secrets). Use OpenAI’s API as the default inference backend for all LLM calls.

I already collected Google News RSS results for “zika” in 2015 (Brazil Portuguese edition) into:

- Input: data/Zika_News_2015.csv (repo root relative)

Implement Step 2: Data Processing + Classification as a reproducible pipeline.
All new code must be created inside:

- src/classification/

GOALS

(1) Inspect the CSV

- Load data/Zika_News_2015.csv with pandas
- Print columns, row count, null rates, sample rows
- Detect URL column robustly (link/item_link/url/etc.) and identify RSS title/description/date columns (handle unknown schema).

(2) URL filtering + de-duplication

- Implement URL canonicalization:
  - lowercase scheme/host
  - remove fragments
  - remove tracking query params (utm\_\*, gclid, fbclid, mc_cid, mc_eid, etc.)
  - normalize trailing slashes
  - sort query params consistently
- Remove duplicates by canonical URL
- Maintain persistent URL registry using SQLite:
  - data/url_registry.sqlite
  - store canonical_url, first_seen, last_seen, original_url, source (rss), and a content_hash if available
- Output deduped feed rows:
  - data/processed/zika_2015_unique.csv

(3) HTML fetch + Text extraction (BeautifulSoup required)
For each unique URL:

- Fetch HTML with polite crawling:
  - randomized delay 1–3s, retries with exponential backoff, timeouts
  - set user-agent
- Save raw HTML for reproducibility:
  - data/raw_html/<sha256(canonical_url)>.html
- Extract main title + main body text:
  - Use BeautifulSoup as baseline
  - Strip scripts/styles/nav/footer/aside/ads and other boilerplate
  - Implement a heuristic “main content” finder (largest text-dense container, paragraph density, etc.)
  - Optional fallback if installed: readability-lxml or trafilatura, but pipeline must work with BeautifulSoup only.
- Output extracted dataset:
  - data/processed/articles_extracted.parquet (preferred) or CSV
  - columns: canonical_url, original_url, rss_title, rss_description, pub_date,
    extracted_title, extracted_text,
    fetch_status, http_status, fetched_at, extraction_method

(4) Ensemble classification: relevance to “cascading impacts of Zika”
Task: classify if each article is relevant to describing cascading impacts (beyond simple case counts), e.g. knock-on effects on health systems, pregnancy/microcephaly social impacts, travel advisories, economy, vector control policy, public fear, political response, inequalities, education, supply chains. Articles that merely mention Zika without downstream consequences should usually be NOT relevant.

Use ensemble learning of 3 models (via OpenAI API backend):

1. Mistral-7B-OpenOrca
2. Zephyr-7B-Beta
3. Meta-Llama-3-70B-Instruct

IMPORTANT:

- Use OpenAI’s API as the default backend, loaded from .env (e.g., OPENAI_API_KEY).
- Implement a single “OpenAI-compatible chat completions” client wrapper that can call these models by name (model IDs must be configurable).
- If the OpenAI API does not provide those exact models, design the code so model names are configurable in config.yaml and can be swapped without code changes.
- Do not assume local inference.

Model prompting requirements:

- Enforce STRICT JSON output per model:
  {"relevant": true/false, "confidence": 0.0-1.0, "rationale": "...", "evidence_spans": ["..."]}
- Add robust JSON parsing/repair (handle minor formatting issues safely).
- Input text length controls:
  - Use extracted_title + first N chars/tokens of extracted_text (configurable)
  - Optionally implement deterministic extractive summarization before LLM calls to reduce tokens/cost.
- Ensemble rule:
  - Majority vote on relevant
  - Tie-break by highest average confidence
  - Store per-model predictions + final decision

Output:

- data/processed/articles_classified.csv containing:
  canonical_url, extracted_title, text_hash, per-model relevant/confidence/rationale,
  final_label, final_confidence, classified_at

(5) Project hygiene + reproducibility

- Put everything in src/classification/
- Provide single entrypoint:
  - src/classification/run_pipeline.py
  - CLI: --input data/Zika_News_2015.csv --outdir data/processed --config src/classification/config.yaml
- Logging to: data/logs/classification_pipeline.log
- Restartable pipeline:
  - skip fetch if HTML exists unless --force
  - skip extraction/classification if outputs exist unless --force
- Add:
  - src/classification/README.md (setup, how to use .env, how to set model names/endpoints)
  - src/classification/requirements.txt
  - src/classification/config.yaml template
  - .env template (example only) listing OPENAI_API_KEY (do not commit secrets)

DELIVERABLES
Create the complete folder src/classification/ with:

- run_pipeline.py
- inspect_csv.py (or integrated)
- deduplicate_urls.py
- fetch_html.py
- extract_text.py
- llm_backends.py (OpenAI API client wrapper)
- classify_ensemble.py
- utils.py (url canonicalization, hashing, retry, json parsing/repair)
- config.yaml (template)
- README.md
- requirements.txt

Start by inspecting the CSV schema and adapt to it without changing the input file.
