# 10) Classify Articles As Relevant (Containing targeted EPPs)

# Classify each row's `full.text` as RelevantYN (Yes/No) for:
# - Screwworm / screw worm (Cochliomyia hominivorax)
# - Desert locust (Schistocerca gregaria)
# - Dengue virus
# - Ebola virus
# - Zika virus

import os
import json
import time
import pandas as pd
from openai import OpenAI

MODEL       = "gpt-5-nano"
INPUT_FILE  = "Sampled500ForNanoClassification.csv"        
OUTPUT_FILE = "500SampleNanoClass.csv"           
BATCH_SIZE  = 1                   
SLEEP_SEC   = 0.2                   

SYSTEM_PROMPT = """You are a strict binary classifier for article relevance.

Task: Fill a column 'RelevantYN'.
For each article in the 'full.text' field, label as:
- Yes = The article contains information about one or more of the following emerging pests and pathogens (EPPs):
  • Screwworm / screw worm (Cochliomyia hominivorax)
  • Desert locust (Schistocerca gregaria)
  • Dengue virus
  • Ebola virus
  • Zika virus
- No = The article does not contain information about these EPPs.

Important rules:
1) Focus on the main body text.
   - Headlines or link lists at the bottom that mention EPPs are NOT part of the main article → code No.

Output format:
- For each text, return only a single field 'relevant' with value "Yes" or "No".
- Do not output anything else.
"""  # CHANGED: REMOVED the line 'If uncertain after best effort, default to "Yes".'

def _schema(n: int) -> dict:
    # ADDED: factored schema to a function (no behavior change)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "rows": {
                "type": "array",
                "minItems": n,
                "maxItems": n,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "relevant": {"type": "string", "enum": ["Yes", "No"]}
                    },
                    "required": ["relevant"]
                }
            }
        },
        "required": ["rows"]
    }

def _validate_labels(raw: dict, n: int) -> List[str]:
    """Validate and return exactly n labels, each 'Yes' or 'No'; else raise."""
    # ADDED: strict validator (replaces all defaults/padding)
    if not isinstance(raw, dict) or "rows" not in raw or not isinstance(raw["rows"], list):
        raise ValueError("Missing or invalid 'rows' array in model output.")
    rows = raw["rows"]
    if len(rows) != n:
        raise ValueError(f"Expected {n} rows, got {len(rows)}.")
    labels = []
    for i, r in enumerate(rows, start=1):
        if not isinstance(r, dict) or "relevant" not in r:
            raise ValueError(f"Row {i} missing 'relevant' field.")
        v = r["relevant"]
        if v not in ("Yes", "No"):
            raise ValueError(f"Row {i} has invalid value '{v}'.")
        labels.append(v)
    return labels

def classify_batch(client: OpenAI, texts: List[str]) -> List[str]:
    """Single try. No defaults. Raises on any invalid/malformed response."""
    n = len(texts)
    schema = _schema(n)

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":
            "Classify the following texts. For each, return an object with a single field 'relevant'.\n\n" +
            "\n\n".join([f"Text {i+1}: {t}" for i, t in enumerate(texts, start=1)])
        },
    ]

    resp = client.responses.create(
        model=MODEL,
        input=msgs,
        text={
            "format": {
                "type": "json_schema",
                "name": "relevance",
                "schema": schema,
                "strict": True
            }
        },
    )

    # ORIGINAL:
    # data = json.loads(resp.output_text or "{}")
    # labels = [row.get("relevant", "Yes") for row in data.get("rows", [])]
    # if len(labels) < n: labels += ["Yes"] * (n - len(labels))
    # elif len(labels) > n: labels = labels[:n]
    # labels = ["Yes" if x not in ("Yes", "No") else x for x in labels]
    # return labels

    # CHANGED: strict parsing + validation; no defaults, no padding, no coercion
    raw = json.loads(resp.output_text or "{}")
    return _validate_labels(raw, n)

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")
    client = OpenAI(api_key=api_key)

    df = pd.read_csv(INPUT_FILE)
    if "full.text" not in df.columns:
        raise ValueError("Input CSV must contain a 'full.text' column.")

    texts = df["full.text"].astype(str).tolist()
    results: List[str] = []

    total = len(texts)
    print(f"Loaded {total} rows from {INPUT_FILE}")

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = texts[start:end]
        print(f"Processing {start+1}–{end} / {total}")
        # ORIGINAL wrapped in try/except with default "Yes" fallback:
        # try:
        #     labels = classify_batch(client, batch)
        # except Exception as e:
        #     print(f"Batch {start//BATCH_SIZE + 1} error: {e}")
        #     labels = ["Yes"] * len(batch)  # fallback
        #
        # if len(labels) != len(batch):
        #     print(...)
        #     labels = (labels + ["Yes"] * len(batch))[:len(batch)]

        # CHANGED: one call, no retries, no fallback—fail fast on error
        labels = classify_batch(client, batch)  # raises if invalid
        if len(labels) != len(batch):
            # Should never happen thanks to strict validation
            raise RuntimeError(f"Got {len(labels)} labels for a {len(batch)}-item batch.")

        results.extend(labels)
        time.sleep(SLEEP_SEC)

    df["RelevantYN"] = results
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved → {OUTPUT_FILE} (rows: {len(df)})")

if __name__ == "__main__":
    main()
