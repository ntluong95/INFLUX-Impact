# classify_domains.py — GPT-5-Nano via OpenAI Responses API
# - Batches only (uses Structured Outputs JSON Schema)
# - Quality guard: auto-split and retry if a batch tail looks degenerate
# - Prints "processed N links" every 10 rows
# - Exact 5-column CSV with your header; preserves order & duplicates

import os
import sys
import json
import time
import argparse
from pathlib import Path
from openai import OpenAI

HEADER = "Domain,News outlet (Yes/No),Short description,Type of content,Confidence"
TYPE_OPTIONS = [
    "General content",
    "Business/Finance",
    "Technology",
    "Health/Science",
    "Sports",
    "Entertainment (music/film/TV/gaming/celebrity)",
    "Lifestyle/Travel",
    "Corporate/Press releases",
    "Trade/Industry",
    "Other",
]

DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_INPUT = "Unique_Domains_All.csv"
DEFAULT_OUTPUT = "Dom_All_GPT_Class_1.csv"
DEFAULT_BATCH_SIZE = 1

# progress counter
PROCESSED_COUNT = 0
PROGRESS_EVERY = 50  # change to 0 to disable

SYSTEM_INSTRUCTIONS = """JSON-only domain classifier. TOP-LEVEL OUTPUT MUST BE AN OBJECT with a key `rows`.
`rows` must be an array with EXACTLY one item per input domain, in the SAME ORDER. Do not skip any.
Each item must be an object with fields:
- domain (string): echo the input domain EXACTLY as given.
- news_outlet (string): "Yes" or "No".
- short_description (string): ≤5 words; avoid commas (use hyphens if needed).
- type_of_content (string): choose exactly one of:
  General content; Business/Finance; Technology;
  Health/Science; Sports; Entertainment (music/film/TV/gaming/celebrity);
  Lifestyle/Travel; Corporate/Press releases; Trade/Industry; Other.
  Branded topical subdomains count as that topic (sports.yahoo.com → Sports); paths do not.
- confidence (number): 0–1 model certainty that “News outlet” is correct.

Rules:
- News outlet: Yes if primarily publishes/broadcasts news; No otherwise.
  Yes examples: national daily newspapers (e.g., manilatimes.net, usatoday.com), local news broadcasters (e.g., fox5atlanta.com, hindustantimes.com).
  No examples: government websites (e.g., nih.gov, blogs.cdc.gov), intergovernmental organization websites (e.g., who.int, paho.org), research institute websites (e.g., pewresearch.org, pasteur.fr), university websites (e.g., publichealth.jhu.edu,
           urmc.rochester.edu), scientific journal websites (e.g., nature.com, bmcinfectdis.biomedcentral.com), magazine websites (e.g., scientificamerican.com, wanderlustmagazine.com),
           tabloid websites (e.g., tmz.com, dailymail.co.uk).
- Output MUST strictly conform to the JSON Schema provided by the API.
- No explanations or extra text—only the required JSON object.
"""


def parse_args():
    p = argparse.ArgumentParser(
        description="Classify domains with GPT-5-Nano via OpenAI Responses API"
    )
    p.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Input file (one URL/domain per line or CSV; first column used)",
    )
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path")
    p.add_argument("--limit", type=int, default=None, help="Only process first N rows")
    p.add_argument(
        "--model", default=DEFAULT_MODEL, help="OpenAI model name (default: gpt-5-nano)"
    )
    p.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size"
    )
    p.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Optional cap on generated tokens (Responses API)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Write raw JSON outputs next to the CSV for inspection",
    )
    return p.parse_args()


def read_first_column_lines(path):
    # utf-8-sig will strip a potential BOM (prevents Ôªø… issues)
    raw = Path(path).read_text(encoding="utf-8-sig").splitlines()
    out = []
    for i, ln in enumerate(raw):
        if not ln.strip():
            continue
        first = ln.split(",", 1)[0].strip().lstrip("\ufeff")
        if i == 0 and first.lower() in {"domain", "url", "links"}:
            continue
        out.append(first)
    return out


def make_schema(n_items: int):
    item_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "domain": {"type": "string"},
            "news_outlet": {"type": "string", "enum": ["Yes", "No"]},
            "short_description": {"type": "string"},
            "type_of_content": {"type": "string", "enum": TYPE_OPTIONS},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": [
            "domain",
            "news_outlet",
            "short_description",
            "type_of_content",
            "confidence",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "rows": {
                "type": "array",
                "minItems": n_items,
                "maxItems": n_items,
                "items": item_schema,
            }
        },
        "required": ["rows"],
    }


def classify_batch(
    client: OpenAI,
    model: str,
    domains: list[str],
    max_output_tokens: int | None,
    debug_path: Path | None,
):
    schema = make_schema(len(domains))
    msgs = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": "Domains (one per line):\n" + "\n".join(domains)},
    ]
    kwargs = dict(
        model=model,
        input=msgs,
        # Structured Outputs via JSON Schema (root must be OBJECT)
        text={
            "format": {
                "type": "json_schema",
                "name": "domain_classifications",
                "schema": schema,
                "strict": True,
            }
        },
    )
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens

    resp = client.responses.create(**kwargs)
    txt = resp.output_text or "{}"
    if debug_path:
        debug_path.write_text(txt, encoding="utf-8")

    data = json.loads(txt)
    items = data.get("rows", [])  # top-level object -> rows array

    cleaned = []
    for i, item in enumerate(items):
        dom = domains[i]  # preserve order & exact echo
        news = item.get("news_outlet", "Yes")
        if isinstance(news, bool):
            news = "Yes" if news else "No"
        short = (item.get("short_description") or "").replace(",", "-").strip()
        if short:
            words = short.split()
            if len(words) > 5:
                short = " ".join(words[:5])
        typ = item.get("type_of_content", "Other")
        conf = item.get("confidence", 0.5)
        try:
            conf = float(conf)
        except Exception:
            conf = 0.5
        conf = max(0.0, min(1.0, conf))

        cleaned.append(
            {
                "Domain": dom,
                "News outlet (Yes/No)": news,
                "Short description": short,
                "Type of content": typ,
                "Confidence": conf,
            }
        )
    return cleaned


# ---- progress printing every 10 rows ----
def write_rows(out_path: Path, rows: list[dict], write_header_if_new=True):
    global PROCESSED_COUNT
    file_exists = out_path.exists()
    if not file_exists and write_header_if_new:
        out_path.write_text(HEADER + "\n", encoding="utf-8")
    with open(out_path, "a", encoding="utf-8", newline="") as f:
        for r in rows:
            f.write(
                "{Domain},{News outlet (Yes/No)},{Short description},{Type of content},{Confidence}\n".format(
                    **r
                )
            )
            PROCESSED_COUNT += 1
            if PROGRESS_EVERY and PROCESSED_COUNT % PROGRESS_EVERY == 0:
                print(f"processed {PROCESSED_COUNT} links", flush=True)


# ---- simple quality guard for degenerate tails ----
def _degenerate_tail(rows: list[dict]) -> bool:
    """
    Returns True if the last up-to-100 rows look degenerate:
    - many zeros in Confidence, OR
    - many blank short descriptions AND mostly 'Other'
    """
    if not rows:
        return True
    tail = rows[-min(100, len(rows)) :]
    n = len(tail)
    zeros = sum(1 for r in tail if float(r.get("Confidence", 0)) == 0.0)
    blanks = sum(1 for r in tail if not str(r.get("Short description", "")).strip())
    others = sum(1 for r in tail if r.get("Type of content") == "Other")
    return (zeros / n >= 0.20) or ((blanks / n >= 0.20) and (others / n >= 0.80))


def _process_batch_with_quality(client, args, batch_domains, out_path, label="batch"):
    try:
        dbg = out_path.with_suffix(f".{label}.json") if args.debug else None
        rows = classify_batch(
            client, args.model, batch_domains, args.max_output_tokens, dbg
        )
    except Exception as e:
        if len(batch_domains) <= 1:
            print(f"{label}: giving up on single item due to error: {e}", flush=True)
            return 0
        mid = len(batch_domains) // 2
        print(
            f"{label}: error: {e} → retrying as {len(batch_domains[:mid])}+{len(batch_domains[mid:])}",
            flush=True,
        )
        return _process_batch_with_quality(
            client, args, batch_domains[:mid], out_path, label=f"{label}L"
        ) + _process_batch_with_quality(
            client, args, batch_domains[mid:], out_path, label=f"{label}R"
        )

    # Quality check (skip for tiny batches)
    if _degenerate_tail(rows) and len(batch_domains) > 50:
        mid = len(batch_domains) // 2
        print(
            f"{label}: quality drop detected → retrying as {mid}+{len(batch_domains) - mid}",
            flush=True,
        )
        return _process_batch_with_quality(
            client, args, batch_domains[:mid], out_path, label=f"{label}L"
        ) + _process_batch_with_quality(
            client, args, batch_domains[mid:], out_path, label=f"{label}R"
        )

    write_rows(out_path, rows, write_header_if_new=False)
    return len(rows)


# ---- batches-only runner ----
def run_batches_only(client, args, domains):
    out_path = Path(args.output)
    out_path.write_text(HEADER + "\n", encoding="utf-8")  # fresh file

    total = len(domains)
    batch_size = args.batch_size
    i = 0
    while i < total:
        batch = domains[i : i + batch_size]
        label = f"{i + 1}-{i + len(batch)}/{total}"
        print(f"Processing batch {label} (size={len(batch)})", flush=True)
        wrote = _process_batch_with_quality(
            client, args, batch, out_path, label=f"batch_{i // batch_size + 1}"
        )
        print(f"Finished batch {label} → wrote {wrote} rows", flush=True)
        i += len(batch)  # advance by planned batch size
        time.sleep(0.05)
    print("All batches done.", flush=True)


def main():
    args = parse_args()
    if not Path(args.input).exists():
        print(f"Missing {args.input}")
        sys.exit(1)

    domains = read_first_column_lines(args.input)
    if args.limit is not None:
        domains = domains[: args.limit]

    total = len(domains)
    if total == 0:
        print("No domains to process.")
        return

    print(f"Loaded {total} lines (order preserved; duplicates kept).")
    print(f"Model: {args.model}")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))  # use standard env var

    run_batches_only(client, args, domains)
    print("Done.")


if __name__ == "__main__":
    main()
