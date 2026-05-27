import argparse
import json
import os
import random
import re
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from openai import OpenAI

MODEL = "gpt-5-nano"
INPUT_FILE = "25_Sampled_GPT_Trial.csv"
OUT_JSON_RECORDS = "25_Sampled_GPT_Trial_Impacts_Records.json"
OUT_JSON_NESTED = "25_Sampled_GPT_Trial_Impacts_Nested.json"
OUT_CSV = "25_Sampled_GPT_Trial_Impacts_Extracted_V5.csv"
CATS_JSON = "Impact_Categories_V5.json"
BATCH_SIZE = 1
SLEEP_SEC = 0.2
MAX_UIGS = 25

ORIGINAL_COLS = [
    "feed_title",
    "feed_link",
    "feed_description",
    "feed_language",
    "url",
    "item_description",
    "item_title",
    "real.url",
    "domain",
    "publication.date",
    "full.text",
    "Domain",
    "NewsYN",
    "RelevantYN",
    "species",
]

NEW_COLS = [
    "article_id",
    "observed_impact_text",
    "cause_of_impact_text",
    "location_of_impact_text",
    "aggregated_location",
    "time_of_impact_text",
    "category_of_impact",
]

AGG_LOC_ENUM = ["GADM0", "GADM1", "GADM2", "GADM3", "GADM4", "GADM5", "Continent", "Globe"]

SYSTEM_DISCOVER = """You are a research scientist extracting observed impacts of emerging pests and pathogens (EPPs) from news articles.

Scope: for each article, the EPP species is provided from the dataset's species column. Extract only impacts that the article attributes to that species.

Rules:
- Every impact must be attributable to the provided species for that article. Do not attribute to other agents.
- If the species is missing, return impacts: [] for that article.
- category_of_impact_free should be a compact reusable label. Reuse the same label when the same impact type appears again.
- Keep the total conceptual category set compact; avoid near-duplicate labels.
- The cause_of_impact_text is the underlying driver or mechanism. It must not be identical to observed_impact_text. If absent, return "N/A".
- Use only information present in the text. Do not infer beyond the article.
- Ignore prescriptive or normative statements.
- If a field is missing, return "N/A" exactly.
- Aggregated Location of Impact must be exactly one of: GADM0, GADM1, GADM2, GADM3, GADM4, GADM5, Continent, Globe, or "N/A".

For each article return:
- article_id (integer; index provided)
- impacts: array of items, one per distinct observed impact with fields:
  * category_of_impact_free
  * observed_impact_text
  * cause_of_impact_text
  * location_of_impact_text
  * aggregated_location
  * time_of_impact_text
"""

SYSTEM_APPLY_TEMPLATE = """You are a research scientist extracting observed impacts of emerging pests and pathogens (EPPs) from news articles.

Scope: for each article, the EPP species is provided from the dataset's species column. Extract only impacts that the article attributes to that species.

Rules:
- Every impact must be attributable to the provided species for that article. Do not attribute to other agents.
- If the species is missing, return impacts: [] for that article.
- The cause_of_impact_text is the underlying driver or mechanism described for that impact; it must not be identical to observed_impact_text. If absent, return "N/A".
- Use only information in the text; do not infer beyond what is stated.
- Ignore prescriptive or normative statements.
- If a field is missing, return "N/A" exactly.
- Aggregated Location of Impact must be exactly one of: GADM0, GADM1, GADM2, GADM3, GADM4, GADM5, Continent, Globe, or "N/A".

Use this fixed category set for category_of_impact (choose exactly one of the provided labels when an impact is present):
{category_list}

For each article return:
- article_id (integer; index provided)
- impacts: array of items, one per distinct observed impact with fields:
  * category_of_impact
  * observed_impact_text
  * cause_of_impact_text
  * location_of_impact_text
  * aggregated_location
  * time_of_impact_text
"""


def normalize_string(value: Any) -> str:
    text = value if isinstance(value, str) else ""
    text = text.strip()
    return text if text else "N/A"


def _canon(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'[\s"“”‘’\'`~.,;:!?()\[\]{}<>-]+', " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _flatten_nested_rows(all_rows: List[dict]) -> pd.DataFrame:
    rows = []
    for article in all_rows:
        article_id = article.get("article_id")
        impacts = article.get("impacts", []) or []
        if impacts:
            for item in impacts:
                observed = normalize_string(item.get("observed_impact_text"))
                cause = normalize_string(item.get("cause_of_impact_text"))
                if _canon(observed) and _canon(observed) == _canon(cause):
                    cause = "N/A"
                rows.append(
                    {
                        "article_id": article_id,
                        "category_of_impact": (
                            normalize_string(item["category_of_impact"])
                            if "category_of_impact" in item
                            else normalize_string(item.get("category_of_impact_free"))
                        ),
                        "observed_impact_text": observed,
                        "cause_of_impact_text": cause,
                        "location_of_impact_text": normalize_string(item.get("location_of_impact_text")),
                        "aggregated_location": normalize_string(item.get("aggregated_location")),
                        "time_of_impact_text": normalize_string(item.get("time_of_impact_text")),
                    }
                )
        else:
            rows.append(
                {
                    "article_id": article_id,
                    "category_of_impact": "N/A",
                    "observed_impact_text": "N/A",
                    "cause_of_impact_text": "N/A",
                    "location_of_impact_text": "N/A",
                    "aggregated_location": "N/A",
                    "time_of_impact_text": "N/A",
                }
            )
    return pd.DataFrame(rows)


def _make_schema(batch_size: int, mode: str, enum_categories: Optional[List[str]]) -> dict:
    impact_props_common = {
        "observed_impact_text": {"type": "string"},
        "cause_of_impact_text": {"type": "string"},
        "location_of_impact_text": {"type": "string"},
        "aggregated_location": {"type": "string", "enum": AGG_LOC_ENUM + ["N/A"]},
        "time_of_impact_text": {"type": "string"},
    }
    if mode == "discover":
        impact_item = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"category_of_impact_free": {"type": "string"}, **impact_props_common},
            "required": ["category_of_impact_free", *impact_props_common.keys()],
        }
    else:
        impact_item = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"category_of_impact": {"type": "string", "enum": enum_categories}, **impact_props_common},
            "required": ["category_of_impact", *impact_props_common.keys()],
        }

    article_item = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "article_id": {"type": "integer"},
            "impacts": {"type": "array", "items": impact_item},
        },
        "required": ["article_id", "impacts"],
    }

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "rows": {
                "type": "array",
                "minItems": batch_size,
                "maxItems": batch_size,
                "items": article_item,
            }
        },
        "required": ["rows"],
    }


def _one_species(cell: str) -> str:
    if not isinstance(cell, str):
        return ""
    parts = [part.strip() for part in re.split(r"[;,|/]", cell) if part.strip()]
    return parts[0] if parts else ""


def _build_messages(texts: List[str], species_list: List[str], start_index: int, system_prompt: str) -> list:
    numbered = []
    for offset, (text, species) in enumerate(zip(texts, species_list), start=1):
        species_clean = _one_species(species)
        if species_clean:
            header = f"Article {start_index + offset}: [Species: {species_clean}]"
        else:
            header = f"Article {start_index + offset}: [Species: (missing) -> return impacts: []]"
        numbered.append(f"{header}\n{text}")
    user = (
        "Extract observed impacts for each article separately. "
        "Only return impacts attributable to the provided species for that article. "
        "Return a JSON object with key 'rows'; each element must correspond to the article at the same position and include 'article_id' and 'impacts'.\n\n"
        + "\n\n".join(numbered)
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]


def _call_batch(
    client: OpenAI,
    model: str,
    texts: List[str],
    species_batch: List[str],
    start_index: int,
    mode: str,
    enum_categories: Optional[List[str]],
) -> dict:
    schema = _make_schema(len(texts), mode=mode, enum_categories=enum_categories)
    system = SYSTEM_DISCOVER if mode == "discover" else SYSTEM_APPLY_TEMPLATE.format(
        category_list=", ".join(f'"{cat}"' for cat in (enum_categories or []))
    )
    messages = _build_messages(texts, species_batch, start_index, system)
    response = client.responses.create(
        model=model,
        input=messages,
        text={"format": {"type": "json_schema", "name": f"impact_{mode}", "schema": schema, "strict": True}},
    )
    data = json.loads(response.output_text or "{}")
    rows = data.get("rows", [])
    if len(rows) < len(texts):
        for _ in range(len(texts) - len(rows)):
            rows.append({"article_id": start_index + len(rows) + 1, "impacts": []})
    elif len(rows) > len(texts):
        rows = rows[: len(texts)]

    for index, row in enumerate(rows):
        if not isinstance(row.get("article_id"), int):
            row["article_id"] = start_index + index + 1
        if not isinstance(row.get("impacts"), list):
            row["impacts"] = []
    return {"rows": rows}


def _dedupe_categories_from_discovery(all_rows: List[dict], min_len: int = 1) -> List[str]:
    categories = []
    for row in all_rows:
        for item in row.get("impacts", []) or []:
            category = normalize_string(item.get("category_of_impact_free"))
            if category != "N/A" and len(category) >= min_len:
                categories.append(category)
    grouped = {}
    for category in categories:
        key = category.lower()
        grouped.setdefault(key, []).append(category)
    return sorted({min(values, key=len) for values in grouped.values()})


def _ensure_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = "N/A"
    return df[columns]


def _make_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    base_url = os.environ.get("OPENAI_BASE_URL") or None
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _load_dataframe(path: str) -> pd.DataFrame:
    input_path = Path(path)
    if input_path.suffix.lower() == ".parquet":
        return pd.read_parquet(input_path)
    return pd.read_csv(input_path)


def _detect_text_column(df: pd.DataFrame) -> str:
    for candidate in ["full.text", "extracted_text", "full_text", "text"]:
        if candidate in df.columns:
            return candidate
    raise ValueError("Input file must contain a text column such as 'full.text' or 'extracted_text'.")


def _detect_species_column(df: pd.DataFrame) -> str:
    for candidate in ["species", "target_species", "epp_species"]:
        if candidate in df.columns:
            return candidate
    raise ValueError("Input file must contain a species column.")


def _detect_relevance_mask(df: pd.DataFrame) -> pd.Series:
    if "final_label" in df.columns:
        return df["final_label"].fillna("").astype(str).str.strip().eq("relevant")
    for candidate in ["RelevantYN", "relevant", "is_relevant"]:
        if candidate in df.columns:
            values = df[candidate].fillna("").astype(str).str.strip().str.lower()
            return values.isin({"1", "true", "yes", "y", "relevant"})
    return pd.Series([True] * len(df), index=df.index)


def _truncate_text(value: Any, max_chars: int) -> str:
    text = str(value or "")
    text = re.sub(r"\s+", " ", text).strip()
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars]
    return text


def _merge_with_base_df(base_df: pd.DataFrame, flat_df: pd.DataFrame) -> pd.DataFrame:
    flat_df = flat_df.copy()
    flat_df["__row_idx"] = flat_df["article_id"] - 1
    base_df = base_df.reset_index(drop=True).copy()
    base_df["__row_idx"] = base_df.index
    merged = base_df.merge(flat_df, on="__row_idx", how="left").drop(columns=["__row_idx"])
    base_columns = ORIGINAL_COLS if set(ORIGINAL_COLS).issubset(set(base_df.columns)) else list(base_df.columns)
    return _ensure_columns(merged, list(dict.fromkeys(base_columns + NEW_COLS)))


def _extract_articles(
    client: OpenAI,
    model: str,
    articles_df: pd.DataFrame,
    text_col: str,
    species_col: str,
    mode: str,
    llm_batch_size: int,
    sleep_sec: float,
    max_text_chars: int,
    enum_categories: Optional[List[str]] = None,
) -> tuple[List[dict], pd.DataFrame]:
    rows = []
    article_uids = articles_df["__article_uid"].tolist()
    texts = [_truncate_text(value, max_text_chars) for value in articles_df[text_col].tolist()]
    species = articles_df[species_col].astype(str).tolist()

    for start in range(0, len(texts), llm_batch_size):
        end = min(start + llm_batch_size, len(texts))
        data = _call_batch(
            client,
            model=model,
            texts=texts[start:end],
            species_batch=species[start:end],
            start_index=start,
            mode=mode,
            enum_categories=enum_categories,
        )
        rows.extend(data["rows"])
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    flat = _flatten_nested_rows(rows)
    uid_map = {index + 1: uid for index, uid in enumerate(article_uids)}
    flat["__article_uid"] = flat["article_id"].map(uid_map)
    return rows, flat


def run_mode(
    mode: str,
    input_file: str,
    sample_size: Optional[int],
    categories_path: Path,
    out_json_records: Path,
    out_json_nested: Path,
    out_csv: Path,
    model: str,
    llm_batch_size: int,
    sleep_sec: float,
    max_text_chars: int,
) -> None:
    client = _make_client()
    df = _load_dataframe(input_file)
    text_col = _detect_text_column(df)
    species_col = _detect_species_column(df)

    base_df = _ensure_columns(df.copy(), ORIGINAL_COLS) if set(ORIGINAL_COLS).issubset(set(df.columns)) else df.copy()
    working_df = df.copy().reset_index(drop=True)
    working_df["__article_uid"] = working_df.index + 1

    if mode == "discover":
        n_rows = min(sample_size or 50, len(working_df))
        subset_df = working_df.head(n_rows).copy()
        nested_rows, flat = _extract_articles(
            client,
            model=model,
            articles_df=subset_df,
            text_col=text_col,
            species_col=species_col,
            mode="discover",
            llm_batch_size=llm_batch_size,
            sleep_sec=sleep_sec,
            max_text_chars=max_text_chars,
        )
        merged = _merge_with_base_df(base_df.head(n_rows), flat)
        categories = _dedupe_categories_from_discovery(nested_rows)
        categories_path.write_text(json.dumps(categories, ensure_ascii=False, indent=2), encoding="utf-8")
        out_json_nested.write_text(json.dumps({"rows": nested_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        merged.to_csv(out_csv, index=False)
        merged.to_json(out_json_records, orient="records", force_ascii=False, indent=2)
        print(f"Wrote nested JSON -> {out_json_nested}")
        print(f"Wrote CSV -> {out_csv}")
        print(f"Wrote records JSON -> {out_json_records}")
        print(f"Wrote discovered categories -> {categories_path}")
        return

    if not categories_path.exists():
        raise FileNotFoundError(f"Missing categories file: {categories_path}")
    enum_categories = json.loads(categories_path.read_text(encoding="utf-8"))
    nested_rows, flat = _extract_articles(
        client,
        model=model,
        articles_df=working_df,
        text_col=text_col,
        species_col=species_col,
        mode="apply",
        llm_batch_size=llm_batch_size,
        sleep_sec=sleep_sec,
        max_text_chars=max_text_chars,
        enum_categories=enum_categories,
    )
    merged = _merge_with_base_df(base_df, flat)
    out_json_nested.write_text(json.dumps({"rows": nested_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    merged.to_csv(out_csv, index=False)
    merged.to_json(out_json_records, orient="records", force_ascii=False, indent=2)
    print(f"Wrote nested JSON -> {out_json_nested}")
    print(f"Wrote CSV -> {out_csv}")
    print(f"Wrote records JSON -> {out_json_records}")


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _canon(text)))


def _uig_similarity(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    left_name = _canon(left.get("name", ""))
    right_name = _canon(right.get("name", ""))
    left_text = _canon(f"{left.get('name', '')} {left.get('definition', '')} {' '.join(left.get('evidence_examples', []))}")
    right_text = _canon(f"{right.get('name', '')} {right.get('definition', '')} {' '.join(right.get('evidence_examples', []))}")

    name_score = SequenceMatcher(None, left_name, right_name).ratio() if left_name and right_name else 0.0
    left_tokens = _tokenize(left_text)
    right_tokens = _tokenize(right_text)
    token_score = len(left_tokens & right_tokens) / len(left_tokens | right_tokens) if left_tokens and right_tokens else 0.0
    body_score = SequenceMatcher(None, left_text, right_text).ratio() if left_text and right_text else 0.0
    return 0.45 * name_score + 0.35 * token_score + 0.20 * body_score


def _build_run_uigs(flat_df: pd.DataFrame, max_uigs: int) -> List[Dict[str, Any]]:
    valid = flat_df[flat_df["category_of_impact"].ne("N/A")].copy()
    if valid.empty:
        return []

    uigs: List[Dict[str, Any]] = []
    for category, group in valid.groupby("category_of_impact"):
        examples = [
            value for value in group["observed_impact_text"].dropna().astype(str).tolist() if value and value != "N/A"
        ]
        causes = [
            value for value in group["cause_of_impact_text"].dropna().astype(str).tolist() if value and value != "N/A"
        ]
        article_uids = sorted({int(uid) for uid in group["__article_uid"].dropna().tolist()})
        uigs.append(
            {
                "name": category,
                "definition": " | ".join(dict.fromkeys(examples[:2] + causes[:1]))[:500],
                "evidence_examples": list(dict.fromkeys(examples[:3])),
                "article_uids": article_uids,
                "count": int(group["__article_uid"].nunique()),
            }
        )
    uigs.sort(key=lambda item: (-item["count"], item["name"].lower()))
    return uigs[:max_uigs]


def _harmonize_uigs(uigs: List[Dict[str, Any]], similarity_threshold: float) -> List[Dict[str, Any]]:
    clusters: List[Dict[str, Any]] = []
    for item in uigs:
        best_index = None
        best_score = 0.0
        for index, cluster in enumerate(clusters):
            score = _uig_similarity(item, cluster["representative"])
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is not None and best_score >= similarity_threshold:
            clusters[best_index]["members"].append(item)
            clusters[best_index]["article_uids"].update(item.get("article_uids", []))
            clusters[best_index]["run_ids"].add(item.get("run_id"))
            clusters[best_index]["batch_ids"].add(item.get("batch_id"))
        else:
            clusters.append(
                {
                    "representative": item,
                    "members": [item],
                    "article_uids": set(item.get("article_uids", [])),
                    "run_ids": {item.get("run_id")},
                    "batch_ids": {item.get("batch_id")},
                }
            )

    summaries: List[Dict[str, Any]] = []
    for cluster_index, cluster in enumerate(clusters, start=1):
        names = [member["name"] for member in cluster["members"] if member.get("name")]
        definitions = [member.get("definition", "") for member in cluster["members"]]
        examples = []
        for member in cluster["members"]:
            examples.extend(member.get("evidence_examples", []))
        canonical_name = Counter(names).most_common(1)[0][0] if names else f"uig_{cluster_index}"
        canonical_definition = max(definitions, key=len) if definitions else canonical_name
        summaries.append(
            {
                "cluster_id": cluster_index,
                "name": canonical_name,
                "definition": canonical_definition,
                "evidence_examples": list(dict.fromkeys(examples))[:3],
                "article_uids": sorted(cluster["article_uids"]),
                "run_ids": sorted(run_id for run_id in cluster["run_ids"] if run_id is not None),
                "batch_ids": sorted(batch_id for batch_id in cluster["batch_ids"] if batch_id is not None),
                "members": cluster["members"],
            }
        )
    return summaries


def _mean_pairwise_jaccard(sets: Sequence[set[str]]) -> Optional[float]:
    if len(sets) < 2:
        return None
    scores = []
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            union = sets[left] | sets[right]
            score = len(sets[left] & sets[right]) / len(union) if union else 1.0
            scores.append(score)
    return round(sum(scores) / len(scores), 4) if scores else None


def _adjudicate_clusters(
    clusters: List[Dict[str, Any]],
    runs_per_batch: int,
    batch_article_count: int,
    support_threshold: float,
    alpha: float,
    max_uigs: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    scored = []
    total_run_uigs = sum(len(cluster["members"]) for cluster in clusters)
    for cluster in clusters:
        support = len(cluster["run_ids"]) / runs_per_batch if runs_per_batch else 0.0
        coverage = len(cluster["article_uids"]) / batch_article_count if batch_article_count else 0.0
        score = alpha * support + (1.0 - alpha) * coverage
        scored.append(
            {
                **cluster,
                "support": round(support, 4),
                "coverage": round(coverage, 4),
                "score": round(score, 4),
            }
        )
    supported = [cluster for cluster in scored if cluster["support"] >= support_threshold]
    supported.sort(key=lambda item: (-item["score"], -item["support"], item["name"].lower()))
    selected = supported[:max_uigs]
    loss = 0.0
    if supported:
        loss = 1.0 - (len(selected) / len(supported))
    merge_rate = ((total_run_uigs - len(clusters)) / total_run_uigs) if total_run_uigs else 0.0
    metrics = {
        "canonical_uig_count": len(clusters),
        "consensus_uig_count": len(supported),
        "retained_uig_count": len(selected),
        "merge_rate": round(merge_rate, 4),
        "loss_from_truncation": round(loss, 4),
        "consensus_uig_rate": round((len(supported) / len(clusters)), 4) if clusters else None,
    }
    return selected, metrics


def _assignment_summary(assignments_df: pd.DataFrame) -> Dict[str, Any]:
    if assignments_df.empty:
        return {
            "assigned_articles": 0,
            "coverage": 0.0,
            "evidence_rate": None,
            "assignments_per_article": None,
        }
    valid = assignments_df[assignments_df["category_of_impact"].ne("N/A")].copy()
    if valid.empty:
        return {
            "assigned_articles": 0,
            "coverage": 0.0,
            "evidence_rate": 0.0,
            "assignments_per_article": 0.0,
        }
    article_count = assignments_df["__article_uid"].nunique()
    evidence_count = valid["observed_impact_text"].fillna("N/A").astype(str).ne("N/A").sum()
    return {
        "assigned_articles": int(valid["__article_uid"].nunique()),
        "coverage": round(valid["__article_uid"].nunique() / article_count, 4) if article_count else 0.0,
        "evidence_rate": round(evidence_count / len(valid), 4) if len(valid) else None,
        "assignments_per_article": round(len(valid) / article_count, 4) if article_count else None,
    }


def _batch_overlap_summary(batch_sets: Sequence[set[int]]) -> Tuple[List[Optional[float]], Optional[float]]:
    if not batch_sets:
        return [], None
    per_batch = []
    pair_scores = []
    for left in range(len(batch_sets)):
        scores = []
        for right in range(len(batch_sets)):
            if left == right:
                continue
            denom = len(batch_sets[left]) if batch_sets[left] else 0
            score = len(batch_sets[left] & batch_sets[right]) / denom if denom else 0.0
            scores.append(score)
            if left < right:
                pair_scores.append(score)
        per_batch.append(round(sum(scores) / len(scores), 4) if scores else None)
    overall = round(sum(pair_scores) / len(pair_scores), 4) if pair_scores else None
    return per_batch, overall


def run_ensemble_mode(
    input_file: str,
    outdir: Path,
    models: List[str],
    bootstrap_batches: int,
    bootstrap_size: int,
    discovery_sample_size: int,
    runs_per_batch: int,
    llm_batch_size: int,
    judge_model: Optional[str],
    sleep_sec: float,
    max_text_chars: int,
    similarity_threshold: float,
    support_threshold: float,
    score_alpha: float,
    global_selection_threshold: float,
    seed: int,
) -> None:
    client = _make_client()
    outdir.mkdir(parents=True, exist_ok=True)
    batch_dir = outdir / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)

    df = _load_dataframe(input_file).reset_index(drop=True)
    text_col = _detect_text_column(df)
    species_col = _detect_species_column(df)
    relevance_mask = _detect_relevance_mask(df)
    relevant_df = df[relevance_mask].copy().reset_index(drop=True)
    if relevant_df.empty:
        raise ValueError("No relevant articles found for ensemble extraction.")

    relevant_df["__article_uid"] = relevant_df.index + 1
    rng = random.Random(seed)
    judge_model = judge_model or models[0]

    batch_sets: List[set[int]] = []
    batch_metrics_rows: List[Dict[str, Any]] = []
    assignments_frames: List[pd.DataFrame] = []
    selected_batch_uigs: List[Dict[str, Any]] = []

    population = relevant_df["__article_uid"].tolist()
    id_to_row = relevant_df.set_index("__article_uid")

    for batch_id in range(1, bootstrap_batches + 1):
        sampled_ids = rng.choices(population, k=bootstrap_size) if bootstrap_size > 0 else []
        sampled_df = id_to_row.loc[sampled_ids].reset_index().rename(columns={"index": "__article_uid"})
        unique_batch_df = sampled_df.drop_duplicates(subset=["__article_uid"]).copy().reset_index(drop=True)
        batch_sets.append(set(unique_batch_df["__article_uid"].tolist()))

        discovery_n = min(discovery_sample_size, len(unique_batch_df))
        discovery_df = unique_batch_df.sample(n=discovery_n, random_state=seed + batch_id).reset_index(drop=True)

        run_records = []
        run_uig_sets = []
        run_uig_items: List[Dict[str, Any]] = []
        for run_id in range(1, runs_per_batch + 1):
            model = models[(run_id - 1) % len(models)]
            if len(discovery_df) > 1:
                run_df = discovery_df.sample(frac=1.0, random_state=seed + batch_id + run_id).reset_index(drop=True)
            else:
                run_df = discovery_df.copy()
            nested_rows, flat = _extract_articles(
                client,
                model=model,
                articles_df=run_df,
                text_col=text_col,
                species_col=species_col,
                mode="discover",
                llm_batch_size=llm_batch_size,
                sleep_sec=sleep_sec,
                max_text_chars=max_text_chars,
            )
            run_uigs = _build_run_uigs(flat, max_uigs=MAX_UIGS)
            for uig in run_uigs:
                uig.update({"batch_id": batch_id, "run_id": run_id, "model": model})
                run_uig_items.append(uig)
            run_records.append(
                {
                    "run_id": run_id,
                    "model": model,
                    "nested_rows": nested_rows,
                    "run_uigs": run_uigs,
                }
            )
            run_uig_sets.append({uig["name"] for uig in run_uigs})

        canonical_clusters = _harmonize_uigs(run_uig_items, similarity_threshold=similarity_threshold)
        selected_uigs, adjudication_metrics = _adjudicate_clusters(
            canonical_clusters,
            runs_per_batch=runs_per_batch,
            batch_article_count=len(unique_batch_df),
            support_threshold=support_threshold,
            alpha=score_alpha,
            max_uigs=MAX_UIGS,
        )
        final_categories = [uig["name"] for uig in selected_uigs]

        if final_categories:
            _, assignment_flat = _extract_articles(
                client,
                model=judge_model,
                articles_df=unique_batch_df,
                text_col=text_col,
                species_col=species_col,
                mode="apply",
                llm_batch_size=llm_batch_size,
                sleep_sec=sleep_sec,
                max_text_chars=max_text_chars,
                enum_categories=final_categories,
            )
        else:
            assignment_flat = pd.DataFrame(columns=NEW_COLS + ["__article_uid"])

        if not assignment_flat.empty:
            assignment_flat = assignment_flat.merge(
                unique_batch_df[["__article_uid", species_col]].rename(columns={species_col: "species"}),
                on="__article_uid",
                how="left",
            )
        assignment_flat["batch_id"] = batch_id
        assignments_frames.append(assignment_flat)

        assignment_metrics = _assignment_summary(assignment_flat)
        mean_jaccard = _mean_pairwise_jaccard(run_uig_sets)
        batch_metrics = {
            "batch_id": batch_id,
            "sampled_rows": len(sampled_df),
            "unique_rows": len(unique_batch_df),
            "discovery_rows": len(discovery_df),
            "mean_run_jaccard": mean_jaccard,
            **adjudication_metrics,
            **assignment_metrics,
        }
        batch_metrics_rows.append(batch_metrics)

        batch_payload = {
            "batch_id": batch_id,
            "sampled_article_uids": sampled_ids,
            "unique_article_uids": unique_batch_df["__article_uid"].tolist(),
            "run_records": run_records,
            "canonical_clusters": canonical_clusters,
            "selected_uigs": selected_uigs,
            "batch_metrics": batch_metrics,
        }
        (batch_dir / f"batch_{batch_id:03d}.json").write_text(
            json.dumps(batch_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        for uig in selected_uigs:
            selected_batch_uigs.append(
                {
                    **uig,
                    "batch_id": batch_id,
                    "article_uids": sorted(
                        {
                            int(uid)
                            for uid in assignment_flat.loc[
                                assignment_flat["category_of_impact"].eq(uig["name"]), "__article_uid"
                            ].dropna().tolist()
                        }
                    )
                    if not assignment_flat.empty
                    else [],
                }
            )

    overlap_by_batch, overall_overlap = _batch_overlap_summary(batch_sets)
    for metrics, overlap in zip(batch_metrics_rows, overlap_by_batch):
        metrics["mean_overlap_with_other_batches"] = overlap

    batch_metrics_df = pd.DataFrame(batch_metrics_rows)
    batch_metrics_df.to_csv(outdir / "uig_batch_metrics.csv", index=False)

    assignments_df = pd.concat(assignments_frames, ignore_index=True) if assignments_frames else pd.DataFrame()
    if not assignments_df.empty:
        assignments_df.to_csv(outdir / "uig_batch_assignments.csv", index=False)

    global_clusters = _harmonize_uigs(selected_batch_uigs, similarity_threshold=similarity_threshold)
    global_taxonomy = []
    mapping = {}
    for cluster in global_clusters:
        selection_probability = len(cluster["batch_ids"]) / bootstrap_batches if bootstrap_batches else 0.0
        robust = selection_probability >= global_selection_threshold
        global_name = cluster["name"]
        global_taxonomy.append(
            {
                "name": global_name,
                "definition": cluster["definition"],
                "evidence_examples": cluster["evidence_examples"],
                "batch_ids": cluster["batch_ids"],
                "article_uids": cluster["article_uids"],
                "selection_probability": round(selection_probability, 4),
                "robust": robust,
            }
        )
        for member in cluster["members"]:
            mapping[(member.get("batch_id"), member.get("name"))] = global_name

    robust_names = {item["name"] for item in global_taxonomy if item["robust"]}
    if not assignments_df.empty:
        assignments_df["global_uig"] = assignments_df.apply(
            lambda row: mapping.get((int(row["batch_id"]), row["category_of_impact"]), "N/A")
            if row.get("category_of_impact") != "N/A"
            else "N/A",
            axis=1,
        )
        assignments_df.to_csv(outdir / "uig_batch_assignments.csv", index=False)
        robust_assignments = assignments_df[assignments_df["global_uig"].isin(robust_names)]
        covered_articles = robust_assignments["__article_uid"].nunique()
        global_coverage = round(covered_articles / relevant_df["__article_uid"].nunique(), 4)
    else:
        global_coverage = 0.0

    batch_uig_sets = []
    for batch_id in range(1, bootstrap_batches + 1):
        batch_uig_sets.append(
            {
                item["name"]
                for item in global_taxonomy
                if batch_id in item["batch_ids"] and item["robust"]
            }
        )
    global_summary = {
        "bootstrap_batches": bootstrap_batches,
        "bootstrap_size": bootstrap_size,
        "runs_per_batch": runs_per_batch,
        "overall_mean_overlap": overall_overlap,
        "resample_coverage": round(len(set().union(*batch_sets)) / relevant_df["__article_uid"].nunique(), 4) if batch_sets else None,
        "global_stability_jaccard": _mean_pairwise_jaccard(batch_uig_sets),
        "global_coverage": global_coverage,
        "selection_probability_counts": {
            ">=0.5": sum(1 for item in global_taxonomy if item["selection_probability"] >= 0.5),
            ">=0.7": sum(1 for item in global_taxonomy if item["selection_probability"] >= 0.7),
            ">=0.9": sum(1 for item in global_taxonomy if item["selection_probability"] >= 0.9),
        },
        "global_taxonomy": global_taxonomy,
    }
    (outdir / "uig_global_taxonomy.json").write_text(
        json.dumps(global_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote batch metrics -> {outdir / 'uig_batch_metrics.csv'}")
    print(f"Wrote global taxonomy -> {outdir / 'uig_global_taxonomy.json'}")
    if not assignments_df.empty:
        print(f"Wrote batch assignments -> {outdir / 'uig_batch_assignments.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["discover", "apply", "pipeline", "ensemble"], default="discover")
    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--out_json_records", default=OUT_JSON_RECORDS)
    parser.add_argument("--out_json_nested", default=OUT_JSON_NESTED)
    parser.add_argument("--out_csv", default=OUT_CSV)
    parser.add_argument("--categories", default=CATS_JSON)
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--models", default=f"{MODEL},{MODEL},{MODEL}")
    parser.add_argument("--judge_model", default=None)
    parser.add_argument("--llm_batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--sleep_sec", type=float, default=SLEEP_SEC)
    parser.add_argument("--max_text_chars", type=int, default=3000)
    parser.add_argument("--outdir", default="data/extraction")
    parser.add_argument("--bootstrap_batches", type=int, default=10)
    parser.add_argument("--bootstrap_size", type=int, default=1000)
    parser.add_argument("--discovery_sample_size", type=int, default=60)
    parser.add_argument("--runs_per_batch", type=int, default=3)
    parser.add_argument("--similarity_threshold", type=float, default=0.58)
    parser.add_argument("--support_threshold", type=float, default=0.6)
    parser.add_argument("--score_alpha", type=float, default=0.7)
    parser.add_argument("--global_selection_threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "pipeline":
        run_mode(
            "discover",
            args.input,
            args.sample_size,
            Path(args.categories),
            Path(args.out_json_records),
            Path(args.out_json_nested),
            Path(args.out_csv),
            model=args.model,
            llm_batch_size=args.llm_batch_size,
            sleep_sec=args.sleep_sec,
            max_text_chars=args.max_text_chars,
        )
        run_mode(
            "apply",
            args.input,
            None,
            Path(args.categories),
            Path(args.out_json_records).with_name(Path(args.out_json_records).stem + ".apply.json"),
            Path(args.out_json_nested).with_name(Path(args.out_json_nested).stem + ".apply.json"),
            Path(args.out_csv).with_name(Path(args.out_csv).stem + ".apply.csv"),
            model=args.model,
            llm_batch_size=args.llm_batch_size,
            sleep_sec=args.sleep_sec,
            max_text_chars=args.max_text_chars,
        )
        return

    if args.mode == "ensemble":
        model_list = [item.strip() for item in args.models.split(",") if item.strip()]
        if not model_list:
            raise ValueError("--models must provide at least one model id.")
        run_ensemble_mode(
            input_file=args.input,
            outdir=Path(args.outdir),
            models=model_list,
            bootstrap_batches=args.bootstrap_batches,
            bootstrap_size=args.bootstrap_size,
            discovery_sample_size=args.discovery_sample_size,
            runs_per_batch=args.runs_per_batch,
            llm_batch_size=args.llm_batch_size,
            judge_model=args.judge_model,
            sleep_sec=args.sleep_sec,
            max_text_chars=args.max_text_chars,
            similarity_threshold=args.similarity_threshold,
            support_threshold=args.support_threshold,
            score_alpha=args.score_alpha,
            global_selection_threshold=args.global_selection_threshold,
            seed=args.seed,
        )
        return

    run_mode(
        args.mode,
        args.input,
        args.sample_size,
        Path(args.categories),
        Path(args.out_json_records),
        Path(args.out_json_nested),
        Path(args.out_csv),
        model=args.model,
        llm_batch_size=args.llm_batch_size,
        sleep_sec=args.sleep_sec,
        max_text_chars=args.max_text_chars,
    )


if __name__ == "__main__":
    main()
