"""Ensemble relevance classification using configurable OpenAI-backed models."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

try:
    from .llm_backends import OpenAIChatClient
    from .utils import (
        choose_extraction_input,
        parse_model_json,
        sanitize_alias,
        sha256_text,
        text_or_empty,
    )
except ImportError:  # pragma: no cover - script execution path
    from llm_backends import OpenAIChatClient
    from utils import (
        choose_extraction_input,
        parse_model_json,
        sanitize_alias,
        sha256_text,
        text_or_empty,
    )


SYSTEM_PROMPT = """You are a strict classification model for news analysis.
Determine whether an article is relevant to cascading impacts of Zika.
Cascading impacts means downstream social, economic, policy, health-system, behavioral, inequality,
travel, education, supply-chain, governance, or public-fear consequences beyond basic mention or case counts.

Return STRICT JSON only with keys:
{
  "relevant": true/false,
  "confidence": 0.0-1.0,
  "rationale": "max 18 words",
  "evidence_spans": ["max 12 words", "max 12 words"]
}

Rules:
- relevant=true only if downstream consequences are present.
- If article only mentions Zika, case counts, or generic mosquito info without knock-on effects, set relevant=false.
- confidence must reflect certainty and be between 0 and 1.
- evidence_spans must contain at most 2 short snippets from the provided text when available.
- Do not add extra keys.
- Output valid JSON only (no markdown/code fences/explanations).
"""


def _build_user_prompt(title: str, text_excerpt: str) -> str:
    return (
        "Classify this article for relevance to cascading impacts of Zika.\n\n"
        f"TITLE:\n{title or '[missing]'}\n\n"
        f"TEXT EXCERPT:\n{text_excerpt or '[missing]'}\n\n"
        "Output only one valid JSON object with concise fields:\n"
        '- rationale <= 18 words\n'
        "- evidence_spans <= 2 items and each <= 12 words."
    )


def _aggregate_ensemble(predictions: List[Dict[str, Any]]) -> Tuple[str, float]:
    valid = [p for p in predictions if isinstance(p.get("relevant"), bool)]
    if not valid:
        return "undetermined", 0.0

    true_conf = [float(p.get("confidence", 0.0)) for p in valid if p["relevant"] is True]
    false_conf = [float(p.get("confidence", 0.0)) for p in valid if p["relevant"] is False]

    if len(true_conf) > len(false_conf):
        return "relevant", float(sum(true_conf) / len(true_conf))
    if len(false_conf) > len(true_conf):
        return "not_relevant", float(sum(false_conf) / len(false_conf))

    avg_true = float(sum(true_conf) / len(true_conf)) if true_conf else 0.0
    avg_false = float(sum(false_conf) / len(false_conf)) if false_conf else 0.0
    if avg_true >= avg_false:
        return "relevant", avg_true
    return "not_relevant", avg_false


def _classify_one_article(
    row: pd.Series,
    client: OpenAIChatClient,
    cls_cfg: Dict[str, Any],
    logger=None,
) -> Dict[str, Any]:
    title = text_or_empty(row.get("extracted_title"))
    text = text_or_empty(row.get("extracted_text"))

    canonical_url = text_or_empty(row.get("canonical_url"))
    min_text = int(cls_cfg.get("min_text_chars_for_llm", 250))

    text_hash = sha256_text(text) if text else ""
    now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    result: Dict[str, Any] = {
        "canonical_url": canonical_url,
        "extracted_title": title,
        "text_hash": text_hash,
        "final_label": "undetermined",
        "final_confidence": 0.0,
        "classified_at": now_iso,
    }

    models = cls_cfg.get("models", [])

    prefilter_keywords = [
        str(k).strip().lower() for k in cls_cfg.get("prefilter_keywords", []) if str(k).strip()
    ]
    if prefilter_keywords:
        haystack = f"{title}\n{text}".lower()
        min_hits = max(1, int(cls_cfg.get("prefilter_min_keyword_hits", 1)))
        keyword_hits = sum(1 for keyword in prefilter_keywords if keyword in haystack)
        if keyword_hits < min_hits:
            for model_cfg in models:
                alias = sanitize_alias(model_cfg.get("alias", model_cfg.get("model_id", "model")))
                result[f"{alias}_model_id"] = model_cfg.get("model_id", "")
                result[f"{alias}_relevant"] = ""
                result[f"{alias}_confidence"] = ""
                result[f"{alias}_rationale"] = ""
                result[f"{alias}_evidence_spans"] = ""
                result[f"{alias}_error"] = f"skipped_keyword_prefilter_hits_{keyword_hits}"
            result["final_label"] = str(cls_cfg.get("prefilter_default_label", "not_relevant"))
            result["final_confidence"] = round(float(cls_cfg.get("prefilter_confidence", 0.95)), 4)
            return result

    if len(text) < min_text:
        for model_cfg in models:
            alias = sanitize_alias(model_cfg.get("alias", model_cfg.get("model_id", "model")))
            result[f"{alias}_model_id"] = model_cfg.get("model_id", "")
            result[f"{alias}_relevant"] = ""
            result[f"{alias}_confidence"] = ""
            result[f"{alias}_rationale"] = ""
            result[f"{alias}_evidence_spans"] = ""
            result[f"{alias}_error"] = "skipped_insufficient_text"
        return result

    prep_title, prep_text = choose_extraction_input(
        title=title,
        text=text,
        max_chars=int(cls_cfg.get("text_max_chars", 12000)),
        summary_cfg=cls_cfg.get("deterministic_summary", {}),
    )

    prompt = _build_user_prompt(prep_title, prep_text)
    gen_cfg = cls_cfg.get("generation", {})
    parallel_models = max(1, int(cls_cfg.get("per_article_parallel_models", 1)))

    parsed_predictions: List[Dict[str, Any]] = []
    model_entries = []
    for model_cfg in models:
        alias = sanitize_alias(model_cfg.get("alias", model_cfg.get("model_id", "model")))
        model_id = model_cfg.get("model_id")
        model_entries.append((alias, model_id, model_cfg))

        result[f"{alias}_model_id"] = model_id
        result[f"{alias}_relevant"] = ""
        result[f"{alias}_confidence"] = ""
        result[f"{alias}_rationale"] = ""
        result[f"{alias}_evidence_spans"] = ""
        result[f"{alias}_error"] = ""

    def _call_model(alias: str, model_id: str, model_cfg: Dict[str, Any]):
        try:
            api_out = client.chat_completion(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=float(gen_cfg.get("temperature", 0.0)),
                max_tokens=int(gen_cfg.get("max_tokens", 500)),
                strict_json=True,
            )

            parsed = parse_model_json(api_out["text"])
            return alias, model_id, parsed, ""
        except Exception as exc:  # pragma: no cover - external API behavior
            return alias, model_id, None, str(exc)

    if parallel_models > 1 and len(model_entries) > 1:
        with ThreadPoolExecutor(max_workers=min(parallel_models, len(model_entries))) as executor:
            futures = [
                executor.submit(_call_model, alias, model_id, model_cfg)
                for alias, model_id, model_cfg in model_entries
            ]
            for future in as_completed(futures):
                alias, model_id, parsed, error = future.result()
                if parsed is not None:
                    result[f"{alias}_relevant"] = parsed["relevant"]
                    result[f"{alias}_confidence"] = parsed["confidence"]
                    result[f"{alias}_rationale"] = parsed["rationale"]
                    result[f"{alias}_evidence_spans"] = " | ".join(parsed["evidence_spans"])
                    parsed_predictions.append(parsed)
                else:
                    result[f"{alias}_error"] = error
                    if logger is not None:
                        logger.warning(
                            "Model call failed for %s on %s: %s",
                            model_id,
                            canonical_url,
                            error,
                        )
    else:
        for alias, model_id, model_cfg in model_entries:
            alias, model_id, parsed, error = _call_model(alias, model_id, model_cfg)
            if parsed is not None:
                result[f"{alias}_relevant"] = parsed["relevant"]
                result[f"{alias}_confidence"] = parsed["confidence"]
                result[f"{alias}_rationale"] = parsed["rationale"]
                result[f"{alias}_evidence_spans"] = " | ".join(parsed["evidence_spans"])
                parsed_predictions.append(parsed)
            else:
                result[f"{alias}_error"] = error
                if logger is not None:
                    logger.warning("Model call failed for %s on %s: %s", model_id, canonical_url, error)

    label, conf = _aggregate_ensemble(parsed_predictions)
    result["final_label"] = label
    result["final_confidence"] = round(float(conf), 4)
    return result


def classify_articles_ensemble(
    extracted_df: pd.DataFrame,
    client: OpenAIChatClient,
    cls_cfg: Dict[str, Any],
    output_path: Path,
    force: bool = False,
    logger=None,
) -> pd.DataFrame:
    """Run 3-model ensemble classification and write CSV output."""
    resume_if_exists = bool(cls_cfg.get("resume_if_exists", True))
    checkpoint_every = int(cls_cfg.get("checkpoint_every", 20))

    existing_rows: List[Dict[str, Any]] = []
    processed_urls = set()
    if output_path.exists() and not force:
        if not resume_if_exists:
            if logger is not None:
                logger.info(
                    "Classification output exists; skipping (use --force to recompute): %s",
                    output_path,
                )
            return pd.read_csv(output_path, low_memory=False)

        existing_df = pd.read_csv(output_path, low_memory=False)
        existing_rows = existing_df.to_dict(orient="records")
        processed_urls = set(existing_df.get("canonical_url", pd.Series(dtype=str)).astype(str))
        if logger is not None:
            logger.info(
                "Resuming classification from existing output: %s rows already done",
                len(existing_rows),
            )

    cache_enabled = bool(cls_cfg.get("cache_by_text_hash", True))
    cache: Dict[str, Dict[str, Any]] = {}
    cache_hits = 0

    rows: List[Dict[str, Any]] = list(existing_rows)
    completed = len(processed_urls)
    total = len(extracted_df)

    for _, row in extracted_df.iterrows():
        canonical_url = text_or_empty(row.get("canonical_url"))
        title = text_or_empty(row.get("extracted_title"))
        text = text_or_empty(row.get("extracted_text"))
        text_hash = sha256_text(text) if text else ""
        cache_key = sha256_text(f"{title}\n{text}") if cache_enabled else ""

        if canonical_url in processed_urls:
            continue

        if cache_enabled and cache_key in cache:
            cached = dict(cache[cache_key])
            cached["canonical_url"] = canonical_url
            cached["extracted_title"] = title
            cached["text_hash"] = text_hash
            rows.append(cached)
            cache_hits += 1
        else:
            classified = _classify_one_article(row, client=client, cls_cfg=cls_cfg, logger=logger)
            rows.append(classified)
            if cache_enabled:
                cache[cache_key] = dict(classified)

        completed += 1
        if logger is not None and (completed % 20 == 0 or completed == total):
            logger.info("Classification progress: %s/%s", completed, total)

        if checkpoint_every > 0 and completed % checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(output_path, index=False)
            if logger is not None:
                logger.info("Classification checkpoint written: %s", output_path)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)
    if logger is not None and cache_enabled:
        logger.info("Classification cache stats: unique_inputs=%s cache_hits=%s", len(cache), cache_hits)
    return out_df
