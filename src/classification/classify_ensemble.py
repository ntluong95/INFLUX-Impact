"""Ensemble relevance classification using configurable OpenAI-backed models."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.classification.llm_backends import OpenAIChatClient
from src.utils.common import sha256_text, text_or_empty
from src.utils.llm_helpers import choose_extraction_input, parse_model_json, sanitize_alias


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


def _to_bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _invoke_single_model(
    client: OpenAIChatClient,
    model_id: str,
    prompt: str,
    gen_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any] | None, str]:
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
        return parsed, ""
    except Exception as exc:  # pragma: no cover - external API behavior
        return None, str(exc)


def _is_complete_classification_row(row: Dict[str, Any], model_aliases: List[str]) -> bool:
    final_label = text_or_empty(row.get("final_label"))
    if not final_label:
        return False
    if final_label == "skipped":
        return True

    for alias in model_aliases:
        rel_col = f"{alias}_relevant"
        err_col = f"{alias}_error"
        rel_val = _to_bool_or_none(row.get(rel_col))
        err_val = text_or_empty(row.get(err_col))
        if rel_val is None and not err_val:
            return False
    return True


def _write_checkpoint(rows: List[Dict[str, Any]], output_path: Path) -> None:
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _classify_one_article(
    row: pd.Series,
    client: OpenAIChatClient,
    cls_cfg: Dict[str, Any],
    logger=None,
) -> Dict[str, Any]:
    title = text_or_empty(row.get("extracted_title"))
    text = text_or_empty(row.get("extracted_text"))

    canonical_url = text_or_empty(row.get("canonical_url"))
    clean_title = title.strip()
    clean_text = text.strip()

    text_hash = sha256_text(clean_text) if clean_text else ""
    now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    result: Dict[str, Any] = {
        "canonical_url": canonical_url,
        "extracted_title": clean_title,
        "text_hash": text_hash,
        "final_label": "undetermined",
        "final_confidence": 0.0,
        "classified_at": now_iso,
    }

    models = cls_cfg.get("models", [])
    missing_reasons: List[str] = []
    if not clean_title:
        missing_reasons.append("missing_extracted_title")
    if not clean_text:
        missing_reasons.append("missing_extracted_text")
    if missing_reasons:
        reason = "skipped_" + "+".join(missing_reasons)
        for model_cfg in models:
            alias = sanitize_alias(model_cfg.get("alias", model_cfg.get("model_id", "model")))
            result[f"{alias}_model_id"] = model_cfg.get("model_id", "")
            result[f"{alias}_relevant"] = ""
            result[f"{alias}_confidence"] = ""
            result[f"{alias}_rationale"] = ""
            result[f"{alias}_evidence_spans"] = ""
            result[f"{alias}_error"] = reason
        result["final_label"] = "skipped"
        result["final_confidence"] = 0.0
        return result

    prep_title, prep_text = choose_extraction_input(
        title=clean_title,
        text=clean_text,
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
    """Run 3-model ensemble classification and write CSV output.

    Execution strategy is model-first (all rows for model A, then model B, ...),
    which avoids heavy local model thrashing on constrained hardware.
    """
    resume_if_exists = bool(cls_cfg.get("resume_if_exists", True))
    checkpoint_every = int(cls_cfg.get("checkpoint_every", 20))
    inter_article_delay_seconds = float(cls_cfg.get("inter_article_delay_seconds", 0.0))
    gen_cfg = cls_cfg.get("generation", {})
    models = cls_cfg.get("models", [])
    model_aliases = [sanitize_alias(m.get("alias", m.get("model_id", "model"))) for m in models]

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
        complete_rows: List[Dict[str, Any]] = []
        partial_rows = 0
        for rec in existing_df.to_dict(orient="records"):
            if _is_complete_classification_row(rec, model_aliases=model_aliases):
                complete_rows.append(rec)
                processed_urls.add(text_or_empty(rec.get("canonical_url")))
            else:
                partial_rows += 1
        existing_rows = complete_rows
        if logger is not None:
            logger.info(
                "Resuming classification from existing output: complete_rows=%s partial_rows_recomputed=%s",
                len(existing_rows),
                partial_rows,
            )

    rows: List[Dict[str, Any]] = list(existing_rows)
    pending_items: List[Dict[str, Any]] = []
    total_rows = len(extracted_df)
    prepared_rows = len(existing_rows)

    for _, row in extracted_df.iterrows():
        canonical_url = text_or_empty(row.get("canonical_url"))
        if canonical_url in processed_urls:
            continue

        title = text_or_empty(row.get("extracted_title")).strip()
        text = text_or_empty(row.get("extracted_text")).strip()
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

        for model_cfg in models:
            alias = sanitize_alias(model_cfg.get("alias", model_cfg.get("model_id", "model")))
            result[f"{alias}_model_id"] = model_cfg.get("model_id", "")
            result[f"{alias}_relevant"] = ""
            result[f"{alias}_confidence"] = ""
            result[f"{alias}_rationale"] = ""
            result[f"{alias}_evidence_spans"] = ""
            result[f"{alias}_error"] = ""

        missing_reasons: List[str] = []
        if not title:
            missing_reasons.append("missing_extracted_title")
        if not text:
            missing_reasons.append("missing_extracted_text")

        if missing_reasons:
            reason = "skipped_" + "+".join(missing_reasons)
            for model_cfg in models:
                alias = sanitize_alias(model_cfg.get("alias", model_cfg.get("model_id", "model")))
                result[f"{alias}_error"] = reason
            result["final_label"] = "skipped"
            result["final_confidence"] = 0.0
            rows.append(result)
            prepared_rows += 1
            continue

        prep_title, prep_text = choose_extraction_input(
            title=title,
            text=text,
            max_chars=int(cls_cfg.get("text_max_chars", 12000)),
            summary_cfg=cls_cfg.get("deterministic_summary", {}),
        )
        prompt = _build_user_prompt(prep_title, prep_text)

        rows.append(result)
        pending_items.append(
            {
                "row_idx": len(rows) - 1,
                "canonical_url": canonical_url,
                "prompt": prompt,
            }
        )
        prepared_rows += 1

    if logger is not None:
        logger.info(
            "Prepared classification rows: total_input=%s pending_for_llm=%s precompleted=%s",
            total_rows,
            len(pending_items),
            len(existing_rows),
        )

    for model_cfg in models:
        alias = sanitize_alias(model_cfg.get("alias", model_cfg.get("model_id", "model")))
        model_id = model_cfg.get("model_id", "")
        model_done = 0

        for item in pending_items:
            target = rows[item["row_idx"]]
            parsed, error = _invoke_single_model(
                client=client,
                model_id=model_id,
                prompt=item["prompt"],
                gen_cfg=gen_cfg,
            )

            if parsed is not None:
                target[f"{alias}_relevant"] = parsed["relevant"]
                target[f"{alias}_confidence"] = parsed["confidence"]
                target[f"{alias}_rationale"] = parsed["rationale"]
                target[f"{alias}_evidence_spans"] = " | ".join(parsed["evidence_spans"])
                target[f"{alias}_error"] = ""
            else:
                target[f"{alias}_error"] = error
                if logger is not None:
                    logger.warning("Model call failed for %s on %s: %s", model_id, item["canonical_url"], error)

            model_done += 1
            if logger is not None and (model_done % 20 == 0 or model_done == len(pending_items)):
                logger.info("Model progress [%s]: %s/%s", alias, model_done, len(pending_items))

            if checkpoint_every > 0 and model_done % checkpoint_every == 0:
                _write_checkpoint(rows, output_path)
                if logger is not None:
                    logger.info("Classification checkpoint written: %s", output_path)

            if inter_article_delay_seconds > 0:
                time.sleep(inter_article_delay_seconds)

    for row in rows:
        if text_or_empty(row.get("final_label")) == "skipped":
            continue

        preds: List[Dict[str, Any]] = []
        for alias in model_aliases:
            rel_val = _to_bool_or_none(row.get(f"{alias}_relevant"))
            if rel_val is None:
                continue
            try:
                conf_val = float(row.get(f"{alias}_confidence", 0.0))
            except Exception:
                conf_val = 0.0
            preds.append(
                {
                    "relevant": rel_val,
                    "confidence": max(0.0, min(1.0, conf_val)),
                }
            )

        label, conf = _aggregate_ensemble(preds)
        row["final_label"] = label
        row["final_confidence"] = round(float(conf), 4)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)
    if logger is not None:
        logger.info("Classification progress: %s/%s", len(rows), total_rows)
    return out_df
