"""Ensemble relevance classification with configurable aggregation and validation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

import pandas as pd

from src.classification.llm_backends import OpenAIChatClient
from src.utils.common import sha256_text, text_or_empty, utc_now_iso
from src.utils.llm_helpers import choose_extraction_input, parse_model_json, sanitize_alias

SYSTEM_PROMPT_TEMPLATE = """You are a strict, high-precision classifier for news relevance to cascading impacts of {target_name}.

Return STRICT JSON only:
{{
  "relevant": true/false,
  "confidence": 0.0-1.0,
  "rationale": "...",
  "evidence_spans": ["...", "..."]
}}

Hard constraints:
- "confidence" must be the estimated probability that the article is relevant.
- rationale must be 12-24 words.
- rationale must include all three components in one sentence:
  1) [DOMAIN] label in square brackets (for example [health_system], [economy], [travel], [policy], [inequality], [education], [maternal_health], [public_fear], [governance], [supply_chain])
  2) downstream outcome
  3) explicit linkage text using "linkage:" and mention {target_name}.
- evidence_spans must contain 0-2 short verbatim snippets from provided text only, each <= 12 words.
- relevant=true only when downstream consequences are present, not only case counts or generic mention.
- Do not add extra keys.
- Output valid JSON only (no markdown/code fences/explanations).
"""


def _target_name(cls_cfg: Dict[str, Any]) -> str:
    return str(cls_cfg.get("target_name") or cls_cfg.get("target_epp") or "Zika").strip()


def _target_aliases(cls_cfg: Dict[str, Any]) -> List[str]:
    target_name = _target_name(cls_cfg)
    raw_aliases = cls_cfg.get("target_aliases") or []
    aliases = [target_name]
    aliases.extend(str(value).strip() for value in raw_aliases if str(value).strip())
    seen = set()
    ordered: List[str] = []
    for alias in aliases:
        lowered = alias.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered.append(alias)
    return ordered


def _build_user_prompt(title: str, text_excerpt: str, target_name: str) -> str:
    return (
        f"Classify this article for relevance to cascading impacts of {target_name}.\n\n"
        f"TITLE:\n{title or '[missing]'}\n\n"
        f"TEXT EXCERPT:\n{text_excerpt or '[missing]'}\n\n"
        "Output only one valid JSON object.\n"
        f'Rationale format: "[DOMAIN] <downstream outcome>; linkage: <how outcome is tied to {target_name}>".\n'
        "- confidence must be the probability that the article is relevant.\n"
        "- rationale must be 12-24 words.\n"
        "- evidence_spans <= 2 items and each <= 12 words."
    )


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


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text or ""))


def _validate_prediction_quality(
    parsed: Dict[str, Any],
    quality_cfg: Dict[str, Any],
    target_aliases: Sequence[str],
) -> tuple[bool, str]:
    rationale = str(parsed.get("rationale", "")).strip()
    min_words = int(quality_cfg.get("min_words", 12))
    max_words = int(quality_cfg.get("max_words", 24))
    require_template = bool(quality_cfg.get("require_template", True))
    require_evidence_for_relevant = bool(
        quality_cfg.get("require_evidence_for_relevant", True)
    )

    wc = _word_count(rationale)
    if wc < min_words:
        return False, f"rationale_too_short:{wc}<{min_words}"
    if wc > max_words:
        return False, f"rationale_too_long:{wc}>{max_words}"

    if require_template:
        low = rationale.lower()
        if "[" not in rationale or "]" not in rationale:
            return False, "rationale_missing_domain_label"
        if "linkage:" not in low:
            return False, "rationale_missing_linkage_tag"
        if not any(alias.lower() in low for alias in target_aliases):
            return False, "rationale_missing_target_reference"

    if require_evidence_for_relevant and bool(parsed.get("relevant")):
        spans = parsed.get("evidence_spans", [])
        if not isinstance(spans, list) or len(spans) == 0:
            return False, "relevant_missing_evidence_span"

    return True, ""


def _invoke_single_model(
    client: OpenAIChatClient,
    model_id: str,
    prompt: str,
    gen_cfg: Dict[str, Any],
    quality_cfg: Dict[str, Any],
    target_name: str,
    target_aliases: Sequence[str],
) -> Tuple[Dict[str, Any] | None, str]:
    max_attempts = max(1, int(quality_cfg.get("max_attempts", 2)))
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_TEMPLATE.format(target_name=target_name),
        },
        {"role": "user", "content": prompt},
    ]
    last_error = ""

    for attempt in range(1, max_attempts + 1):
        try:
            api_out = client.chat_completion(
                model=model_id,
                messages=messages,
                temperature=float(gen_cfg.get("temperature", 0.0)),
                max_tokens=int(gen_cfg.get("max_tokens", 500)),
                strict_json=True,
            )
            parsed = parse_model_json(api_out["text"])
            valid, reason = _validate_prediction_quality(
                parsed,
                quality_cfg=quality_cfg,
                target_aliases=target_aliases,
            )
            if valid:
                return parsed, ""

            last_error = f"quality_warning:{reason}"
            if attempt < max_attempts:
                messages.extend(
                    [
                        {"role": "assistant", "content": api_out["text"]},
                        {
                            "role": "user",
                            "content": (
                                f"Revise previous JSON only. Failed check: {reason}. "
                                "Keep decision consistent unless clearly unsupported. "
                                f"Rationale must be 12-24 words and include [DOMAIN] plus "
                                f'"linkage:" with explicit {target_name} tie.'
                            ),
                        },
                    ]
                )
                continue
            return parsed, last_error
        except Exception as exc:  # pragma: no cover - external API behavior
            last_error = str(exc)
            if attempt < max_attempts:
                continue

    return None, last_error


def _is_complete_classification_row(
    row: Dict[str, Any], model_aliases: Sequence[str]
) -> bool:
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


def _parse_evidence_spans(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = text_or_empty(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(" | ") if part.strip()]


def _coerce_relevance_probability(
    label: bool,
    confidence: float | None,
    semantics: str,
) -> float:
    conf = 0.0 if confidence is None or math.isnan(confidence) else float(confidence)
    conf = max(0.0, min(1.0, conf))

    if semantics == "decision_confidence":
        return conf if label else 1.0 - conf
    if semantics == "relevance_probability":
        return conf

    if label is True and conf < 0.5:
        return 1.0 - conf
    if label is False and conf > 0.5:
        return 1.0 - conf
    return conf


def _extract_prediction_records(
    row: Dict[str, Any],
    model_aliases: Sequence[str],
    settings: Dict[str, Any],
) -> List[Dict[str, Any]]:
    semantics = str(settings.get("confidence_semantics", "auto")).strip().lower()
    predictions: List[Dict[str, Any]] = []
    for alias in model_aliases:
        relevant = _to_bool_or_none(row.get(f"{alias}_relevant"))
        if relevant is None:
            continue
        confidence = _to_float_or_none(row.get(f"{alias}_confidence"))
        probability = _coerce_relevance_probability(relevant, confidence, semantics)
        predictions.append(
            {
                "alias": alias,
                "relevant": relevant,
                "confidence": 0.0 if confidence is None else max(0.0, min(1.0, confidence)),
                "relevance_probability": probability,
                "weight": float(settings.get("weights", {}).get(alias, 0.0)),
                "rationale": text_or_empty(row.get(f"{alias}_rationale")),
                "evidence_spans": _parse_evidence_spans(row.get(f"{alias}_evidence_spans")),
            }
        )
    return predictions


def _consolidate_evidence_spans(
    predictions: Sequence[Dict[str, Any]],
    final_label: str,
) -> List[str]:
    if not predictions:
        return []

    if final_label == "relevant":
        source_predictions = [pred for pred in predictions if pred["relevant"]]
    else:
        source_predictions = [
            max(
                predictions,
                key=lambda pred: (pred.get("weight", 0.0), pred.get("confidence", 0.0)),
            )
        ]

    seen = set()
    merged: List[str] = []
    for pred in source_predictions:
        for span in pred.get("evidence_spans", []):
            norm = span.lower()
            if norm in seen:
                continue
            seen.add(norm)
            merged.append(span)
    return merged


def _aggregate_prediction_records(
    predictions: Sequence[Dict[str, Any]],
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    if not predictions:
        return {
            "final_label": "undetermined",
            "final_confidence": 0.0,
            "final_relevance_probability": 0.0,
            "final_positive_agreement": 0.0,
            "final_supporting_models": "",
            "final_evidence_spans": "",
            "final_evidence_present": False,
        }

    available = [pred for pred in predictions if pred.get("weight", 0.0) > 0.0]
    if not available:
        available = list(predictions)

    weights_total = sum(float(pred.get("weight", 0.0)) for pred in available)
    if weights_total <= 0:
        equal_weight = 1.0 / len(available)
        normalized_weights = {pred["alias"]: equal_weight for pred in available}
    else:
        normalized_weights = {
            pred["alias"]: float(pred.get("weight", 0.0)) / weights_total
            for pred in available
        }

    weighted_probability = sum(
        normalized_weights[pred["alias"]] * float(pred["relevance_probability"])
        for pred in available
    )
    positive_votes = sum(1 for pred in available if pred["relevant"])
    positive_agreement = positive_votes / len(available)
    threshold = float(settings.get("threshold", 0.75))
    require_min_agreement = bool(settings.get("require_min_agreement", True))
    min_agreement = float(settings.get("min_agreement", 0.0)) if require_min_agreement else 0.0

    is_relevant = weighted_probability >= threshold
    if require_min_agreement:
        is_relevant = is_relevant and positive_agreement >= min_agreement

    final_label = "relevant" if is_relevant else "not_relevant"
    evidence_spans = _consolidate_evidence_spans(available, final_label=final_label)
    supporting_models = [pred["alias"] for pred in available if pred["relevant"]]

    return {
        "final_label": final_label,
        "final_confidence": round(
            weighted_probability if is_relevant else (1.0 - weighted_probability),
            4,
        ),
        "final_relevance_probability": round(weighted_probability, 4),
        "final_positive_agreement": round(positive_agreement, 4),
        "final_supporting_models": ",".join(supporting_models),
        "final_evidence_spans": " | ".join(evidence_spans),
        "final_evidence_present": bool(evidence_spans),
    }


def _normalize_weights(
    model_aliases: Sequence[str],
    raw_weights: Dict[str, Any] | None,
) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for alias in model_aliases:
        if isinstance(raw_weights, dict):
            try:
                weights[alias] = float(raw_weights.get(alias, 0.0))
            except Exception:
                weights[alias] = 0.0
        else:
            weights[alias] = 1.0

    total = sum(max(value, 0.0) for value in weights.values())
    if total <= 0:
        equal = 1.0 / len(model_aliases) if model_aliases else 0.0
        return {alias: equal for alias in model_aliases}

    return {
        alias: round(max(weight, 0.0) / total, 6) for alias, weight in weights.items()
    }


def _default_aggregation_settings(
    cls_cfg: Dict[str, Any],
    model_aliases: Sequence[str],
    model_weight_defaults: Dict[str, float],
) -> Dict[str, Any]:
    agg_cfg = cls_cfg.get("aggregation", {})
    return {
        "threshold": float(agg_cfg.get("threshold", 0.75)),
        "min_agreement": float(agg_cfg.get("min_agreement", 0.67)),
        "require_min_agreement": bool(agg_cfg.get("require_min_agreement", True)),
        "confidence_semantics": str(
            agg_cfg.get("confidence_semantics", "auto")
        ).strip().lower(),
        "weights": _normalize_weights(
            model_aliases,
            agg_cfg.get("weights") or model_weight_defaults,
        ),
        "source": "config",
    }


def _load_reference_labels(validation_cfg: Dict[str, Any], logger=None) -> pd.DataFrame | None:
    labels_path_value = text_or_empty(validation_cfg.get("labels_path")).strip()
    if not labels_path_value:
        return None

    labels_path = Path(labels_path_value)
    if not labels_path.exists():
        if logger is not None:
            logger.warning("Validation labels file not found: %s", labels_path)
        return None

    if labels_path.suffix.lower() == ".parquet":
        labels_df = pd.read_parquet(labels_path)
    else:
        labels_df = pd.read_csv(labels_path, low_memory=False)

    url_col = text_or_empty(validation_cfg.get("url_column")).strip() or "canonical_url"
    label_col = text_or_empty(validation_cfg.get("label_column")).strip() or "relevant"
    if url_col not in labels_df.columns or label_col not in labels_df.columns:
        if logger is not None:
            logger.warning(
                "Validation labels missing required columns: url=%s label=%s",
                url_col,
                label_col,
            )
        return None

    labels_df = labels_df.copy()
    labels_df["canonical_url"] = labels_df[url_col].map(lambda value: text_or_empty(value).strip())
    labels_df["_reference_label"] = labels_df[label_col].map(_to_bool_or_none)
    labels_df = labels_df[labels_df["canonical_url"].ne("") & labels_df["_reference_label"].notna()].copy()
    if labels_df.empty:
        if logger is not None:
            logger.warning("Validation labels file contains no usable rows: %s", labels_path)
        return None

    keep_cols = ["canonical_url", "_reference_label"]
    for optional_col in ["evidence_support", "attribution_correct"]:
        if optional_col in labels_df.columns:
            keep_cols.append(optional_col)
    labels_df = labels_df[keep_cols].drop_duplicates(subset=["canonical_url"], keep="first")
    return labels_df


def _compositions(total_units: int, n_parts: int) -> Iterator[Tuple[int, ...]]:
    if n_parts == 1:
        yield (total_units,)
        return
    for first in range(total_units + 1):
        for rest in _compositions(total_units - first, n_parts - 1):
            yield (first,) + rest


def _candidate_weight_maps(
    model_aliases: Sequence[str],
    base_weights: Dict[str, float],
    tune_weights: bool,
    step: float,
) -> List[Dict[str, float]]:
    candidates = [_normalize_weights(model_aliases, base_weights)]
    if not tune_weights or len(model_aliases) <= 1 or len(model_aliases) > 4:
        return candidates
    if step <= 0 or step >= 1:
        return candidates

    units = round(1.0 / step)
    if not math.isclose(units * step, 1.0, rel_tol=0.0, abs_tol=1e-6):
        return candidates

    seen = {
        tuple(round(candidate[alias], 6) for alias in model_aliases)
        for candidate in candidates
    }
    for parts in _compositions(int(units), len(model_aliases)):
        if sum(parts) == 0:
            continue
        weights = {
            alias: round(part / units, 6)
            for alias, part in zip(model_aliases, parts)
        }
        key = tuple(weights[alias] for alias in model_aliases)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(weights)
    return candidates


def _confusion_counts(y_true: Sequence[bool], y_pred: Sequence[bool]) -> Dict[str, int]:
    tp = fp = tn = fn = 0
    for actual, predicted in zip(y_true, y_pred):
        if actual and predicted:
            tp += 1
        elif actual and not predicted:
            fn += 1
        elif not actual and predicted:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _classification_metrics_from_counts(counts: Dict[str, int]) -> Dict[str, float | int]:
    tp = counts["tp"]
    fp = counts["fp"]
    tn = counts["tn"]
    fn = counts["fn"]

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = (recall + specificity) / 2.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    return {
        **counts,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "balanced_accuracy": round(balanced_accuracy, 4),
        "accuracy": round(accuracy, 4),
    }


def _brier_score(probabilities: Sequence[float], y_true: Sequence[bool]) -> float | None:
    if not probabilities:
        return None
    return round(
        sum((float(prob) - float(actual)) ** 2 for prob, actual in zip(probabilities, y_true))
        / len(probabilities),
        4,
    )


def _expected_calibration_error(
    probabilities: Sequence[float],
    y_true: Sequence[bool],
    n_bins: int,
) -> float | None:
    if not probabilities or n_bins <= 0:
        return None
    bin_totals = [0 for _ in range(n_bins)]
    bin_conf = [0.0 for _ in range(n_bins)]
    bin_true = [0.0 for _ in range(n_bins)]

    for prob, actual in zip(probabilities, y_true):
        idx = min(int(float(prob) * n_bins), n_bins - 1)
        bin_totals[idx] += 1
        bin_conf[idx] += float(prob)
        bin_true[idx] += float(actual)

    total = len(probabilities)
    ece = 0.0
    for idx in range(n_bins):
        if bin_totals[idx] == 0:
            continue
        avg_conf = bin_conf[idx] / bin_totals[idx]
        avg_true = bin_true[idx] / bin_totals[idx]
        ece += (bin_totals[idx] / total) * abs(avg_true - avg_conf)
    return round(ece, 4)


def _auprc(
    probabilities: Sequence[float],
    y_true: Sequence[bool],
    positive_agreements: Sequence[float],
    min_agreement: float,
    require_min_agreement: bool,
) -> float | None:
    if not probabilities:
        return None
    positives = sum(1 for value in y_true if value)
    if positives == 0:
        return None

    thresholds = sorted({round(float(prob), 6) for prob in probabilities}, reverse=True)
    if thresholds[-1] != 0.0:
        thresholds.append(0.0)
    if thresholds[0] != 1.0:
        thresholds.insert(0, 1.0)

    prev_recall = 0.0
    area = 0.0
    for threshold in thresholds:
        preds = []
        for prob, agreement in zip(probabilities, positive_agreements):
            positive = float(prob) >= threshold
            if require_min_agreement:
                positive = positive and float(agreement) >= min_agreement
            preds.append(bool(positive))
        counts = _confusion_counts(y_true, preds)
        metrics = _classification_metrics_from_counts(counts)
        precision = float(metrics["precision"])
        recall = float(metrics["recall"])
        if recall > prev_recall:
            area += precision * (recall - prev_recall)
            prev_recall = recall
    return round(area, 4)


def _agreement_summary(df: pd.DataFrame, model_aliases: Sequence[str]) -> Dict[str, float | int | None]:
    row_count = 0
    full_agreement_rows = 0
    disagreement_rows = 0
    pairwise_values: List[float] = []

    for _, row in df.iterrows():
        labels = [
            _to_bool_or_none(row.get(f"{alias}_relevant")) for alias in model_aliases
        ]
        labels = [label for label in labels if label is not None]
        if not labels:
            continue
        row_count += 1
        if len(set(labels)) == 1:
            full_agreement_rows += 1
        positive_agreement = sum(1 for label in labels if label) / len(labels)
        if 0.0 < positive_agreement < 1.0:
            disagreement_rows += 1

    for left, right in combinations(model_aliases, 2):
        matches = total = 0
        for _, row in df.iterrows():
            left_label = _to_bool_or_none(row.get(f"{left}_relevant"))
            right_label = _to_bool_or_none(row.get(f"{right}_relevant"))
            if left_label is None or right_label is None:
                continue
            total += 1
            if left_label == right_label:
                matches += 1
        if total:
            pairwise_values.append(matches / total)

    mean_pairwise = sum(pairwise_values) / len(pairwise_values) if pairwise_values else None
    return {
        "rows_with_votes": row_count,
        "full_model_agreement_rows": full_agreement_rows,
        "full_model_agreement_pct": round(
            (full_agreement_rows / row_count) * 100, 2
        ) if row_count else None,
        "mean_pairwise_agreement": round(mean_pairwise, 4) if mean_pairwise is not None else None,
        "disagreement_rate": round(disagreement_rows / row_count, 4) if row_count else None,
    }


def _evaluate_candidate(
    merged_df: pd.DataFrame,
    model_aliases: Sequence[str],
    settings: Dict[str, Any],
    ece_bins: int,
) -> Dict[str, Any]:
    scored_rows: List[Dict[str, Any]] = []
    for _, row in merged_df.iterrows():
        predictions = _extract_prediction_records(row.to_dict(), model_aliases, settings)
        if not predictions:
            continue
        aggregate = _aggregate_prediction_records(predictions, settings)
        scored_rows.append(
            {
                "y_true": bool(row["_reference_label"]),
                "prediction": aggregate["final_label"] == "relevant",
                "probability": float(aggregate["final_relevance_probability"]),
                "positive_agreement": float(aggregate["final_positive_agreement"]),
            }
        )

    if not scored_rows:
        return {
            "n_labeled_rows": len(merged_df),
            "n_scored_rows": 0,
            "n_unscored_rows": len(merged_df),
            "metrics": None,
            "brier_score": None,
            "ece": None,
            "auprc": None,
            "agreement": _agreement_summary(merged_df, model_aliases),
        }

    y_true = [row["y_true"] for row in scored_rows]
    y_pred = [row["prediction"] for row in scored_rows]
    probabilities = [row["probability"] for row in scored_rows]
    positive_agreements = [row["positive_agreement"] for row in scored_rows]

    counts = _confusion_counts(y_true, y_pred)
    metrics = _classification_metrics_from_counts(counts)
    evidence_support = None
    attribution_correct = None
    if "evidence_support" in merged_df.columns:
        vals = merged_df["evidence_support"].map(_to_bool_or_none).dropna().tolist()
        if vals:
            evidence_support = round(sum(1 for value in vals if value) / len(vals), 4)
    if "attribution_correct" in merged_df.columns:
        vals = merged_df["attribution_correct"].map(_to_bool_or_none).dropna().tolist()
        if vals:
            attribution_correct = round(sum(1 for value in vals if value) / len(vals), 4)

    return {
        "n_labeled_rows": len(merged_df),
        "n_scored_rows": len(scored_rows),
        "n_unscored_rows": len(merged_df) - len(scored_rows),
        "metrics": metrics,
        "brier_score": _brier_score(probabilities, y_true),
        "ece": _expected_calibration_error(probabilities, y_true, n_bins=ece_bins),
        "auprc": _auprc(
            probabilities,
            y_true,
            positive_agreements,
            min_agreement=float(settings.get("min_agreement", 0.0)),
            require_min_agreement=bool(settings.get("require_min_agreement", True)),
        ),
        "agreement": _agreement_summary(merged_df, model_aliases),
        "evidence_support_rate": evidence_support,
        "attribution_correct_rate": attribution_correct,
    }


def _model_validation_summaries(
    merged_df: pd.DataFrame,
    model_aliases: Sequence[str],
    confidence_semantics: str,
    ece_bins: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for alias in model_aliases:
        scored: List[Dict[str, Any]] = []
        for _, row in merged_df.iterrows():
            relevant = _to_bool_or_none(row.get(f"{alias}_relevant"))
            if relevant is None:
                continue
            probability = _coerce_relevance_probability(
                relevant,
                _to_float_or_none(row.get(f"{alias}_confidence")),
                confidence_semantics,
            )
            scored.append(
                {
                    "y_true": bool(row["_reference_label"]),
                    "prediction": relevant,
                    "probability": probability,
                }
            )
        if not scored:
            continue
        y_true = [item["y_true"] for item in scored]
        y_pred = [item["prediction"] for item in scored]
        probabilities = [item["probability"] for item in scored]
        counts = _confusion_counts(y_true, y_pred)
        metrics = _classification_metrics_from_counts(counts)
        rows.append(
            {
                "alias": alias,
                **metrics,
                "brier_score": _brier_score(probabilities, y_true),
                "ece": _expected_calibration_error(probabilities, y_true, ece_bins),
                "n_scored_rows": len(scored),
            }
        )
    return rows


def _choose_aggregation_settings(
    classified_df: pd.DataFrame,
    cls_cfg: Dict[str, Any],
    model_aliases: Sequence[str],
    model_weight_defaults: Dict[str, float],
    logger=None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    settings = _default_aggregation_settings(cls_cfg, model_aliases, model_weight_defaults)
    validation_cfg = cls_cfg.get("validation", {})
    payload: Dict[str, Any] = {
        "enabled": bool(validation_cfg.get("enabled", False)),
        "labels_available": False,
        "settings_source": settings["source"],
    }

    labels_df = _load_reference_labels(validation_cfg, logger=logger)
    if labels_df is None:
        return settings, payload

    merged_df = classified_df.merge(labels_df, on="canonical_url", how="inner")
    if merged_df.empty:
        if logger is not None:
            logger.warning("Validation labels did not match any classified canonical_url values.")
        return settings, payload

    threshold_grid = validation_cfg.get("threshold_grid") or [settings["threshold"]]
    min_agreement_grid = validation_cfg.get("min_agreement_grid") or [settings["min_agreement"]]
    recall_floor = float(validation_cfg.get("recall_floor", 0.75))
    tune_weights = bool(validation_cfg.get("tune_weights", True))
    weight_grid_step = float(validation_cfg.get("weight_grid_step", 0.25))
    ece_bins = int(validation_cfg.get("ece_bins", 10))
    weight_candidates = _candidate_weight_maps(
        model_aliases,
        settings["weights"],
        tune_weights=tune_weights,
        step=weight_grid_step,
    )

    best_settings = settings
    best_eval: Dict[str, Any] | None = None
    best_key: Tuple[Any, ...] | None = None
    candidates_evaluated = 0

    for weights in weight_candidates:
        for threshold in threshold_grid:
            for min_agreement in min_agreement_grid:
                candidate = {
                    **settings,
                    "weights": weights,
                    "threshold": float(threshold),
                    "min_agreement": float(min_agreement),
                    "source": "validation_tuned",
                }
                evaluation = _evaluate_candidate(
                    merged_df,
                    model_aliases=model_aliases,
                    settings=candidate,
                    ece_bins=ece_bins,
                )
                candidates_evaluated += 1
                metrics = evaluation.get("metrics") or {}
                precision = float(metrics.get("precision", 0.0))
                recall = float(metrics.get("recall", 0.0))
                f1 = float(metrics.get("f1", 0.0))
                constraint_ok = recall >= recall_floor
                key = (
                    1 if constraint_ok else 0,
                    precision if constraint_ok else f1,
                    f1,
                    recall,
                    float(threshold),
                    float(min_agreement),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_settings = candidate
                    best_eval = evaluation

    model_summaries = _model_validation_summaries(
        merged_df,
        model_aliases,
        confidence_semantics=str(best_settings.get("confidence_semantics", "auto")),
        ece_bins=ece_bins,
    )
    best_single = {}
    if model_summaries:
        best_single = {
            metric: max(float(summary.get(metric, 0.0)) for summary in model_summaries)
            for metric in ["precision", "recall", "f1"]
        }

    ensemble_metrics = (best_eval or {}).get("metrics") or {}
    payload = {
        "enabled": True,
        "labels_available": True,
        "labels_rows": int(len(labels_df)),
        "matched_rows": int(len(merged_df)),
        "settings_source": best_settings["source"],
        "recall_floor": recall_floor,
        "candidates_evaluated": candidates_evaluated,
        "selected_metrics": best_eval,
        "per_model": model_summaries,
        "ensemble_gain": {
            metric: round(float(ensemble_metrics.get(metric, 0.0)) - float(best_single.get(metric, 0.0)), 4)
            for metric in ["precision", "recall", "f1"]
        }
        if best_single
        else {},
    }
    return best_settings, payload


def _model_run_summary(classified_df: pd.DataFrame, model_aliases: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for alias in model_aliases:
        probability_col = f"{alias}_relevance_probability"
        if probability_col in classified_df.columns:
            avg_probability = pd.to_numeric(
                classified_df[probability_col], errors="coerce"
            ).mean()
        else:
            avg_probability = float("nan")
        rows.append(
            {
                "alias": alias,
                "model_id": text_or_empty(
                    classified_df.get(f"{alias}_model_id", pd.Series(dtype=str)).dropna().iloc[0]
                    if f"{alias}_model_id" in classified_df.columns
                    and not classified_df[f"{alias}_model_id"].dropna().empty
                    else ""
                ),
                "relevant_true": int((classified_df.get(f"{alias}_relevant") == True).sum())
                if f"{alias}_relevant" in classified_df.columns
                else 0,
                "relevant_false": int((classified_df.get(f"{alias}_relevant") == False).sum())
                if f"{alias}_relevant" in classified_df.columns
                else 0,
                "avg_relevance_probability": round(float(avg_probability), 4)
                if pd.notna(avg_probability)
                else None,
                "error_rows": int(
                    classified_df.get(f"{alias}_error", pd.Series(dtype=str))
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .ne("")
                    .sum()
                )
                if f"{alias}_error" in classified_df.columns
                else 0,
            }
        )
    return rows


def _build_metrics_payload(
    classified_df: pd.DataFrame,
    cls_cfg: Dict[str, Any],
    settings: Dict[str, Any],
    validation_payload: Dict[str, Any],
    model_aliases: Sequence[str],
) -> Dict[str, Any]:
    final_counts = {}
    if "final_label" in classified_df.columns:
        final_counts = {
            str(key): int(value)
            for key, value in classified_df["final_label"].value_counts(dropna=False).to_dict().items()
        }
    return {
        "generated_at": utc_now_iso(),
        "target_name": _target_name(cls_cfg),
        "aggregation": {
            "source": settings.get("source"),
            "threshold": round(float(settings.get("threshold", 0.75)), 4),
            "min_agreement": round(float(settings.get("min_agreement", 0.0)), 4),
            "require_min_agreement": bool(settings.get("require_min_agreement", True)),
            "confidence_semantics": settings.get("confidence_semantics", "auto"),
            "weights": settings.get("weights", {}),
        },
        "descriptive": {
            "rows": int(len(classified_df)),
            "final_label_counts": final_counts,
            **_agreement_summary(classified_df, model_aliases),
        },
        "models": _model_run_summary(classified_df, model_aliases),
        "validation": validation_payload,
    }


def _metrics_output_path(output_path: Path, cls_cfg: Dict[str, Any]) -> Path:
    configured = (
        text_or_empty(cls_cfg.get("metrics_output_path")).strip()
        or text_or_empty(cls_cfg.get("validation", {}).get("metrics_output_path")).strip()
    )
    if configured:
        return Path(configured)
    return output_path.with_name("classification_metrics.json")


def classify_articles_ensemble(
    extracted_df: pd.DataFrame,
    client: OpenAIChatClient,
    cls_cfg: Dict[str, Any],
    output_path: Path,
    force: bool = False,
    logger=None,
) -> pd.DataFrame:
    """Run configurable ensemble classification and write CSV output."""
    resume_if_exists = bool(cls_cfg.get("resume_if_exists", True))
    checkpoint_every = int(cls_cfg.get("checkpoint_every", 20))
    inter_article_delay_seconds = float(cls_cfg.get("inter_article_delay_seconds", 0.0))
    per_model_parallel_requests = max(
        1, int(cls_cfg.get("per_model_parallel_requests", 1))
    )
    gen_cfg = cls_cfg.get("generation", {})
    quality_cfg = cls_cfg.get("rationale_quality", {})
    models = cls_cfg.get("models", [])
    model_aliases = [
        sanitize_alias(model_cfg.get("alias", model_cfg.get("model_id", "model")))
        for model_cfg in models
    ]
    model_weight_defaults = {
        sanitize_alias(model_cfg.get("alias", model_cfg.get("model_id", "model"))): float(model_cfg.get("weight", 1.0))
        for model_cfg in models
    }
    target_name = _target_name(cls_cfg)
    target_aliases = _target_aliases(cls_cfg)

    existing_by_url: Dict[str, Dict[str, Any]] = {}
    if output_path.exists() and not force:
        if not resume_if_exists:
            if logger is not None:
                logger.info(
                    "Classification output exists; skipping (use --force to recompute): %s",
                    output_path,
                )
            return pd.read_csv(output_path, low_memory=False)

        existing_df = pd.read_csv(output_path, low_memory=False)
        for rec in existing_df.to_dict(orient="records"):
            canonical_url = text_or_empty(rec.get("canonical_url")).strip()
            if canonical_url:
                existing_by_url[canonical_url] = rec
        if logger is not None:
            complete_rows = sum(
                1
                for rec in existing_by_url.values()
                if _is_complete_classification_row(rec, model_aliases=model_aliases)
            )
            partial_rows = len(existing_by_url) - complete_rows
            logger.info(
                "Resuming classification from existing output: loaded_rows=%s complete_rows=%s partial_rows=%s",
                len(existing_by_url),
                complete_rows,
                partial_rows,
            )

    rows: List[Dict[str, Any]] = []
    pending_items: List[Dict[str, Any]] = []
    total_rows = len(extracted_df)

    for _, row in extracted_df.iterrows():
        canonical_url = text_or_empty(row.get("canonical_url"))
        title = text_or_empty(row.get("extracted_title")).strip()
        text = text_or_empty(row.get("extracted_text")).strip()
        text_hash = sha256_text(text) if text else ""

        result: Dict[str, Any] = dict(existing_by_url.get(canonical_url, {}))
        result["canonical_url"] = canonical_url
        result["extracted_title"] = title
        result["text_hash"] = text_or_empty(result.get("text_hash")) or text_hash
        result["classified_at"] = text_or_empty(result.get("classified_at")) or utc_now_iso()
        result["final_label"] = text_or_empty(result.get("final_label")) or "undetermined"
        try:
            result["final_confidence"] = float(result.get("final_confidence", 0.0))
        except Exception:
            result["final_confidence"] = 0.0
        result["final_relevance_probability"] = result.get("final_relevance_probability", 0.0)
        result["final_positive_agreement"] = result.get("final_positive_agreement", 0.0)
        result["final_supporting_models"] = result.get("final_supporting_models", "")
        result["final_evidence_spans"] = result.get("final_evidence_spans", "")
        result["final_evidence_present"] = result.get("final_evidence_present", False)

        for model_cfg in models:
            alias = sanitize_alias(
                model_cfg.get("alias", model_cfg.get("model_id", "model"))
            )
            result[f"{alias}_model_id"] = model_cfg.get("model_id", "")
            result[f"{alias}_relevant"] = result.get(f"{alias}_relevant", "")
            result[f"{alias}_confidence"] = result.get(f"{alias}_confidence", "")
            result[f"{alias}_relevance_probability"] = result.get(
                f"{alias}_relevance_probability", ""
            )
            result[f"{alias}_rationale"] = result.get(f"{alias}_rationale", "")
            result[f"{alias}_evidence_spans"] = result.get(f"{alias}_evidence_spans", "")
            result[f"{alias}_error"] = result.get(f"{alias}_error", "")

        missing_reasons: List[str] = []
        if not title:
            missing_reasons.append("missing_extracted_title")
        if not text:
            missing_reasons.append("missing_extracted_text")

        if missing_reasons:
            reason = "skipped_" + "+".join(missing_reasons)
            for model_cfg in models:
                alias = sanitize_alias(
                    model_cfg.get("alias", model_cfg.get("model_id", "model"))
                )
                result[f"{alias}_error"] = reason
            result["final_label"] = "skipped"
            result["final_confidence"] = 0.0
            result["final_relevance_probability"] = 0.0
            result["final_positive_agreement"] = 0.0
            result["final_supporting_models"] = ""
            result["final_evidence_spans"] = ""
            result["final_evidence_present"] = False
            rows.append(result)
            continue

        prep_title, prep_text = choose_extraction_input(
            title=title,
            text=text,
            max_chars=int(cls_cfg.get("text_max_chars", 12000)),
            summary_cfg=cls_cfg.get("deterministic_summary", {}),
        )
        prompt = _build_user_prompt(prep_title, prep_text, target_name=target_name)

        rows.append(result)
        pending_items.append(
            {
                "row_idx": len(rows) - 1,
                "canonical_url": canonical_url,
                "prompt": prompt,
            }
        )

    if logger is not None:
        precompleted = sum(
            1 for rec in rows if _is_complete_classification_row(rec, model_aliases)
        )
        logger.info(
            "Prepared classification rows: total_input=%s pending_for_llm=%s precompleted=%s",
            total_rows,
            len(pending_items),
            precompleted,
        )

    for model_cfg in models:
        alias = sanitize_alias(
            model_cfg.get("alias", model_cfg.get("model_id", "model"))
        )
        model_id = model_cfg.get("model_id", "")
        model_done = 0
        model_pending_items: List[Dict[str, Any]] = []
        model_skipped_existing = 0

        for item in pending_items:
            target = rows[item["row_idx"]]
            rel_val = _to_bool_or_none(target.get(f"{alias}_relevant"))
            err_val = text_or_empty(target.get(f"{alias}_error")).strip()
            if rel_val is not None or err_val:
                model_skipped_existing += 1
                continue
            model_pending_items.append(item)

        if logger is not None:
            logger.info(
                "Model start [%s]: pending=%s skipped_existing=%s",
                alias,
                len(model_pending_items),
                model_skipped_existing,
            )

        if not model_pending_items:
            continue

        def _apply_model_result(item: Dict[str, Any], parsed, error) -> None:
            target = rows[item["row_idx"]]
            if parsed is not None:
                probability = _coerce_relevance_probability(
                    bool(parsed["relevant"]),
                    _to_float_or_none(parsed.get("confidence")),
                    str(
                        cls_cfg.get("aggregation", {}).get("confidence_semantics", "auto")
                    ).strip().lower(),
                )
                target[f"{alias}_relevant"] = parsed["relevant"]
                target[f"{alias}_confidence"] = parsed["confidence"]
                target[f"{alias}_relevance_probability"] = round(probability, 4)
                target[f"{alias}_rationale"] = parsed["rationale"]
                target[f"{alias}_evidence_spans"] = " | ".join(parsed["evidence_spans"])
                target[f"{alias}_error"] = error
                if error and logger is not None:
                    logger.warning(
                        "Model quality warning for %s on %s: %s",
                        model_id,
                        item["canonical_url"],
                        error,
                    )
            else:
                target[f"{alias}_error"] = error
                if logger is not None:
                    logger.warning(
                        "Model call failed for %s on %s: %s",
                        model_id,
                        item["canonical_url"],
                        error,
                    )

        if per_model_parallel_requests > 1 and len(model_pending_items) > 1:
            with ThreadPoolExecutor(
                max_workers=min(per_model_parallel_requests, len(model_pending_items))
            ) as executor:
                future_to_item = {
                    executor.submit(
                        _invoke_single_model,
                        client,
                        model_id,
                        item["prompt"],
                        gen_cfg,
                        quality_cfg,
                        target_name,
                        target_aliases,
                    ): item
                    for item in model_pending_items
                }

                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    parsed, error = future.result()
                    _apply_model_result(item, parsed, error)

                    model_done += 1
                    if logger is not None and (
                        model_done % 20 == 0 or model_done == len(model_pending_items)
                    ):
                        logger.info(
                            "Model progress [%s]: %s/%s",
                            alias,
                            model_done,
                            len(model_pending_items),
                        )

                    if checkpoint_every > 0 and model_done % checkpoint_every == 0:
                        _write_checkpoint(rows, output_path)
                        if logger is not None:
                            logger.info("Classification checkpoint written: %s", output_path)

                    if inter_article_delay_seconds > 0:
                        time.sleep(inter_article_delay_seconds)
        else:
            for item in model_pending_items:
                parsed, error = _invoke_single_model(
                    client=client,
                    model_id=model_id,
                    prompt=item["prompt"],
                    gen_cfg=gen_cfg,
                    quality_cfg=quality_cfg,
                    target_name=target_name,
                    target_aliases=target_aliases,
                )
                _apply_model_result(item, parsed, error)

                model_done += 1
                if logger is not None and (
                    model_done % 20 == 0 or model_done == len(model_pending_items)
                ):
                    logger.info(
                        "Model progress [%s]: %s/%s",
                        alias,
                        model_done,
                        len(model_pending_items),
                    )

                if checkpoint_every > 0 and model_done % checkpoint_every == 0:
                    _write_checkpoint(rows, output_path)
                    if logger is not None:
                        logger.info("Classification checkpoint written: %s", output_path)

                if inter_article_delay_seconds > 0:
                    time.sleep(inter_article_delay_seconds)

    pre_agg_df = pd.DataFrame(rows)
    aggregation_settings, validation_payload = _choose_aggregation_settings(
        pre_agg_df,
        cls_cfg=cls_cfg,
        model_aliases=model_aliases,
        model_weight_defaults=model_weight_defaults,
        logger=logger,
    )

    for row in rows:
        if text_or_empty(row.get("final_label")) == "skipped":
            continue
        predictions = _extract_prediction_records(
            row,
            model_aliases=model_aliases,
            settings=aggregation_settings,
        )
        aggregated = _aggregate_prediction_records(predictions, settings=aggregation_settings)
        row.update(aggregated)
        row["classified_at"] = utc_now_iso()

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)

    metrics_payload = _build_metrics_payload(
        out_df,
        cls_cfg=cls_cfg,
        settings=aggregation_settings,
        validation_payload=validation_payload,
        model_aliases=model_aliases,
    )
    metrics_path = _metrics_output_path(output_path, cls_cfg)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    if logger is not None:
        logger.info(
            "Selected aggregation settings: source=%s threshold=%.3f min_agreement=%.3f weights=%s",
            aggregation_settings.get("source"),
            float(aggregation_settings.get("threshold", 0.0)),
            float(aggregation_settings.get("min_agreement", 0.0)),
            aggregation_settings.get("weights", {}),
        )
        logger.info("Classification output written: %s", output_path)
        logger.info("Classification metrics written: %s", metrics_path)
        logger.info("Classification progress: %s/%s", len(rows), total_rows)
    return out_df
