"""LLM output parsing and deterministic input shaping helpers."""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.utils.common import normalize_whitespace


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def _heuristic_parse_model_output(text: str) -> Optional[Dict[str, Any]]:
    src = _strip_code_fences(str(text))
    if not src.strip():
        return None

    relevant: Optional[bool] = None
    rel_match = re.search(
        r"""(?is)["']?\s*relevant\s*["']?\s*[:=]\s*["']?\s*(true|false|yes|no|1|0|relevant|not[\s_-]*relevant)\s*["']?""",
        src,
    )
    if rel_match:
        token = rel_match.group(1).strip().lower().replace("_", " ").replace("-", " ")
        if token in {"true", "yes", "1", "relevant"}:
            relevant = True
        elif token in {"false", "no", "0", "not relevant"}:
            relevant = False

    confidence = 0.0
    conf_match = re.search(
        r"""(?is)["']?\s*confidence\s*["']?\s*[:=]\s*["']?([0-9]*\.?[0-9]+)\s*["']?""",
        src,
    )
    if conf_match:
        try:
            confidence = float(conf_match.group(1))
        except Exception:
            confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    rationale = ""
    rat_match = re.search(r'(?is)["\']?\s*rationale\s*["\']?\s*[:=]\s*"([^"]*)"', src)
    if not rat_match:
        rat_match = re.search(r"""(?is)["']?\s*rationale\s*["']?\s*[:=]\s*'([^']*)'""", src)
    if not rat_match:
        rat_match = re.search(r"""(?is)["']?\s*rationale\s*["']?\s*[:=]\s*["']?([^,\n\]}]+)""", src)
    if not rat_match:
        rat_match = re.search(r"""(?is)["']?\s*rationale\s*["']?\s*[:=]\s*["']?(.+)$""", src)
    if rat_match:
        rationale = normalize_whitespace(rat_match.group(1).strip(" \"'"))

    evidence_spans: List[str] = []
    ev_match = re.search(r"""(?is)["']?\s*evidence_spans?\s*["']?\s*[:=]\s*\[(.*?)\]""", src)
    if ev_match:
        evidence_blob = ev_match.group(1)
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', evidence_blob)
        for a, b in quoted:
            value = normalize_whitespace(a or b)
            if value:
                evidence_spans.append(value)

    if relevant is None and not rationale and not evidence_spans and not conf_match:
        return None

    return {
        "relevant": bool(relevant) if relevant is not None else False,
        "confidence": confidence,
        "rationale": rationale,
        "evidence_spans": evidence_spans,
    }


def _repair_truncated_json_candidate(candidate: str) -> str:
    text = (candidate or "").strip()
    if not text:
        return text

    if text.endswith(","):
        text = text[:-1].rstrip()

    unescaped_quote_count = len(re.findall(r'(?<!\\)"', text))
    if unescaped_quote_count % 2 == 1:
        text += '"'

    open_sq = text.count("[") - text.count("]")
    if open_sq > 0:
        text += "]" * open_sq

    open_curly = text.count("{") - text.count("}")
    if open_curly > 0:
        text += "}" * open_curly

    return text


def parse_model_json(raw_text: str) -> Dict[str, Any]:
    """Parse model JSON robustly and normalize expected fields."""
    if raw_text is None:
        raise ValueError("Empty model output")

    candidate = _extract_json_object(_strip_code_fences(str(raw_text)))
    repaired = _repair_truncated_json_candidate(candidate)
    attempts = [
        candidate,
        repaired,
        re.sub(r",\s*([}\]])", r"\1", candidate),
        re.sub(r'(\btrue\b|\bfalse\b|\bnull\b|[}\]0-9"])\s*(?="[^"]+"\s*:)', r"\1, ", candidate, flags=re.IGNORECASE),
    ]

    parsed: Optional[Dict[str, Any]] = None
    for attempt in attempts:
        try:
            maybe = json.loads(attempt)
            if isinstance(maybe, dict):
                parsed = maybe
                break
        except Exception:
            continue

    if parsed is None:
        try:
            maybe = ast.literal_eval(candidate)
            if isinstance(maybe, dict):
                parsed = maybe
        except Exception as exc:
            heuristic = _heuristic_parse_model_output(raw_text)
            if heuristic is not None:
                parsed = heuristic
            else:
                raise ValueError(f"Could not parse model JSON: {exc}") from exc

    if parsed is None:
        heuristic = _heuristic_parse_model_output(raw_text)
        if heuristic is not None:
            parsed = heuristic
        else:
            raise ValueError("Could not parse model JSON")

    relevant_raw = parsed.get("relevant", False)
    if isinstance(relevant_raw, bool):
        relevant = relevant_raw
    elif isinstance(relevant_raw, (int, float)):
        relevant = bool(relevant_raw)
    else:
        relevant_text = str(relevant_raw).strip().lower()
        relevant = relevant_text in {"true", "yes", "1", "relevant"}

    confidence_raw = parsed.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    rationale = normalize_whitespace(str(parsed.get("rationale", "")))

    evidence = parsed.get("evidence_spans", [])
    if isinstance(evidence, str):
        evidence_spans = [normalize_whitespace(evidence)] if evidence.strip() else []
    elif isinstance(evidence, list):
        evidence_spans = [normalize_whitespace(str(x)) for x in evidence if str(x).strip()]
    else:
        evidence_spans = []

    return {
        "relevant": relevant,
        "confidence": confidence,
        "rationale": rationale,
        "evidence_spans": evidence_spans,
    }


def sanitize_alias(value: str) -> str:
    """Convert model aliases to safe column suffixes."""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def split_sentences(text: str) -> List[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def deterministic_extractive_summary(
    text: str,
    max_chars: int = 6000,
    max_sentences: int = 20,
    keywords: Optional[Sequence[str]] = None,
) -> str:
    """Deterministic extractive summary for cost control."""
    sentences = split_sentences(text)
    if not sentences:
        return text[:max_chars]

    key_terms = [k.lower() for k in (keywords or [])]

    scored: List[Tuple[float, int, str]] = []
    for idx, sentence in enumerate(sentences):
        lowered = sentence.lower()
        kw_hits = sum(lowered.count(k) for k in key_terms)
        punctuation = len(re.findall(r"[,;:]", sentence))
        length_bonus = min(len(sentence), 260) / 260.0
        score = (kw_hits * 2.0) + (punctuation * 0.15) + length_bonus
        scored.append((score, idx, sentence))

    scored.sort(key=lambda x: (-x[0], x[1]))
    selected = sorted(scored[:max_sentences], key=lambda x: x[1])

    out_parts: List[str] = []
    total = 0
    for _, _, sentence in selected:
        add_len = len(sentence) + 1
        if total + add_len > max_chars:
            break
        out_parts.append(sentence)
        total += add_len

    if not out_parts:
        return text[:max_chars]

    return " ".join(out_parts).strip()


def choose_extraction_input(
    title: str,
    text: str,
    max_chars: int,
    summary_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """Prepare deterministic model input for classification."""
    summary_cfg = summary_cfg or {}
    use_summary = bool(summary_cfg.get("enabled", False))

    clean_title = normalize_whitespace(title or "")
    clean_text = normalize_whitespace(text or "")

    if use_summary and clean_text:
        summary = deterministic_extractive_summary(
            clean_text,
            max_chars=int(summary_cfg.get("max_chars", max_chars)),
            max_sentences=int(summary_cfg.get("max_sentences", 20)),
            keywords=summary_cfg.get(
                "keywords",
                [
                    "zika",
                    "microcefalia",
                    "microcephaly",
                    "gestante",
                    "pregnancy",
                    "economia",
                    "economy",
                    "saude",
                    "health",
                    "inequality",
                    "policy",
                    "travel",
                ],
            ),
        )
        return clean_title, summary[:max_chars]

    return clean_title, clean_text[:max_chars]
