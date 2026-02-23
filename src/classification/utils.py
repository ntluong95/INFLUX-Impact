"""Utility helpers for the classification pipeline."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import pandas as pd
import requests

TRACKING_PARAMS_DEFAULT = {
    "gclid",
    "fbclid",
    "dclid",
    "msclkid",
    "mc_cid",
    "mc_eid",
    "igshid",
    "vero_conv",
    "vero_id",
    "yclid",
    "gbraid",
    "wbraid",
    "mkt_tok",
    "cmpid",
    "spm",
    "ref",
    "ref_src",
    "sourceid",
}

URL_COLUMN_PRIORITY = [
    "canonical_url",
    "url",
    "item_link",
    "item_url",
    "link",
    "article_url",
    "permalink",
    "href",
]

TITLE_COLUMN_PRIORITY = [
    "rss_title",
    "item_title",
    "title",
    "headline",
    "news_title",
]

DESCRIPTION_COLUMN_PRIORITY = [
    "rss_description",
    "item_description",
    "description",
    "summary",
    "snippet",
    "abstract",
]

DATE_COLUMN_PRIORITY = [
    "pub_date",
    "item_pub_date",
    "published_at",
    "publish_date",
    "date",
    "datetime",
    "item_date",
]

GOOGLE_NEWS_HOST_RE = re.compile(r"(^|\.)news\.google\.[a-z.]+$", flags=re.IGNORECASE)
GOOGLE_NEWS_ARTICLE_RE = re.compile(r"/(?:rss/)?articles/([^/?#]+)", flags=re.IGNORECASE)


def ensure_dir(path: Path) -> None:
    """Create a directory (and parents) if missing."""
    path.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    """Current UTC timestamp as ISO-8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_text(value: str) -> str:
    """Stable SHA-256 hash of a string."""
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def normalize_whitespace(value: str) -> str:
    """Collapse repeated whitespace and trim."""
    return re.sub(r"\s+", " ", value).strip()


def is_probable_url(value: Any) -> bool:
    """Fast URL heuristic for schema inference."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    text = str(value).strip()
    return bool(re.match(r"^https?://", text, flags=re.IGNORECASE))


def _is_tracking_param(name: str, tracking_params: Optional[Iterable[str]] = None) -> bool:
    n = name.lower().strip()
    tracking = set(p.lower() for p in (tracking_params or TRACKING_PARAMS_DEFAULT))
    return n.startswith("utm_") or n in tracking


def canonicalize_url(
    url: Any,
    tracking_params: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """Canonicalize URLs for stable deduplication.

    Rules:
    - lowercase scheme/host
    - remove fragments
    - remove tracking query params
    - normalize trailing slash
    - sort query params
    """
    if url is None or (isinstance(url, float) and pd.isna(url)):
        return None

    raw = str(url).strip()
    if not raw:
        return None

    if raw.startswith("//"):
        raw = f"https:{raw}"
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw):
        if raw.startswith("www."):
            raw = f"https://{raw}"
        else:
            return None

    try:
        parsed = urlsplit(raw)
    except Exception:
        return None

    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        return None

    host = (parsed.hostname or "").lower()
    if not host:
        return None

    port = parsed.port
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"
    else:
        netloc = host

    path = parsed.path or "/"
    path = re.sub(r"/{2,}", "/", path)
    if path != "/":
        path = path.rstrip("/") or "/"

    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    filtered_pairs = [
        (k, v)
        for (k, v) in query_pairs
        if not _is_tracking_param(k, tracking_params=tracking_params)
    ]
    filtered_pairs.sort(key=lambda kv: (kv[0], kv[1]))
    normalized_query = urlencode(filtered_pairs, doseq=True)

    canonical = urlunsplit((scheme, netloc, path, normalized_query, ""))
    return canonical


def is_google_news_wrapper_url(url: Any) -> bool:
    if not url:
        return False
    try:
        parsed = urlsplit(str(url).strip())
    except Exception:
        return False
    host = (parsed.hostname or "").lower()
    if not GOOGLE_NEWS_HOST_RE.search(host):
        return False
    return bool(GOOGLE_NEWS_ARTICLE_RE.search(parsed.path or ""))


def extract_google_news_article_id(url: Any) -> Optional[str]:
    if not url:
        return None
    try:
        parsed = urlsplit(str(url).strip())
    except Exception:
        return None
    host = (parsed.hostname or "").lower()
    if not GOOGLE_NEWS_HOST_RE.search(host):
        return None
    m = GOOGLE_NEWS_ARTICLE_RE.search(parsed.path or "")
    if not m:
        return None
    token = (m.group(1) or "").strip()
    return token or None


def _decode_varint(data: bytes, offset: int = 0) -> Tuple[Optional[int], int]:
    value = 0
    shift = 0
    pos = offset
    for _ in range(10):
        if pos >= len(data):
            return None, offset
        b = data[pos]
        pos += 1
        value |= (b & 0x7F) << shift
        if not (b & 0x80):
            return value, pos
        shift += 7
    return None, offset


def _decode_google_news_article_id_offline(article_id: str) -> Optional[str]:
    if not article_id:
        return None

    try:
        payload = base64.urlsafe_b64decode(article_id + "=" * (-len(article_id) % 4))
    except Exception:
        return None

    trimmed = payload
    if trimmed.startswith(b"\x08\x13\x22"):
        trimmed = trimmed[3:]
    elif trimmed.startswith(b"\x08\x13"):
        trimmed = trimmed[2:]

    if trimmed.endswith(b"\xd2\x01\x00"):
        trimmed = trimmed[:-3]

    length, pos = _decode_varint(trimmed, 0)
    if length is not None and pos + length <= len(trimmed):
        candidate = trimmed[pos : pos + length].decode("utf-8", errors="ignore").strip()
        if re.match(r"^https?://", candidate, flags=re.IGNORECASE):
            return candidate

    decoded = trimmed.decode("utf-8", errors="ignore")
    m = re.search(r"https?://[^\s\"'<>\\]+", decoded, flags=re.IGNORECASE)
    if m:
        return m.group(0).strip()
    return None


def _extract_google_news_sig_ts(html: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    id_match = re.search(r'data-n-a-id="([^"]+)"', html)
    ts_match = re.search(r'data-n-a-ts="([^"]+)"', html)
    sg_match = re.search(r'data-n-a-sg="([^"]+)"', html)
    article_id = id_match.group(1).strip() if id_match else None
    ts = ts_match.group(1).strip() if ts_match else None
    sg = sg_match.group(1).strip() if sg_match else None
    return article_id, ts, sg


def _build_google_news_batchexecute_payload(article_id: str, ts: str, sg: str) -> str:
    inner = (
        '["garturlreq",[['
        '"en-US","US",["FINANCE_TOP_INDICES","WEB_TEST_1_0_0"],null,null,1,1,"US:en",'
        'null,180,null,null,null,null,null,0,null,null,[1608992183,723341000]],'
        '"en-US","US",1,[2,3,4,8],1,0,"655000234",0,0,null,0],'
        f'"{article_id}",{ts},"{sg}"]'
    )
    escaped_inner = inner.replace("\\", "\\\\").replace('"', '\\"')
    req = f'[[["Fbv4je","{escaped_inner}",null,"generic"]]]'
    return f"f.req={quote(req)}"


def _decode_google_news_article_id_online(
    wrapper_url: str,
    article_id: str,
    timeout_seconds: int = 20,
    session: Optional[requests.Session] = None,
) -> Tuple[Optional[str], str]:
    own_session = session is None
    client = session or requests.Session()

    try:
        wrapper_resp = client.get(
            wrapper_url,
            timeout=timeout_seconds,
            allow_redirects=True,
        )
        html = wrapper_resp.text or ""
        page_article_id, ts, sg = _extract_google_news_sig_ts(html)
        use_article_id = page_article_id or article_id

        if not use_article_id or not ts or not sg:
            return None, "missing_sig_ts"

        payload = _build_google_news_batchexecute_payload(use_article_id, ts, sg)
        batch_resp = client.post(
            "https://news.google.com/_/DotsSplashUi/data/batchexecute?rpcids=Fbv4je",
            headers={
                "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
                "Referer": "https://news.google.com/",
            },
            data=payload,
            timeout=timeout_seconds,
        )
        text = batch_resp.text or ""

        # Primary path: parse the batchexecute JSON line and then parse
        # the embedded payload for the Fbv4je RPC result.
        for line in text.splitlines():
            ln = line.strip()
            if not ln or ln.startswith(")]}'"):
                continue
            try:
                outer = json.loads(ln)
            except Exception:
                continue

            if not isinstance(outer, list):
                continue

            for entry in outer:
                if not (isinstance(entry, list) and len(entry) >= 3):
                    continue
                if entry[1] != "Fbv4je":
                    continue
                inner_raw = entry[2]
                if not isinstance(inner_raw, str):
                    continue
                try:
                    inner = json.loads(inner_raw)
                except Exception:
                    continue
                if not (isinstance(inner, list) and inner):
                    continue
                if inner[0] != "garturlres":
                    continue

                for candidate in inner[1:]:
                    if isinstance(candidate, str) and re.match(r"^https?://", candidate, flags=re.IGNORECASE):
                        return candidate.strip(), ""

        # Fallback: regex extraction for providers that tweak line format.
        m = re.search(
            r'garturlres\\",\\\"(https?:\\\\/\\\\/[^\\\"]+)',
            text,
            flags=re.IGNORECASE,
        )
        if m:
            decoded = m.group(1).replace("\\/", "/").strip()
            decoded = bytes(decoded, "utf-8").decode("unicode_escape").strip()
            if re.match(r"^https?://", decoded, flags=re.IGNORECASE):
                return decoded, ""

        return None, "missing_garturlres"
    except Exception as exc:
        return None, str(exc)
    finally:
        if own_session:
            client.close()


def decode_google_news_url(
    url: Any,
    timeout_seconds: int = 20,
    session: Optional[requests.Session] = None,
) -> Tuple[Optional[str], str, str]:
    """Resolve Google News wrapper URLs to publisher URLs.

    Returns (resolved_url, method, error).
    """
    if not url:
        return None, "invalid_input", "empty_url"

    raw_url = str(url).strip()
    article_id = extract_google_news_article_id(raw_url)
    if not article_id:
        return None, "not_google_wrapper", ""

    offline = _decode_google_news_article_id_offline(article_id)
    if offline:
        return offline, "google_decode_offline", ""

    online, err = _decode_google_news_article_id_online(
        wrapper_url=raw_url,
        article_id=article_id,
        timeout_seconds=timeout_seconds,
        session=session,
    )
    if online:
        return online, "google_decode_batchexecute", ""
    return None, "google_decode_failed", err


def is_google_consent_interstitial(html: str, fetched_url: str = "") -> bool:
    url_text = (fetched_url or "").lower()
    body = (html or "").lower()

    if "consent.google.com" in url_text:
        return True
    if "before you continue to google" in body:
        return True

    # Strong consent fingerprint on Google interstitial pages.
    if (
        "g.co/privacytools" in body
        and "cookies and data" in body
        and "accept all" in body
        and "reject all" in body
    ):
        return True

    return False


def _best_by_priority(columns: Sequence[str], priority_names: Sequence[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for name in priority_names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def _best_column_by_url_ratio(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    scored: List[Tuple[float, str]] = []
    for col in candidates:
        if col not in df.columns:
            continue
        sample = df[col].dropna().astype(str).head(300)
        if sample.empty:
            continue
        ratio = float(sample.apply(is_probable_url).mean())
        name_bonus = 0.0
        name_lower = col.lower()
        if "url" in name_lower or "link" in name_lower or "href" in name_lower:
            name_bonus = 0.15
        if "guid" in name_lower:
            name_bonus = 0.05
        scored.append((ratio + name_bonus, col))

    if not scored:
        return None
    scored.sort(reverse=True, key=lambda x: x[0])
    winner_score, winner_col = scored[0]
    if winner_score < 0.2:
        return None
    return winner_col


def _best_textual_column(
    df: pd.DataFrame,
    priority_names: Sequence[str],
    fallback_patterns: Sequence[str],
) -> Optional[str]:
    direct = _best_by_priority(df.columns, priority_names)
    if direct:
        return direct

    pattern = re.compile("|".join(re.escape(p) for p in fallback_patterns), flags=re.IGNORECASE)
    for col in df.columns:
        if pattern.search(col):
            return col
    return None


def _best_date_column(df: pd.DataFrame) -> Optional[str]:
    direct = _best_by_priority(df.columns, DATE_COLUMN_PRIORITY)
    if direct:
        return direct

    date_like = [c for c in df.columns if re.search(r"date|time|published", c, flags=re.IGNORECASE)]
    if not date_like:
        return None

    best_col = None
    best_ratio = -1.0
    for col in date_like:
        sample = df[col].dropna().astype(str).head(300)
        if sample.empty:
            continue
        parsed_any = pd.to_datetime(sample, errors="coerce", utc=False)
        ratio_any = float(parsed_any.notna().mean())
        parsed_epoch = pd.to_datetime(sample, errors="coerce", utc=False, unit="s")
        ratio_epoch = float(parsed_epoch.notna().mean())
        ratio = max(ratio_any, ratio_epoch)
        if ratio > best_ratio:
            best_ratio = ratio
            best_col = col
    return best_col


def detect_schema(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Infer key RSS fields from unknown schemas."""
    columns = list(df.columns)

    direct_url = _best_by_priority(columns, URL_COLUMN_PRIORITY)
    url_candidates = list(columns)
    inferred_url = _best_column_by_url_ratio(df, [direct_url] + url_candidates if direct_url else url_candidates)

    title_col = _best_textual_column(
        df,
        TITLE_COLUMN_PRIORITY,
        fallback_patterns=["title", "headline"],
    )
    description_col = _best_textual_column(
        df,
        DESCRIPTION_COLUMN_PRIORITY,
        fallback_patterns=["description", "summary", "snippet", "abstract"],
    )
    date_col = _best_date_column(df)

    return {
        "url_col": inferred_url,
        "title_col": title_col,
        "description_col": description_col,
        "date_col": date_col,
    }


def safe_read_csv(path: Path) -> pd.DataFrame:
    """Read CSV with fallback encodings."""
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    last_error: Optional[Exception] = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except Exception as exc:  # pragma: no cover - fallback behavior
            last_error = exc
    raise RuntimeError(f"Failed to read CSV at {path}: {last_error}")


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
    """Best-effort extraction when strict JSON parsing fails."""
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
        # Truncated output fallback: capture tail text after rationale key.
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

    # If a quoted string was cut off, close it.
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


def save_dataframe_with_parquet_fallback(df: pd.DataFrame, target_parquet: Path) -> Path:
    """Try saving parquet; fallback to CSV when parquet dependencies are missing."""
    try:
        df.to_parquet(target_parquet, index=False)
        return target_parquet
    except Exception:
        csv_path = target_parquet.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


def setup_logger(log_file: Path) -> logging.Logger:
    """Set up pipeline logger with file + console handlers."""
    ensure_dir(log_file.parent)
    logger = logging.getLogger("classification_pipeline")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def text_or_empty(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)
