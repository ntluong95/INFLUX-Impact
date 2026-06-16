"""URL canonicalization and Google wrapper resolution helpers."""

from __future__ import annotations

import base64
import json
import re
from typing import Any, Dict, Iterable, Optional, Tuple
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

GOOGLE_NEWS_HOST_RE = re.compile(r"(^|\.)news\.google\.[a-z.]+$", flags=re.IGNORECASE)
GOOGLE_NEWS_ARTICLE_RE = re.compile(r"/(?:rss/)?articles/([^/?#]+)", flags=re.IGNORECASE)


def _is_tracking_param(name: str, tracking_params: Optional[Iterable[str]] = None) -> bool:
    n = name.lower().strip()
    tracking = set(p.lower() for p in (tracking_params or TRACKING_PARAMS_DEFAULT))
    return n.startswith("utm_") or n in tracking


def canonicalize_url(
    url: Any,
    tracking_params: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """Canonicalize URLs for stable deduplication."""
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

    return urlunsplit((scheme, netloc, path, normalized_query, ""))


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
    match = GOOGLE_NEWS_ARTICLE_RE.search(parsed.path or "")
    if not match:
        return None
    token = (match.group(1) or "").strip()
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
    match = re.search(r"https?://[^\s\"'<>\\]+", decoded, flags=re.IGNORECASE)
    if match:
        return match.group(0).strip()
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
        "null,180,null,null,null,null,null,0,null,null,[1608992183,723341000]],"
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

        match = re.search(
            r'garturlres\\",\\\"(https?:\\\\/\\\\/[^\\\"]+)',
            text,
            flags=re.IGNORECASE,
        )
        if match:
            decoded = match.group(1).replace("\\/", "/").strip()
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
    """Resolve Google News wrapper URLs to publisher URLs."""
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
    """Detect the Google consent/interstitial page."""
    url_text = (fetched_url or "").lower()
    body = (html or "").lower()

    if "consent.google.com" in url_text:
        return True
    if "before you continue to google" in body:
        return True
    if (
        "g.co/privacytools" in body
        and "cookies and data" in body
        and "accept all" in body
        and "reject all" in body
    ):
        return True

    return False
