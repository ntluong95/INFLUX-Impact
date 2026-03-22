from __future__ import annotations

import base64
import csv
import json
import random
import re
import time
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import pandas as pd
import requests
from bs4 import BeautifulSoup
from readability import Document

from src.utils.common import ensure_dir, normalize_whitespace, sha256_text


warnings.filterwarnings("ignore", category=requests.exceptions.RequestsDependencyWarning)

try:
    import trafilatura
except Exception:  # pragma: no cover - optional dependency behavior
    trafilatura = None

try:
    from newspaper import Article
except Exception:  # pragma: no cover - optional dependency behavior
    Article = None


TRACKING_PARAMS = {
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
JSON_LD_ARTICLE_TYPES = {
    "article",
    "newsarticle",
    "report",
    "analysisnewsarticle",
    "liveblogposting",
}


def canonicalize_url(url: str | None) -> str | None:
    if not url:
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

    parsed = urlsplit(raw)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        return None
    host = (parsed.hostname or "").lower()
    if not host:
        return None
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/") or "/"
    query_pairs = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if not (k.lower().startswith("utm_") or k.lower() in TRACKING_PARAMS)
    ]
    query_pairs.sort(key=lambda item: (item[0], item[1]))
    query = urlencode(query_pairs, doseq=True)
    port = parsed.port
    if port and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        netloc = f"{host}:{port}"
    else:
        netloc = host
    return urlunsplit((scheme, netloc, path, query, ""))


def extract_google_news_article_id(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlsplit(url)
    if not GOOGLE_NEWS_HOST_RE.search((parsed.hostname or "").lower()):
        return None
    match = GOOGLE_NEWS_ARTICLE_RE.search(parsed.path or "")
    if not match:
        return None
    return (match.group(1) or "").strip() or None


def _decode_varint(data: bytes, offset: int = 0) -> tuple[int | None, int]:
    value = 0
    shift = 0
    pos = offset
    for _ in range(10):
        if pos >= len(data):
            return None, offset
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, pos
        shift += 7
    return None, offset


def decode_google_news_article_id_offline(article_id: str | None) -> str | None:
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


def _extract_google_sig_ts(html: str) -> tuple[str | None, str | None, str | None]:
    id_match = re.search(r'data-n-a-id="([^"]+)"', html)
    ts_match = re.search(r'data-n-a-ts="([^"]+)"', html)
    sg_match = re.search(r'data-n-a-sg="([^"]+)"', html)
    return (
        id_match.group(1).strip() if id_match else None,
        ts_match.group(1).strip() if ts_match else None,
        sg_match.group(1).strip() if sg_match else None,
    )


def _build_batchexecute_payload(article_id: str, ts: str, sg: str) -> str:
    inner = (
        '["garturlreq",[["en-US","US",["FINANCE_TOP_INDICES","WEB_TEST_1_0_0"],null,null,1,1,"US:en",'
        "null,180,null,null,null,null,null,0,null,null,[1608992183,723341000]],"
        '"en-US","US",1,[2,3,4,8],1,0,"655000234",0,0,null,0],'
        f'"{article_id}",{ts},"{sg}"]'
    )
    escaped_inner = inner.replace("\\", "\\\\").replace('"', '\\"')
    req = f'[[["Fbv4je","{escaped_inner}",null,"generic"]]]'
    return f"f.req={quote(req)}"


def decode_google_news_url_online(
    wrapper_url: str,
    article_id: str,
    timeout_seconds: int,
    session: requests.Session,
) -> tuple[str | None, str]:
    try:
        wrapper_resp = session.get(
            wrapper_url,
            timeout=timeout_seconds,
            allow_redirects=True,
        )
        article_token, ts, sg = _extract_google_sig_ts(wrapper_resp.text or "")
        token = article_token or article_id
        if not token or not ts or not sg:
            return None, "missing_sig_ts"

        payload = _build_batchexecute_payload(token, ts, sg)
        batch_resp = session.post(
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
            line = line.strip()
            if not line or line.startswith(")]}'"):
                continue
            try:
                outer = json.loads(line)
            except Exception:
                continue
            if not isinstance(outer, list):
                continue
            for entry in outer:
                if not (
                    isinstance(entry, list) and len(entry) >= 3 and entry[1] == "Fbv4je"
                ):
                    continue
                try:
                    inner = json.loads(entry[2])
                except Exception:
                    continue
                if not (isinstance(inner, list) and inner and inner[0] == "garturlres"):
                    continue
                for candidate in inner[1:]:
                    if isinstance(candidate, str) and re.match(
                        r"^https?://", candidate, flags=re.IGNORECASE
                    ):
                        return candidate.strip(), ""
        return None, "missing_garturlres"
    except Exception as exc:
        return None, str(exc)


def resolve_google_news_url(
    wrapper_url: str,
    timeout_seconds: int,
    session: requests.Session,
) -> tuple[str, str, str]:
    article_id = extract_google_news_article_id(wrapper_url)
    if not article_id:
        return wrapper_url, "not_google_wrapper", ""
    offline = decode_google_news_article_id_offline(article_id)
    if offline:
        return offline, "google_decode_offline", ""
    online, err = decode_google_news_url_online(
        wrapper_url,
        article_id,
        timeout_seconds,
        session,
    )
    if online:
        return online, "google_decode_batchexecute", ""
    return wrapper_url, "google_decode_failed", err


def polite_fetch(
    session: requests.Session,
    url: str,
    cfg: dict[str, Any],
) -> requests.Response:
    timeout_seconds = int(cfg.get("timeout_seconds", 30))
    max_retries = int(cfg.get("max_retries", 3))
    backoff = float(cfg.get("retry_backoff_seconds", 1.5))
    rate_limit = float(cfg.get("rate_limit_per_second", 0.5))
    sleep_seconds = 0.0 if rate_limit <= 0 else 1.0 / rate_limit
    user_agent = cfg.get("user_agent", "Mozilla/5.0")
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        time.sleep(sleep_seconds + random.uniform(0, min(0.25, sleep_seconds)))
        try:
            response = session.get(
                url,
                headers={"User-Agent": user_agent},
                timeout=timeout_seconds,
                allow_redirects=True,
            )
            if (
                response.status_code in {429, 500, 502, 503, 504}
                and attempt < max_retries
            ):
                time.sleep(backoff * (2 ** (attempt - 1)))
                continue
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(backoff * (2 ** (attempt - 1)))

    raise RuntimeError(str(last_error) if last_error else "Unknown request failure")


def extract_meta_content(soup: BeautifulSoup, selectors: list[tuple[str, str]]) -> str:
    for attr_name, attr_value in selectors:
        node = soup.find("meta", attrs={attr_name: attr_value})
        if node:
            content = node.get("content")
            if isinstance(content, str) and content.strip():
                return normalize_whitespace(content)
    return ""


def parse_json_ld(soup: BeautifulSoup) -> dict[str, Any]:
    for node in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = node.string or node.get_text(" ", strip=True)
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        candidates = parsed if isinstance(parsed, list) else [parsed]
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            node_type = str(candidate.get("@type", "")).lower()
            if node_type in JSON_LD_ARTICLE_TYPES:
                return candidate
            if node_type == "graph" and isinstance(candidate.get("@graph"), list):
                for entry in candidate["@graph"]:
                    if (
                        isinstance(entry, dict)
                        and str(entry.get("@type", "")).lower() in JSON_LD_ARTICLE_TYPES
                    ):
                        return entry
    return {}


def extract_authors(json_ld: dict[str, Any], soup: BeautifulSoup) -> list[str]:
    authors: list[str] = []
    author_field = json_ld.get("author")
    if isinstance(author_field, list):
        for item in author_field:
            if isinstance(item, dict) and item.get("name"):
                authors.append(normalize_whitespace(str(item["name"])))
            elif isinstance(item, str):
                authors.append(normalize_whitespace(item))
    elif isinstance(author_field, dict) and author_field.get("name"):
        authors.append(normalize_whitespace(str(author_field["name"])))
    elif isinstance(author_field, str):
        authors.append(normalize_whitespace(author_field))

    meta_author = extract_meta_content(
        soup,
        [
            ("name", "author"),
            ("property", "article:author"),
            ("name", "parsely-author"),
            ("name", "sailthru.author"),
            ("name", "byl"),
        ],
    )
    if meta_author:
        authors.extend(
            [
                normalize_whitespace(part)
                for part in re.split(r"[;,|]", meta_author)
                if normalize_whitespace(part)
            ]
        )

    deduped: list[str] = []
    seen = set()
    for author in authors:
        key = author.lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(author)
    return deduped


def extract_metadata(
    html: str,
    final_url: str,
    response: requests.Response | None = None,
) -> dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    json_ld = parse_json_ld(soup)

    canonical_url = (
        extract_meta_content(soup, [("property", "og:url")])
        or (
            soup.find("link", attrs={"rel": "canonical"}).get("href")
            if soup.find("link", attrs={"rel": "canonical"})
            else ""
        )
        or final_url
    )
    page_title = (
        extract_meta_content(
            soup,
            [("property", "og:title"), ("name", "twitter:title")],
        )
        or normalize_whitespace(
            json_ld.get("headline", "") if isinstance(json_ld, dict) else ""
        )
        or normalize_whitespace(
            soup.title.get_text(" ", strip=True) if soup.title else ""
        )
    )
    published_at = normalize_whitespace(
        str(json_ld.get("datePublished", ""))
    ) or extract_meta_content(
        soup,
        [
            ("property", "article:published_time"),
            ("name", "parsely-pub-date"),
            ("name", "pubdate"),
            ("name", "publish-date"),
            ("itemprop", "datePublished"),
        ],
    )
    language = (
        normalize_whitespace((soup.html.get("lang") if soup.html else "") or "")
        or extract_meta_content(
            soup,
            [("http-equiv", "content-language"), ("property", "og:locale")],
        )
        or (response.headers.get("Content-Language", "") if response else "")
    )
    authors = extract_authors(json_ld, soup)
    domain = (urlsplit(final_url).hostname or "").lower()

    return {
        "canonical_url": canonical_url,
        "page_title": page_title,
        "published_at": published_at,
        "authors": authors,
        "language": language,
        "domain": domain,
    }


def extract_with_trafilatura(html: str) -> tuple[str, str]:
    if trafilatura is None:
        return "", ""
    text = trafilatura.extract(html, include_comments=False, include_tables=False)
    return (normalize_whitespace(text), "trafilatura") if text else ("", "")


def extract_with_readability(html: str) -> tuple[str, str]:
    try:
        doc = Document(html)
        content_html = doc.summary(html_partial=True)
        soup = BeautifulSoup(content_html, "lxml")
        text = normalize_whitespace(soup.get_text(" ", strip=True))
        return (text, "readability_lxml") if text else ("", "")
    except Exception:
        return "", ""


def extract_with_newspaper(html: str, url: str) -> tuple[str, str]:
    if Article is None:
        return "", ""
    try:
        article = Article(url=url)
        article.set_html(html)
        article.parse()
        text = normalize_whitespace(article.text or "")
        return (text, "newspaper3k") if text else ("", "")
    except Exception:
        return "", ""


def extract_full_text(html: str, final_url: str) -> tuple[str, str]:
    for extractor in (extract_with_trafilatura, extract_with_readability):
        text, method = extractor(html)
        if text:
            return text, method
    text, method = extract_with_newspaper(html, final_url)
    if text:
        return text, method
    return "", ""


def write_dataframe(
    df: pd.DataFrame,
    csv_path: Path,
    parquet_path: Path | None = None,
) -> None:
    ensure_dir(csv_path.parent)
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    if parquet_path is not None:
        df.to_parquet(parquet_path, index=False)


def load_existing_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
