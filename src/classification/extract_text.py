"""Article text extraction from cached raw HTML using BeautifulSoup heuristics."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from bs4 import BeautifulSoup

try:
    from .utils import (
        is_google_consent_interstitial,
        normalize_whitespace,
        save_dataframe_with_parquet_fallback,
        sha256_text,
        text_or_empty,
    )
except ImportError:  # pragma: no cover - script execution path
    from utils import (
        is_google_consent_interstitial,
        normalize_whitespace,
        save_dataframe_with_parquet_fallback,
        sha256_text,
        text_or_empty,
    )


try:  # Optional fallback
    import trafilatura  # type: ignore

    HAS_TRAFILATURA = True
except Exception:  # pragma: no cover - optional dependency
    HAS_TRAFILATURA = False


try:  # Optional fallback
    from readability import Document  # type: ignore

    HAS_READABILITY = True
except Exception:  # pragma: no cover - optional dependency
    HAS_READABILITY = False


BOILERPLATE_TAGS = [
    "script",
    "style",
    "noscript",
    "nav",
    "footer",
    "aside",
    "iframe",
    "form",
    "button",
    "input",
    "svg",
    "canvas",
]

BOILERPLATE_HINTS = re.compile(
    r"cookie|consent|subscribe|newsletter|advert|ads|promo|share|social|"
    r"related|recommended|breadcrumb|menu|header|footer|sidebar",
    flags=re.IGNORECASE,
)


def _remove_boilerplate(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(BOILERPLATE_TAGS):
        tag.decompose()

    for node in soup.find_all(True):
        # bs4 can yield tags that were already decomposed when an ancestor
        # matched; those tags have attrs=None and must be skipped.
        attrs = getattr(node, "attrs", None)
        if not isinstance(attrs, dict):
            continue

        classes = attrs.get("class", [])
        if isinstance(classes, str):
            classes = [classes]
        elif not isinstance(classes, list):
            classes = []

        marker = " ".join(classes) + " " + str(attrs.get("id", ""))
        if marker and BOILERPLATE_HINTS.search(marker):
            if node.name not in {"html", "body", "article", "main"}:
                node.decompose()


def _extract_title(soup: BeautifulSoup) -> str:
    meta_og = soup.find("meta", attrs={"property": "og:title"})
    if meta_og and meta_og.get("content"):
        return normalize_whitespace(meta_og.get("content"))

    meta_tw = soup.find("meta", attrs={"name": "twitter:title"})
    if meta_tw and meta_tw.get("content"):
        return normalize_whitespace(meta_tw.get("content"))

    if soup.title and soup.title.get_text():
        return normalize_whitespace(soup.title.get_text())

    h1 = soup.find("h1")
    if h1 and h1.get_text():
        return normalize_whitespace(h1.get_text())

    return ""


def _node_text_features(node) -> Tuple[float, str]:
    paragraphs = [normalize_whitespace(p.get_text(" ", strip=True)) for p in node.find_all("p")]
    paragraphs = [p for p in paragraphs if len(p) >= 40]

    paragraph_text = "\n\n".join(paragraphs)
    paragraph_chars = len(paragraph_text)

    all_text = normalize_whitespace(node.get_text(" ", strip=True))
    all_chars = len(all_text)

    link_chars = sum(
        len(normalize_whitespace(a.get_text(" ", strip=True))) for a in node.find_all("a")
    )
    link_ratio = link_chars / max(all_chars, 1)

    punctuation = len(re.findall(r"[\.,;:!?]", paragraph_text))
    p_count = len(paragraphs)
    density = paragraph_chars / max(all_chars, 1)

    score = (
        paragraph_chars
        + (p_count * 120.0)
        + (punctuation * 1.5)
        + (density * 400.0)
        - (link_ratio * 500.0)
    )

    return score, paragraph_text


def extract_with_bs4_density(html: str, min_chars: int = 250) -> Tuple[str, str, str]:
    """Extract title/body using BeautifulSoup + text-density heuristic."""
    soup = BeautifulSoup(html, "lxml")
    _remove_boilerplate(soup)

    extracted_title = _extract_title(soup)

    candidates = soup.find_all(["article", "main", "section", "div"])
    best_score = -1e9
    best_text = ""

    for node in candidates:
        score, txt = _node_text_features(node)
        if score > best_score and len(txt) >= min_chars:
            best_score = score
            best_text = txt

    if not best_text:
        paragraphs = [normalize_whitespace(p.get_text(" ", strip=True)) for p in soup.find_all("p")]
        paragraphs = [p for p in paragraphs if len(p) >= 30]
        best_text = "\n\n".join(paragraphs)

    best_text = normalize_whitespace(best_text)
    return extracted_title, best_text, "bs4_density"


def _extract_with_trafilatura(html: str) -> Tuple[str, str, str]:
    if not HAS_TRAFILATURA:
        return "", "", ""

    text = trafilatura.extract(html, include_comments=False, include_tables=False)  # type: ignore
    if text:
        return "", normalize_whitespace(text), "trafilatura"
    return "", "", ""


def _extract_with_readability(html: str) -> Tuple[str, str, str]:
    if not HAS_READABILITY:
        return "", "", ""

    doc = Document(html)
    title = normalize_whitespace(doc.short_title() or "")
    content_html = doc.summary(html_partial=True)
    soup = BeautifulSoup(content_html, "lxml")
    text = normalize_whitespace(soup.get_text(" ", strip=True))
    if text:
        return title, text, "readability_lxml"
    return "", "", ""


def extract_from_html(
    html: str,
    extraction_cfg: Dict[str, Any],
) -> Tuple[str, str, str]:
    if is_google_consent_interstitial(html):
        return "blocked_interstitial", "", "interstitial_blocked"

    min_chars = int(extraction_cfg.get("min_text_chars", 250))
    title, text, method = extract_with_bs4_density(html, min_chars=min_chars)

    if len(text) >= min_chars:
        return title, text, method

    if extraction_cfg.get("fallback_trafilatura", True):
        f_title, f_text, f_method = _extract_with_trafilatura(html)
        if len(f_text) > len(text):
            title = f_title or title
            text = f_text
            method = f_method

    if extraction_cfg.get("fallback_readability", True):
        r_title, r_text, r_method = _extract_with_readability(html)
        if len(r_text) > len(text):
            title = r_title or title
            text = r_text
            method = r_method

    return title, text, method


def build_extracted_dataset(
    unique_df: pd.DataFrame,
    fetch_df: pd.DataFrame,
    raw_html_dir: Path,
    extraction_cfg: Dict[str, Any],
    logger=None,
) -> pd.DataFrame:
    """Create article extraction dataset from deduped URLs + fetch metadata."""
    fetch_lookup = fetch_df.set_index("canonical_url", drop=False) if not fetch_df.empty else None

    rows = []
    for i, (_, row) in enumerate(unique_df.iterrows(), start=1):
        canonical_url = text_or_empty(row.get("canonical_url"))
        original_url = text_or_empty(row.get("original_url"))
        rss_title = text_or_empty(row.get("rss_title"))
        rss_description = text_or_empty(row.get("rss_description"))
        pub_date = text_or_empty(row.get("pub_date"))

        fetch_status = "not_fetched"
        http_status = None
        fetched_at = ""
        extraction_method = "none"
        extracted_title = ""
        extracted_text = ""

        html_path = raw_html_dir / f"{sha256_text(canonical_url)}.html"

        if fetch_lookup is not None and canonical_url in fetch_lookup.index:
            frow = fetch_lookup.loc[canonical_url]
            fetch_status = text_or_empty(frow.get("fetch_status"))
            http_status = frow.get("http_status")
            fetched_at = text_or_empty(frow.get("fetched_at"))

        if html_path.exists():
            html = html_path.read_text(encoding="utf-8", errors="ignore")
            extracted_title, extracted_text, extraction_method = extract_from_html(
                html,
                extraction_cfg=extraction_cfg,
            )

        rows.append(
            {
                "canonical_url": canonical_url,
                "original_url": original_url,
                "rss_title": rss_title,
                "rss_description": rss_description,
                "pub_date": pub_date,
                "extracted_title": extracted_title,
                "extracted_text": extracted_text,
                "fetch_status": fetch_status,
                "http_status": http_status,
                "fetched_at": fetched_at,
                "extraction_method": extraction_method,
            }
        )

        if logger is not None and (i % 50 == 0 or i == len(unique_df)):
            logger.info("Extraction progress: %s/%s", i, len(unique_df))

    return pd.DataFrame(rows)


def save_extracted_dataset(df: pd.DataFrame, outdir: Path, prefer_parquet: bool = True) -> Path:
    target_parquet = outdir / "articles_extracted.parquet"
    if prefer_parquet:
        return save_dataframe_with_parquet_fallback(df, target_parquet)

    target_csv = outdir / "articles_extracted.csv"
    df.to_csv(target_csv, index=False)
    return target_csv
