from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.utils.common import (
    atomic_write_text,
    ensure_dir,
    normalize_whitespace,
    utc_now_iso,
    write_dataframe_atomic,
    write_json_atomic,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_URL = (
    "https://www.woah.org/en/what-we-do/animal-health-and-welfare/animal-diseases/"
)
DEFAULT_RAW_HTML_DIR = REPO_ROOT / "data" / "raw_html" / "woah_animal_diseases"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "data" / "inputs" / "animal_diseases_woah_raw.csv"
DEFAULT_OUTPUT_PARQUET = (
    REPO_ROOT / "data" / "inputs" / "animal_diseases_woah_raw.parquet"
)
DEFAULT_OUTPUT_JSON = REPO_ROOT / "data" / "inputs" / "animal_diseases_woah_raw.json"
DEFAULT_LOG_PATH = REPO_ROOT / "data" / "logs" / "woah_animal_diseases.log"


@dataclass(slots=True)
class ExtractorRuntime:
    SmartScraperGraph: Any
    sync_playwright: Any
    PlaywrightTimeoutError: type[Exception]
    DiseasePage: type[Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve WOAH terrestrial and aquatic animal disease cards by "
            "rendering the paginated list in Playwright, then extracting the "
            "visible cards from each page with ScrapeGraphAI + Ollama."
        )
    )
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument(
        "--ollama-model",
        # TODO May not need it, we can use lighter version
        default="llama3.2",
        help="Ollama chat model name, without the ollama/ prefix.",
    )
    parser.add_argument(
        "--embedding-model",
        default="",
        help=(
            "Optional Ollama embedding model name, without the ollama/ prefix. "
            "Leave blank to skip embeddings."
        ),
    )
    parser.add_argument(
        "--ollama-base-url",
        default="http://127.0.0.1:11434",
        help="Root Ollama URL. Example: http://127.0.0.1:11434",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=30,
        help="Upper bound for paginated WOAH pages.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=90,
        help="Browser and request timeout in seconds.",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Playwright headless. Use --no-headless for debugging.",
    )
    parser.add_argument(
        "--raw-html-dir",
        type=Path,
        default=DEFAULT_RAW_HTML_DIR,
        help="Directory used to save rendered list fragments page by page.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=DEFAULT_OUTPUT_PARQUET,
        help="Optional output parquet path.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Output JSON metadata path.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Log file path.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose ScrapeGraphAI execution logging.",
    )
    return parser.parse_args()


def load_runtime() -> ExtractorRuntime:
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
        from pydantic import BaseModel, Field
        from scrapegraphai.graphs import SmartScraperGraph
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "Missing dependencies for the WOAH ScrapeGraph extractor.\n"
            "Create the side environment first, for example:\n"
            "  UV_CACHE_DIR=/tmp/uv-cache uv venv --python "
            "/Users/luongnguyen/.local/share/uv/python/"
            "cpython-3.11-macos-aarch64-none/bin/python3.11 .venv-sga311\n"
            "  UV_CACHE_DIR=/tmp/uv-cache uv pip install --python "
            ".venv-sga311/bin/python scrapegraphai playwright pydantic\n"
            "  .venv-sga311/bin/playwright install chromium\n"
            f"Original import error: {exc}"
        ) from exc

    class DiseaseCard(BaseModel):
        disease_name: str = Field(
            description="Exact disease title as shown on the WOAH card."
        )
        detail_url: str = Field(
            description="Absolute disease detail URL from the card title link."
        )
        disease_categories: list[str] = Field(
            default_factory=list,
            description=(
                "Card category labels split into a list, such as "
                "Listed Diseases or With Official Disease Status."
            ),
        )
        animal_types: list[str] = Field(
            default_factory=list,
            description=(
                "Type of animals split into a list, for example Aquatics and "
                "Crustaceans."
            ),
        )
        top_level_group: str | None = Field(
            default=None,
            description=(
                "Use aquatic when the card belongs to Aquatics and terrestrial "
                "when the card belongs to Terrestrials."
            ),
        )

    class DiseasePage(BaseModel):
        diseases: list[DiseaseCard] = Field(
            default_factory=list,
            description="All disease cards visible in the provided HTML fragment.",
        )

    return ExtractorRuntime(
        SmartScraperGraph=SmartScraperGraph,
        sync_playwright=sync_playwright,
        PlaywrightTimeoutError=PlaywrightTimeoutError,
        DiseasePage=DiseasePage,
    )


def setup_logger(log_path: Path) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger("woah_animal_diseases")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def normalize_ollama_identifier(value: str) -> str:
    cleaned = normalize_whitespace(value)
    if not cleaned:
        return value
    return cleaned if cleaned.startswith("ollama/") else f"ollama/{cleaned}"


def normalize_base_url(base_url: str) -> str:
    cleaned = normalize_whitespace(base_url).rstrip("/")
    if cleaned.endswith("/v1"):
        cleaned = cleaned[: -len("/v1")]
    return cleaned


def split_terms(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        text = normalize_whitespace(str(value))
        if not text:
            return []
        items = re.split(r"\s*,\s*", text)
    results: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = normalize_whitespace(str(item))
        if not text:
            continue
        lowered = text.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        results.append(text)
    return results


def parse_cards_from_html(
    html: str, page_number: int, page_url: str
) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, Any]] = []
    for card in soup.select("li.cards__item"):
        title_link = card.select_one("h3.cards__title a")
        if title_link is None:
            continue
        disease_name = normalize_whitespace(title_link.get_text(" ", strip=True))
        detail_url = normalize_whitespace(title_link.get("href", ""))
        if not disease_name or not detail_url:
            continue

        type_text = normalize_whitespace(
            (card.select_one("p.cards__type") or {}).get_text(" ", strip=True)
            if card.select_one("p.cards__type")
            else ""
        )
        disease_categories = split_terms(type_text)

        animal_types: list[str] = []
        for info_item in card.select("div.cards__infos-item"):
            label = normalize_whitespace(
                info_item.select_one(".cards__infos-item-label").get_text(
                    " ", strip=True
                )
                if info_item.select_one(".cards__infos-item-label")
                else ""
            )
            if label.casefold() != "type of animals":
                continue
            value_node = info_item.select_one(".cards__infos-item-value")
            animal_types = split_terms(
                normalize_whitespace(value_node.get_text(", ", strip=True))
                if value_node is not None
                else ""
            )
            break

        top_level_group = normalize_top_level_group(None, animal_types=animal_types)
        rows.append(
            {
                "page_number": page_number,
                "source_page_url": page_url,
                "disease_name": disease_name,
                "detail_url": detail_url,
                "top_level_group": top_level_group,
                "animal_types": animal_types,
                "animal_subtypes": [
                    item
                    for item in animal_types
                    if item.casefold() not in {"aquatics", "terrestrials"}
                ],
                "disease_categories": disease_categories,
            }
        )
    return rows


def normalize_top_level_group(value: Any, animal_types: list[str]) -> str:
    choices = [normalize_whitespace(str(value))]
    choices.extend(animal_types)
    for choice in choices:
        lowered = choice.casefold()
        if lowered.startswith("aquatic"):
            return "aquatic"
        if lowered.startswith("terrestrial"):
            return "terrestrial"
    return "unknown"


def fetch_declared_total_pages(source_url: str, timeout_seconds: int) -> int | None:
    response = requests.get(
        source_url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    match = re.search(r'"total_pages":(\d+)', response.text)
    if not match:
        return None
    return int(match.group(1))


def dismiss_cookie_banner(page: Any) -> None:
    candidates = [
        "button:has-text('Accept')",
        "button:has-text('Accept all')",
        "button:has-text('Allow all')",
        "button:has-text('I accept')",
    ]
    for selector in candidates:
        try:
            button = page.locator(selector).first
            if button.is_visible(timeout=1000):
                button.click(timeout=2000)
                page.wait_for_timeout(500)
                return
        except Exception:
            continue


def get_first_card_title(page: Any) -> str:
    try:
        return normalize_whitespace(
            page.locator("h3.cards__title a").first.text_content() or ""
        )
    except Exception:
        return ""


def wait_for_results(
    page: Any,
    expected_page: int,
    previous_first_title: str,
    timeout_ms: int,
) -> None:
    page.wait_for_function(
        """
        ({ expectedPage, previousFirstTitle }) => {
          const active = document.querySelector(".facetwp-page.active");
          const cards = document.querySelectorAll(
            "ul.cards.cards--grid.facetwp-template li.cards__item"
          );
          if (!active || cards.length === 0) {
            return false;
          }
          const raw = active.getAttribute("data-page") || active.textContent || "";
          const current = Number(String(raw).trim());
          const firstTitleNode = document.querySelector("h3.cards__title a");
          const firstTitle = firstTitleNode ? firstTitleNode.textContent.replace(/\\s+/g, " ").trim() : "";
          return current === expectedPage && firstTitle && firstTitle !== previousFirstTitle;
        }
        """,
        arg={
            "expectedPage": expected_page,
            "previousFirstTitle": previous_first_title,
        },
        timeout=timeout_ms,
    )


def capture_rendered_html_pages(
    runtime: ExtractorRuntime,
    source_url: str,
    raw_html_dir: Path,
    max_pages: int,
    timeout_seconds: int,
    headless: bool,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    ensure_dir(raw_html_dir)
    page_payloads: list[dict[str, Any]] = []
    timeout_ms = int(timeout_seconds * 1000)

    with runtime.sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        context = browser.new_context()
        page = context.new_page()
        logger.info("Opening WOAH disease portal: %s", source_url)
        page.goto(source_url, wait_until="domcontentloaded", timeout=timeout_ms)
        dismiss_cookie_banner(page)
        page.wait_for_selector(
            "ul.cards.cards--grid.facetwp-template li.cards__item",
            timeout=timeout_ms,
        )

        total_pages = fetch_declared_total_pages(source_url, timeout_seconds) or 1
        total_pages = max(1, min(total_pages, max_pages))
        logger.info("WOAH page declares %s paginated pages", total_pages)

        for page_number in range(1, total_pages + 1):
            if page_number > 1:
                logger.info("Navigating to page %s", page_number)
                previous_first_title = get_first_card_title(page)
                next_button = page.locator("a.facetwp-page.next").first
                next_button.click(timeout=timeout_ms)
                wait_for_results(
                    page,
                    expected_page=page_number,
                    previous_first_title=previous_first_title,
                    timeout_ms=timeout_ms,
                )

            list_html = page.locator("ul.cards.cards--grid.facetwp-template").evaluate(
                "node => node.outerHTML"
            )
            html_path = raw_html_dir / f"page_{page_number:03d}.html"
            atomic_write_text(html_path, list_html)
            page_payloads.append(
                {
                    "page_number": page_number,
                    "page_url": page.url,
                    "html_path": str(html_path),
                    "html": list_html,
                }
            )
            logger.info("Saved rendered HTML for page %s to %s", page_number, html_path)

        context.close()
        browser.close()

    return page_payloads


def extract_visible_cards(
    runtime: ExtractorRuntime,
    html: str,
    page_number: int,
    page_url: str,
    ollama_model: str,
    embedding_model: str,
    base_url: str,
    verbose: bool,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    expected_rows = parse_cards_from_html(
        html, page_number=page_number, page_url=page_url
    )
    prompt = (
        "Extract all disease cards visible in this WOAH animal diseases HTML fragment. "
        "Return only the cards explicitly present in the HTML fragment. "
        "For each card, keep the exact disease title, the absolute disease detail URL, "
        "the card categories as a list split on commas, the type of animals as a list "
        "split on commas, and a top_level_group set to aquatic or terrestrial. "
        "Do not infer hidden pages or missing fields."
    )
    graph_config = {
        "llm": {
            "model": normalize_ollama_identifier(ollama_model),
            "temperature": 0,
            "format": "json",
            "base_url": base_url,
        },
        "verbose": verbose,
        "headless": True,
    }
    if normalize_whitespace(embedding_model):
        graph_config["embeddings"] = {
            "model": normalize_ollama_identifier(embedding_model),
            "base_url": base_url,
        }

    logger.info(
        "Running ScrapeGraphAI extraction for WOAH page %s (expected %s cards)",
        page_number,
        len(expected_rows),
    )
    try:
        graph = runtime.SmartScraperGraph(
            prompt=prompt,
            source=html,
            schema=runtime.DiseasePage,
            config=graph_config,
        )
        raw_result = graph.run()
        payload = (
            raw_result.model_dump() if hasattr(raw_result, "model_dump") else raw_result
        )
        diseases = payload.get("diseases", []) if isinstance(payload, dict) else []
    except Exception as exc:
        logger.warning(
            "ScrapeGraphAI failed on page %s, falling back to deterministic parse: %s",
            page_number,
            exc,
        )
        return expected_rows

    normalized_rows: list[dict[str, Any]] = []
    for disease in diseases:
        name = normalize_whitespace(str(disease.get("disease_name", "")))
        detail_url = normalize_whitespace(str(disease.get("detail_url", "")))
        if not name or not detail_url:
            continue
        animal_types = split_terms(disease.get("animal_types"))
        disease_categories = split_terms(disease.get("disease_categories"))
        top_level_group = normalize_top_level_group(
            disease.get("top_level_group"),
            animal_types=animal_types,
        )
        normalized_rows.append(
            {
                "page_number": page_number,
                "source_page_url": page_url,
                "disease_name": name,
                "detail_url": detail_url,
                "top_level_group": top_level_group,
                "animal_types": animal_types,
                "animal_subtypes": [
                    item
                    for item in animal_types
                    if item.casefold() not in {"aquatics", "terrestrials"}
                ],
                "disease_categories": disease_categories,
            }
        )
    if len(normalized_rows) != len(expected_rows):
        logger.warning(
            "ScrapeGraphAI returned %s cards on page %s, expected %s from HTML; "
            "falling back to deterministic parse",
            len(normalized_rows),
            page_number,
            len(expected_rows),
        )
        return expected_rows
    expected_identity = {
        (
            normalize_whitespace(row.get("detail_url", "")).casefold(),
            normalize_whitespace(row.get("disease_name", "")).casefold(),
        )
        for row in expected_rows
    }
    actual_identity = {
        (
            normalize_whitespace(row.get("detail_url", "")).casefold(),
            normalize_whitespace(row.get("disease_name", "")).casefold(),
        )
        for row in normalized_rows
    }
    if actual_identity != expected_identity:
        logger.warning(
            "ScrapeGraphAI card identities differed from rendered HTML on page %s; "
            "falling back to deterministic parse",
            page_number,
        )
        return expected_rows
    return normalized_rows


def deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (
            normalize_whitespace(row.get("detail_url", "")).casefold(),
            normalize_whitespace(row.get("disease_name", "")).casefold(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def rows_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["animal_types"] = df["animal_types"].apply(lambda values: " | ".join(values))
    df["animal_subtypes"] = df["animal_subtypes"].apply(
        lambda values: " | ".join(values)
    )
    df["disease_categories"] = df["disease_categories"].apply(
        lambda values: " | ".join(values)
    )
    return df.sort_values(["top_level_group", "disease_name"]).reset_index(drop=True)


def build_metadata(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    rendered_pages: list[dict[str, Any]],
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        group = row.get("top_level_group", "unknown")
        counts[group] = counts.get(group, 0) + 1
    page_manifest = [
        {
            "page_number": int(payload["page_number"]),
            "page_url": str(payload["page_url"]),
            "html_path": str(payload["html_path"]),
        }
        for payload in rendered_pages
    ]
    return {
        "retrieved_at_utc": utc_now_iso(),
        "source_url": args.source_url,
        "ollama_model": args.ollama_model,
        "embedding_model": args.embedding_model,
        "ollama_base_url": normalize_base_url(args.ollama_base_url),
        "rendered_pages": page_manifest,
        "row_count": len(rows),
        "counts_by_group": counts,
        "rows": rows,
    }


def main() -> int:
    args = parse_args()
    logger = setup_logger(args.log_path)
    runtime = load_runtime()
    base_url = normalize_base_url(args.ollama_base_url)

    rendered_pages = capture_rendered_html_pages(
        runtime=runtime,
        source_url=args.source_url,
        raw_html_dir=args.raw_html_dir,
        max_pages=args.max_pages,
        timeout_seconds=args.timeout_seconds,
        headless=args.headless,
        logger=logger,
    )

    rows: list[dict[str, Any]] = []
    for payload in rendered_pages:
        page_rows = extract_visible_cards(
            runtime=runtime,
            html=payload["html"],
            page_number=int(payload["page_number"]),
            page_url=str(payload["page_url"]),
            ollama_model=args.ollama_model,
            embedding_model=args.embedding_model,
            base_url=base_url,
            verbose=args.verbose,
            logger=logger,
        )
        if not page_rows:
            logger.warning("No diseases extracted from page %s", payload["page_number"])
        rows.extend(page_rows)

    deduped_rows = deduplicate_rows(rows)
    df = rows_to_dataframe(deduped_rows)
    if df.empty:
        raise SystemExit(
            "Extraction completed but no disease rows were returned. "
            "Check Playwright rendering, the Ollama server, and the selected model."
        )

    ensure_dir(args.output_csv.parent)
    write_dataframe_atomic(
        df, csv_path=args.output_csv, parquet_path=args.output_parquet
    )

    metadata = build_metadata(
        args=args, rows=deduped_rows, rendered_pages=rendered_pages
    )
    write_json_atomic(metadata, args.output_json)

    logger.info("Wrote %s deduplicated disease rows to %s", len(df), args.output_csv)
    logger.info("Also wrote metadata JSON to %s", args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
