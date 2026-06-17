from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_WORKBOOK = SCRIPT_DIR / "EPPs input file.xlsx"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "openai_outputs"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_CLAUDE_BASE_URL = "https://api.anthropic.com"
DEFAULT_OPENAI_MODEL = "gpt-5.1"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
PROVIDERS = ("openai", "claude")
MAIN_SHEET = "Main list"
COUNTRY_SHEET_CANDIDATES = ("Country", "Country sheet")
STANDARD_NAME_COLUMN = "Scientific name_standardize"
EVENT_DEFINITION = (
    "Emergence event can be defined as either ongoing new occurrence or a "
    "temporarily bounded and geographically/epidemiologically linked episode. "
    "Events should be merged only when they represent continuous or recurrent "
    "occurrence for at least two consecutive years in the same outbreak area "
    "or in neighboring countries within the same UNSD sub-region. Events "
    "should not be merged across sub-regions and continents."
)


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip()).lower()


def now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def yes_no(value: Any) -> str:
    return "Yes" if bool(value) else "No"


def normalize_provider(value: str | None) -> str:
    provider = (value or "openai").strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider '{value}'. Choose one of: {', '.join(PROVIDERS)}.")
    return provider


def default_model_for_provider(provider: str) -> str:
    provider = normalize_provider(provider)
    return DEFAULT_CLAUDE_MODEL if provider == "claude" else DEFAULT_OPENAI_MODEL
