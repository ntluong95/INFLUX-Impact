from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.utils.project import ProjectPaths


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "src" / "config" / "pipeline.yaml"


def load_project_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()

    load_dotenv(REPO_ROOT / ".env", override=False)
    load_dotenv(path.parent / ".env", override=False)

    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    cfg.setdefault("paths", {})
    cfg.setdefault("inputs", {})
    cfg.setdefault("search", {})
    cfg.setdefault("rss", {})
    cfg.setdefault("classification", {})
    cfg.setdefault("fulltext", {})
    cfg.setdefault("bertopic", {})
    cfg["_config_path"] = str(path)
    return cfg


def config_path(config: dict[str, Any]) -> Path:
    return Path(str(config["_config_path"]))


def project_paths(config: dict[str, Any]) -> ProjectPaths:
    data_root = REPO_ROOT / str(config.get("paths", {}).get("data_root", "data"))
    return ProjectPaths(repo_root=REPO_ROOT, data_root=data_root)


def require_openai_api_key(config: dict[str, Any]) -> str:
    openai_cfg = config.get("classification", {}).get("openai", {})
    api_key_env = str(openai_cfg.get("api_key_env", "OPENAI_API_KEY"))
    api_key = os.getenv(api_key_env, "").strip()
    if api_key:
        return api_key
    api_key = str(openai_cfg.get("api_key", "")).strip()
    if api_key:
        return api_key
    raise ValueError(
        f"Missing {api_key_env}. Add it to the local .env file before using OpenAI batch classification."
    )
