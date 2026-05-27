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


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------

def _resolve_api_key(cfg_section: dict[str, Any], default_env: str) -> str:
    """Resolve API key from env var (preferred) or inline config."""
    env_name = str(cfg_section.get("api_key_env", default_env))
    key = os.getenv(env_name, "").strip()
    if key:
        return key
    key = str(cfg_section.get("api_key", "")).strip()
    if key:
        return key
    raise ValueError(
        f"Missing {env_name}. Add it to the local .env file before using batch classification."
    )


def require_openai_api_key(config: dict[str, Any]) -> str:
    openai_cfg = config.get("classification", {}).get("openai", {})
    return _resolve_api_key(openai_cfg, "OPENAI_API_KEY")


def require_anthropic_api_key(config: dict[str, Any]) -> str:
    anthropic_cfg = config.get("classification", {}).get("anthropic", {})
    return _resolve_api_key(anthropic_cfg, "ANTHROPIC_API_KEY")


# ---------------------------------------------------------------------------
# Batch provider factory
# ---------------------------------------------------------------------------

def create_batch_provider(
    config: dict[str, Any],
    provider_override: str | None = None,
) -> Any:
    """Create the appropriate BatchProvider based on config or override.

    The provider is selected by `classification.provider` in the YAML config,
    defaulting to 'openai'. An explicit override (used when hydrating batches
    that were submitted by a different provider) takes precedence.
    """
    from src.utils.anthropic_batch import AnthropicBatchProvider
    from src.utils.openai_batch import OpenAIBatchProvider

    cls_cfg = config.get("classification", {})
    provider_name = (provider_override or str(cls_cfg.get("provider", "openai"))).strip().lower()

    if provider_name == "anthropic":
        anthropic_cfg = cls_cfg.get("anthropic", {})
        return AnthropicBatchProvider(
            api_key=require_anthropic_api_key(config),
            base_url=str(anthropic_cfg.get("base_url", "") or ""),
            timeout_seconds=int(anthropic_cfg.get("timeout_seconds", 60)),
        )

    # Default to OpenAI
    openai_cfg = cls_cfg.get("openai", {})
    batch_cfg = openai_cfg.get("batch", {})
    return OpenAIBatchProvider(
        api_key=require_openai_api_key(config),
        base_url=str(openai_cfg.get("base_url", "") or ""),
        timeout_seconds=int(openai_cfg.get("timeout_seconds", 60)),
        completion_window=str(batch_cfg.get("completion_window", "24h")),
    )
