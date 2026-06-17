from __future__ import annotations

from typing import Any

from epp_emergence_analysis_claude_provider import (
    ClaudeClient,
    call_review_claude,
    call_search_claude,
    resolve_anthropic_api_key,
)
from epp_emergence_analysis_config import default_model_for_provider, normalize_provider
from epp_emergence_analysis_openai_provider import (
    call_review_openai,
    call_search_openai,
    list_openai_models,
)


def resolve_runtime_args(args: Any) -> Any:
    provider = normalize_provider(args.provider) if args.provider else None
    args.search_provider = normalize_provider(args.search_provider or provider or "openai")
    args.review_provider = normalize_provider(args.review_provider or provider or args.search_provider)
    args.search_model = args.model or default_model_for_provider(args.search_provider)
    args.review_model_resolved = args.review_model or (
        args.search_model
        if args.review_provider == args.search_provider
        else default_model_for_provider(args.review_provider)
    )
    return args


def required_providers(args: Any) -> set[str]:
    return {args.search_provider, args.review_provider}


def build_api_clients(args: Any) -> dict[str, Any]:
    clients: dict[str, Any] = {}
    if "openai" in required_providers(args):
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing OpenAI dependency. Install project dependencies or run: "
                "python3 -m pip install openai python-dotenv"
            ) from exc
        clients["openai"] = OpenAI(base_url=args.openai_base_url)
    if "claude" in required_providers(args):
        clients["claude"] = ClaudeClient(
            api_key=resolve_anthropic_api_key(),
            base_url=args.claude_base_url,
            timeout=args.timeout_seconds,
        )
    return clients


def list_provider_models(provider: str, args: Any) -> list[str]:
    provider = normalize_provider(provider)
    if provider == "openai":
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing OpenAI dependency. Install project dependencies or run: "
                "python3 -m pip install openai python-dotenv"
            ) from exc
        return list_openai_models(OpenAI(base_url=args.openai_base_url))
    client = ClaudeClient(
        api_key=resolve_anthropic_api_key(),
        base_url=args.claude_base_url,
        timeout=args.timeout_seconds,
    )
    return client.list_models()


def call_search_provider(
    clients: dict[str, Any],
    row: dict[str, Any],
    country_rows_text: str,
    args: Any,
) -> dict[str, Any]:
    if args.search_provider == "claude":
        return call_search_claude(clients["claude"], row, country_rows_text, args)
    return call_search_openai(clients["openai"], row, country_rows_text, args)


def call_review_provider(
    clients: dict[str, Any],
    row: dict[str, Any],
    country_rows_text: str,
    search_result: dict[str, Any],
    args: Any,
) -> dict[str, Any]:
    if args.review_provider == "claude":
        return call_review_claude(clients["claude"], row, country_rows_text, search_result, args)
    return call_review_openai(clients["openai"], row, country_rows_text, search_result, args)


call_openai = call_search_openai
