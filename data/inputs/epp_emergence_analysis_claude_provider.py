from __future__ import annotations

import os
from typing import Any

import requests

from epp_emergence_analysis_config import DEFAULT_CLAUDE_BASE_URL
from epp_emergence_analysis_prompt import ORIGINAL_USER_PROMPT, VERIFICATION_USER_PROMPT
from epp_emergence_analysis_provider_utils import parse_response_text, stage_result
from epp_emergence_analysis_schema import response_schema
from epp_emergence_analysis_workbook import build_review_prompt, build_user_prompt


ANTHROPIC_VERSION = "2023-06-01"
CLAUDE_WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search"}
SEARCH_PROMPT = ORIGINAL_USER_PROMPT
REVIEW_PROMPT = VERIFICATION_USER_PROMPT


class ClaudeClient:
    def __init__(self, api_key: str, base_url: str = DEFAULT_CLAUDE_BASE_URL, timeout: int = 120) -> None:
        self.api_key = api_key
        self.base_url = normalize_base_url(base_url)
        self.timeout = timeout

    def headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    def post_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/v1/messages",
            headers=self.headers(),
            json=payload,
            timeout=self.timeout,
        )
        return parse_http_response(response)

    def list_models(self) -> list[str]:
        response = requests.get(
            f"{self.base_url}/v1/models",
            headers=self.headers(),
            timeout=self.timeout,
        )
        data = parse_http_response(response)
        models = data.get("data") or []
        return sorted(str(model.get("id", "")) for model in models if model.get("id"))


def normalize_base_url(value: str | None) -> str:
    base_url = (value or DEFAULT_CLAUDE_BASE_URL).strip().rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3].rstrip("/")
    return base_url or DEFAULT_CLAUDE_BASE_URL


def parse_http_response(response: requests.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError:
        data = {"raw_text": response.text}
    if response.status_code >= 400:
        message = data.get("error", {}).get("message") if isinstance(data.get("error"), dict) else ""
        detail = message or data.get("raw_text") or response.text
        raise RuntimeError(f"Claude API error {response.status_code}: {detail}")
    return data


def resolve_anthropic_api_key() -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not found in .env or the environment.")
    return api_key


def extract_output_text(response: dict[str, Any]) -> str:
    if response.get("stop_reason") == "refusal":
        raise RuntimeError("Claude model refusal.")
    parts: list[str] = []
    for block in response.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    text = "".join(parts).strip()
    if not text:
        raise RuntimeError("Claude response did not contain output text.")
    return text


def create_structured_message(
    client: ClaudeClient,
    instructions: str,
    user_input: str,
    model: str,
    max_output_tokens: int,
    web_search: bool,
    include_review: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_output_tokens,
        "system": instructions,
        "messages": [{"role": "user", "content": user_input}],
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": response_schema(include_review=include_review),
            }
        },
    }
    if web_search:
        payload["tools"] = [CLAUDE_WEB_SEARCH_TOOL]
    try:
        return client.post_message(payload)
    except RuntimeError as exc:
        if not web_search:
            raise
        print(f"Claude web-search request failed; retrying without web search: {exc}")
        payload.pop("tools", None)
        return client.post_message(payload)


def parse_response_json(response: dict[str, Any], row: dict[str, Any], max_output_tokens: int) -> dict[str, Any]:
    return parse_response_text(
        provider="Claude",
        text=extract_output_text(response),
        row=row,
        max_output_tokens=max_output_tokens,
        incomplete_details={"stop_reason": response.get("stop_reason")},
    )


def call_search_claude(
    client: ClaudeClient,
    row: dict[str, Any],
    country_rows_text: str,
    args: Any,
) -> dict[str, Any]:
    response = create_structured_message(
        client=client,
        instructions=SEARCH_PROMPT,
        user_input=build_user_prompt(row, country_rows_text, args.year_window),
        model=args.search_model,
        max_output_tokens=args.max_output_tokens,
        web_search=args.web_search,
        include_review=False,
    )
    parsed = parse_response_json(response, row, args.max_output_tokens)
    return stage_result(
        provider="claude",
        row=row,
        stage="search",
        response_id=str(response.get("id", "")),
        status=str(response.get("stop_reason", "")),
        model=str(response.get("model", args.search_model)),
        usage=response.get("usage") or {},
        parsed_response=parsed,
    )


def call_review_claude(
    client: ClaudeClient,
    row: dict[str, Any],
    country_rows_text: str,
    search_result: dict[str, Any],
    args: Any,
) -> dict[str, Any]:
    response = create_structured_message(
        client=client,
        instructions=REVIEW_PROMPT,
        user_input=build_review_prompt(row, country_rows_text, search_result),
        model=args.review_model_resolved,
        max_output_tokens=args.review_max_output_tokens,
        web_search=args.web_search,
        include_review=True,
    )
    parsed = parse_response_json(response, row, args.review_max_output_tokens)
    return stage_result(
        provider="claude",
        row=row,
        stage="review",
        response_id=str(response.get("id", "")),
        status=str(response.get("stop_reason", "")),
        model=str(response.get("model", args.review_model_resolved)),
        usage=response.get("usage") or {},
        parsed_response=parsed,
        search_result=search_result,
    )
