from __future__ import annotations

import json
from typing import Any

from epp_emergence_analysis_prompt import ORIGINAL_USER_PROMPT, VERIFICATION_USER_PROMPT
from epp_emergence_analysis_schema import response_schema
from epp_emergence_analysis_workbook import build_review_prompt, build_user_prompt


SEARCH_PROMPT = ORIGINAL_USER_PROMPT
REVIEW_PROMPT = VERIFICATION_USER_PROMPT


def is_model_not_found_error(exc: Exception, model: str) -> bool:
    text = str(exc).lower()
    return "not found" in text and model.lower() in text


def extract_output_text(response: Any) -> str:
    if getattr(response, "output_text", None):
        return response.output_text
    data = response.model_dump(mode="json") if hasattr(response, "model_dump") else dict(response)
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                return content["text"]
            if content.get("type") == "refusal":
                raise RuntimeError(f"Model refusal: {content.get('refusal', '')}")
    raise RuntimeError("OpenAI response did not contain output text.")


def usage_dict(response: Any) -> dict[str, Any]:
    data = response.model_dump(mode="json") if hasattr(response, "model_dump") else dict(response)
    return data.get("usage") or {}


def parse_response_json(response: Any, row: dict[str, Any], max_output_tokens: int) -> dict[str, Any]:
    text = extract_output_text(response)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        details = getattr(response, "incomplete_details", None)
        detail_text = f" Response incomplete details: {details}." if details else ""
        raise RuntimeError(
            "OpenAI returned incomplete JSON for input row "
            f"{row.get('Input row')}.{detail_text} Retry with a larger cap, for example "
            f"--max-output-tokens {max(max_output_tokens * 2, 20000)}."
        ) from exc


def create_structured_response(
    client: Any,
    instructions: str,
    user_input: str,
    model: str,
    max_output_tokens: int,
    reasoning_effort: str,
    store: bool,
    web_search: bool,
    search_context_size: str,
    include_review: bool,
) -> Any:
    request: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": user_input,
        "max_output_tokens": max_output_tokens,
        "reasoning": {"effort": reasoning_effort},
        "store": store,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "epp_emergence_event_profile",
                "strict": True,
                "schema": response_schema(include_review=include_review),
            }
        },
    }
    if web_search:
        request["tools"] = [
            {"type": "web_search_preview", "search_context_size": search_context_size}
        ]
        request["include"] = ["web_search_call.action.sources"]
    try:
        return client.responses.create(**request)
    except Exception as exc:
        if is_model_not_found_error(exc, model):
            raise RuntimeError(
                f"Model '{model}' is not available to this API key. "
                "Run this script with --list-models, then retry with --model MODEL_ID."
            ) from exc
        if not web_search:
            raise
        print(f"Web-search request failed; retrying without web search: {exc}")
        request.pop("tools", None)
        request.pop("include", None)
        return client.responses.create(**request)


def call_search_openai(
    client: Any,
    row: dict[str, Any],
    country_rows_text: str,
    args: Any,
) -> dict[str, Any]:
    response = create_structured_response(
        client=client,
        instructions=SEARCH_PROMPT,
        user_input=build_user_prompt(row, country_rows_text, args.year_window),
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
        store=args.store,
        web_search=args.web_search,
        search_context_size=args.search_context_size,
        include_review=False,
    )
    parsed = parse_response_json(response, row, args.max_output_tokens)
    return {
        "input_row": row,
        "stage": "search",
        "openai_response_id": getattr(response, "id", ""),
        "openai_status": getattr(response, "status", ""),
        "openai_model": getattr(response, "model", args.model),
        "openai_usage": usage_dict(response),
        "parsed_response": parsed,
    }


def call_review_openai(
    client: Any,
    row: dict[str, Any],
    country_rows_text: str,
    search_result: dict[str, Any],
    args: Any,
) -> dict[str, Any]:
    model = args.review_model or args.model
    response = create_structured_response(
        client=client,
        instructions=REVIEW_PROMPT,
        user_input=build_review_prompt(row, country_rows_text, search_result),
        model=model,
        max_output_tokens=args.review_max_output_tokens,
        reasoning_effort=args.review_reasoning_effort,
        store=args.store,
        web_search=args.web_search,
        search_context_size=args.search_context_size,
        include_review=True,
    )
    parsed = parse_response_json(response, row, args.review_max_output_tokens)
    return {
        "input_row": row,
        "stage": "review",
        "search_result": search_result,
        "openai_response_id": getattr(response, "id", ""),
        "openai_status": getattr(response, "status", ""),
        "openai_model": getattr(response, "model", model),
        "openai_usage": usage_dict(response),
        "parsed_response": parsed,
    }


call_openai = call_search_openai
