from __future__ import annotations

from typing import Any

from epp_emergence_analysis_prompt import ORIGINAL_USER_PROMPT, VERIFICATION_USER_PROMPT
from epp_emergence_analysis_provider_utils import parse_response_text, stage_result
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
                raise RuntimeError(f"OpenAI model refusal: {content.get('refusal', '')}")
    raise RuntimeError("OpenAI response did not contain output text.")


def usage_dict(response: Any) -> dict[str, Any]:
    data = response.model_dump(mode="json") if hasattr(response, "model_dump") else dict(response)
    return data.get("usage") or {}


def parse_response_json(response: Any, row: dict[str, Any], max_output_tokens: int) -> dict[str, Any]:
    return parse_response_text(
        provider="OpenAI",
        text=extract_output_text(response),
        row=row,
        max_output_tokens=max_output_tokens,
        incomplete_details=getattr(response, "incomplete_details", None),
    )


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
    request = build_structured_response_request(
        instructions=instructions,
        user_input=user_input,
        model=model,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        store=store,
        web_search=web_search,
        search_context_size=search_context_size,
        include_review=include_review,
    )
    try:
        return client.responses.create(**request)
    except Exception as exc:
        if is_model_not_found_error(exc, model):
            raise RuntimeError(
                f"OpenAI model '{model}' is not available to this API key. "
                "Run this script with --list-models, then retry with --model MODEL_ID."
            ) from exc
        if not web_search:
            raise
        print(f"OpenAI web-search request failed; retrying without web search: {exc}")
        request.pop("tools", None)
        request.pop("include", None)
        return client.responses.create(**request)


def build_structured_response_request(
    instructions: str,
    user_input: str,
    model: str,
    max_output_tokens: int,
    reasoning_effort: str,
    store: bool,
    web_search: bool,
    search_context_size: str,
    include_review: bool,
) -> dict[str, Any]:
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
        request["tools"] = [{"type": "web_search_preview", "search_context_size": search_context_size}]
        request["include"] = ["web_search_call.action.sources"]
    return request


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
        model=args.search_model,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
        store=args.store,
        web_search=args.web_search,
        search_context_size=args.search_context_size,
        include_review=False,
    )
    parsed = parse_response_json(response, row, args.max_output_tokens)
    return stage_result(
        provider="openai",
        row=row,
        stage="search",
        response_id=getattr(response, "id", ""),
        status=getattr(response, "status", ""),
        model=getattr(response, "model", args.search_model),
        usage=usage_dict(response),
        parsed_response=parsed,
    )


def call_review_openai(
    client: Any,
    row: dict[str, Any],
    country_rows_text: str,
    search_result: dict[str, Any],
    args: Any,
) -> dict[str, Any]:
    response = create_structured_response(
        client=client,
        instructions=REVIEW_PROMPT,
        user_input=build_review_prompt(row, country_rows_text, search_result),
        model=args.review_model_resolved,
        max_output_tokens=args.review_max_output_tokens,
        reasoning_effort=args.review_reasoning_effort,
        store=args.store,
        web_search=args.web_search,
        search_context_size=args.search_context_size,
        include_review=True,
    )
    parsed = parse_response_json(response, row, args.review_max_output_tokens)
    return stage_result(
        provider="openai",
        row=row,
        stage="review",
        response_id=getattr(response, "id", ""),
        status=getattr(response, "status", ""),
        model=getattr(response, "model", args.review_model_resolved),
        usage=usage_dict(response),
        parsed_response=parsed,
        search_result=search_result,
    )


def list_openai_models(client: Any) -> list[str]:
    return sorted(model.id for model in client.models.list().data)
