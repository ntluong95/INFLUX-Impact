from __future__ import annotations

import json
from typing import Any


def parse_response_text(
    *,
    provider: str,
    text: str,
    row: dict[str, Any],
    max_output_tokens: int,
    incomplete_details: Any = None,
) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        detail_text = f" Response incomplete details: {incomplete_details}." if incomplete_details else ""
        raise RuntimeError(
            f"{provider} returned incomplete JSON for input row {row.get('Input row')}."
            f"{detail_text} Retry with a larger cap, for example "
            f"--max-output-tokens {max(max_output_tokens * 2, 20000)}."
        ) from exc


def stage_result(
    *,
    provider: str,
    row: dict[str, Any],
    stage: str,
    response_id: str,
    status: str,
    model: str,
    usage: dict[str, Any],
    parsed_response: dict[str, Any],
    search_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "input_row": row,
        "stage": stage,
        "provider": provider,
        "response_id": response_id,
        "status": status,
        "model": model,
        "usage": usage,
        "parsed_response": parsed_response,
    }
    if search_result is not None:
        result["search_result"] = search_result
    if provider == "openai":
        result.update(
            {
                "openai_response_id": response_id,
                "openai_status": status,
                "openai_model": model,
                "openai_usage": usage,
            }
        )
    elif provider == "claude":
        result.update(
            {
                "claude_response_id": response_id,
                "claude_status": status,
                "claude_model": model,
                "claude_usage": usage,
            }
        )
    return result
