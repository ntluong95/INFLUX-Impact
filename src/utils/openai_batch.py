from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests


def normalized_base_url(value: str | None, default: str = "https://api.openai.com/v1") -> str:
    base_url = (value or "").strip() or default
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def openai_headers(api_key: str) -> dict[str, str]:
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def openai_max_tokens_param(model: str) -> str:
    normalized = model.strip().lower()
    return "max_completion_tokens" if normalized.startswith("gpt-5") else "max_tokens"


def openai_supports_temperature(model: str) -> bool:
    return not model.strip().lower().startswith("gpt-5")


def upload_batch_file(
    jsonl_path: Path,
    *,
    api_key: str,
    base_url: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    url = f"{normalized_base_url(base_url)}/files"
    headers = {"Authorization": f"Bearer {api_key}"}
    with jsonl_path.open("rb") as handle:
        response = requests.post(
            url,
            headers=headers,
            data={"purpose": "batch"},
            files={"file": (jsonl_path.name, handle, "application/jsonl")},
            timeout=timeout_seconds,
        )
    response.raise_for_status()
    return response.json()


def create_batch(
    *,
    input_file_id: str,
    api_key: str,
    base_url: str,
    endpoint: str,
    completion_window: str,
    metadata: dict[str, str] | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    url = f"{normalized_base_url(base_url)}/batches"
    payload: dict[str, Any] = {
        "input_file_id": input_file_id,
        "endpoint": endpoint,
        "completion_window": completion_window,
    }
    if metadata:
        payload["metadata"] = metadata
    response = requests.post(
        url,
        headers=openai_headers(api_key),
        json=payload,
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response.json()


def retrieve_batch(
    *,
    batch_id: str,
    api_key: str,
    base_url: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    url = f"{normalized_base_url(base_url)}/batches/{batch_id}"
    response = requests.get(
        url,
        headers=openai_headers(api_key),
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response.json()


def download_file_content(
    *,
    file_id: str,
    api_key: str,
    base_url: str,
    timeout_seconds: int,
) -> str:
    url = f"{normalized_base_url(base_url)}/files/{file_id}/content"
    response = requests.get(
        url,
        headers=openai_headers(api_key),
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response.text


def extract_message_content(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content or "")


def parse_json_object(raw_text: str) -> dict[str, Any] | None:
    cleaned = raw_text.strip()
    if not cleaned:
        return None
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    try:
        return json.loads(cleaned)
    except Exception:
        return None
