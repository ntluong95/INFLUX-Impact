"""OpenAI Batch API provider implementation.

Implements the BatchProvider protocol for submitting headline classification
requests via the OpenAI Batch API (file upload → batch creation → polling →
result download).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

from src.utils.batch_provider import BatchResult, BatchStatus, write_request_jsonl


# ---------------------------------------------------------------------------
# OpenAI-specific helpers
# ---------------------------------------------------------------------------

def _normalized_base_url(value: str | None, default: str = "https://api.openai.com/v1") -> str:
    base_url = (value or "").strip() or default
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def _headers(api_key: str) -> dict[str, str]:
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _max_tokens_param(model: str) -> str:
    return "max_completion_tokens" if model.strip().lower().startswith("gpt-5") else "max_tokens"


def _supports_temperature(model: str) -> bool:
    return not model.strip().lower().startswith("gpt-5")


def _extract_message_content(body: dict[str, Any]) -> str:
    """Extract assistant text from an OpenAI chat completion response body."""
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


def _parse_json_object(raw_text: str) -> dict[str, Any] | None:
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


def _normalize_status(openai_status: str) -> str:
    """Map OpenAI batch statuses to the normalized status set."""
    mapping = {
        "validating": "validating",
        "in_progress": "in_progress",
        "finalizing": "finalizing",
        "submitted": "submitted",
        "completed": "completed",
        "failed": "failed",
        "expired": "expired",
        "cancelled": "cancelled",
        "cancelling": "cancelled",
    }
    return mapping.get(openai_status, openai_status)


# ---------------------------------------------------------------------------
# Raw HTTP helpers
# ---------------------------------------------------------------------------

def _upload_file(
    jsonl_path: Path, *, api_key: str, base_url: str, timeout: int
) -> dict[str, Any]:
    url = f"{_normalized_base_url(base_url)}/files"
    with jsonl_path.open("rb") as handle:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            data={"purpose": "batch"},
            files={"file": (jsonl_path.name, handle, "application/jsonl")},
            timeout=timeout,
        )
    resp.raise_for_status()
    return resp.json()


def _create_batch(
    *, input_file_id: str, api_key: str, base_url: str,
    completion_window: str, metadata: dict[str, str] | None, timeout: int,
) -> dict[str, Any]:
    url = f"{_normalized_base_url(base_url)}/batches"
    payload: dict[str, Any] = {
        "input_file_id": input_file_id,
        "endpoint": "/v1/chat/completions",
        "completion_window": completion_window,
    }
    if metadata:
        payload["metadata"] = metadata
    resp = requests.post(
        url, headers=_headers(api_key), json=payload, timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def _retrieve_batch(
    *, batch_id: str, api_key: str, base_url: str, timeout: int,
) -> dict[str, Any]:
    url = f"{_normalized_base_url(base_url)}/batches/{batch_id}"
    resp = requests.get(url, headers=_headers(api_key), timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _download_file(
    *, file_id: str, api_key: str, base_url: str, timeout: int,
) -> str:
    url = f"{_normalized_base_url(base_url)}/files/{file_id}/content"
    resp = requests.get(url, headers=_headers(api_key), timeout=timeout)
    resp.raise_for_status()
    return resp.text


# ---------------------------------------------------------------------------
# OpenAI BatchProvider
# ---------------------------------------------------------------------------

class OpenAIBatchProvider:
    """BatchProvider implementation for the OpenAI Batch API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "",
        timeout_seconds: int = 60,
        completion_window: str = "24h",
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout_seconds
        self._completion_window = completion_window

    @property
    def provider_name(self) -> str:
        return "openai"

    def build_batch_request(
        self,
        custom_id: str,
        system_prompt: str,
        user_prompt: str,
        model: str,
        max_tokens: int,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            _max_tokens_param(model): max_tokens,
        }
        if _supports_temperature(model):
            body["temperature"] = float((extra or {}).get("temperature", 0.0))
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }

    def submit_batch(
        self,
        requests_list: list[dict[str, Any]],
        metadata: dict[str, str],
        jsonl_path: Path | None = None,
    ) -> tuple[str, str]:
        if jsonl_path is None:
            raise ValueError("OpenAI batch provider requires jsonl_path for file upload.")
        write_request_jsonl(requests_list, jsonl_path)
        upload = _upload_file(
            jsonl_path, api_key=self._api_key,
            base_url=self._base_url, timeout=self._timeout,
        )
        batch = _create_batch(
            input_file_id=str(upload["id"]),
            api_key=self._api_key,
            base_url=self._base_url,
            completion_window=self._completion_window,
            metadata=metadata,
            timeout=self._timeout,
        )
        return str(batch["id"]), _normalize_status(str(batch.get("status", "submitted")))

    def poll_batch(self, batch_id: str) -> BatchStatus:
        raw = _retrieve_batch(
            batch_id=batch_id, api_key=self._api_key,
            base_url=self._base_url, timeout=self._timeout,
        )
        return BatchStatus(
            batch_id=batch_id,
            status=_normalize_status(str(raw.get("status", ""))),
            output_file_id=str(raw.get("output_file_id") or ""),
            error_file_id=str(raw.get("error_file_id") or ""),
            raw=raw,
        )

    def retrieve_results(self, batch_id: str, status: BatchStatus) -> list[BatchResult]:
        results: list[BatchResult] = []
        for file_id, is_error_file in [
            (status.output_file_id, False),
            (status.error_file_id, True),
        ]:
            if not file_id:
                continue
            text = _download_file(
                file_id=file_id, api_key=self._api_key,
                base_url=self._base_url, timeout=self._timeout,
            )
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                custom_id = str(record.get("custom_id", ""))
                if not custom_id:
                    continue
                error_obj = record.get("error")
                response = record.get("response") or {}
                status_code = int(response.get("status_code", 0) or 0)
                body = response.get("body") or {}

                if error_obj:
                    results.append(BatchResult(
                        custom_id=custom_id, success=False,
                        raw_text="", error=str(error_obj),
                    ))
                elif status_code >= 400:
                    results.append(BatchResult(
                        custom_id=custom_id, success=False,
                        raw_text="",
                        error=json.dumps(body, ensure_ascii=False)[:500],
                    ))
                else:
                    results.append(BatchResult(
                        custom_id=custom_id, success=True,
                        raw_text=_extract_message_content(body),
                        error="",
                    ))
        return results
