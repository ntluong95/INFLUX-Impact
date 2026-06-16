"""Anthropic Message Batches API provider implementation.

Implements the BatchProvider protocol for submitting headline classification
requests via the Anthropic Batch API. Unlike OpenAI, Anthropic batches send
requests inline (no file upload step) and stream results as JSONL from a
dedicated endpoint.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests as http_requests

from src.utils.batch_provider import BatchResult, BatchStatus, write_request_jsonl


_DEFAULT_BASE_URL = "https://api.anthropic.com"
_API_VERSION = "2023-06-01"


def _headers(api_key: str) -> dict[str, str]:
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set.")
    return {
        "x-api-key": api_key,
        "anthropic-version": _API_VERSION,
        "content-type": "application/json",
    }


def _normalize_status(anthropic_status: str) -> str:
    """Map Anthropic processing_status to normalized status set.

    Anthropic has only 3 statuses: in_progress, canceling, ended.
    We map 'ended' to 'completed' since per-result errors are handled
    individually in retrieve_results.
    """
    mapping = {
        "in_progress": "in_progress",
        "canceling": "cancelled",
        "ended": "completed",
    }
    return mapping.get(anthropic_status, anthropic_status)


def _extract_message_text(message: dict[str, Any]) -> str:
    """Extract text from an Anthropic Messages API response."""
    content = message.get("content") or []
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


class AnthropicBatchProvider:
    """BatchProvider implementation for the Anthropic Message Batches API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "",
        timeout_seconds: int = 60,
    ) -> None:
        self._api_key = api_key
        self._base_url = (base_url or "").strip().rstrip("/") or _DEFAULT_BASE_URL
        self._timeout = timeout_seconds

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def build_batch_request(
        self,
        custom_id: str,
        system_prompt: str,
        user_prompt: str,
        model: str,
        max_tokens: int,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        temperature = (extra or {}).get("temperature")
        if temperature is not None:
            params["temperature"] = float(temperature)
        return {"custom_id": custom_id, "params": params}

    def submit_batch(
        self,
        requests_list: list[dict[str, Any]],
        metadata: dict[str, str],
        jsonl_path: Path | None = None,
    ) -> tuple[str, str]:
        # Save JSONL for audit trail even though Anthropic sends inline
        if jsonl_path is not None:
            write_request_jsonl(requests_list, jsonl_path)

        url = f"{self._base_url}/v1/messages/batches"
        payload = {"requests": requests_list}
        resp = http_requests.post(
            url,
            headers=_headers(self._api_key),
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        batch_id = str(data["id"])
        status = _normalize_status(str(data.get("processing_status", "in_progress")))
        return batch_id, status

    def poll_batch(self, batch_id: str) -> BatchStatus:
        url = f"{self._base_url}/v1/messages/batches/{batch_id}"
        resp = http_requests.get(
            url, headers=_headers(self._api_key), timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return BatchStatus(
            batch_id=batch_id,
            status=_normalize_status(str(data.get("processing_status", ""))),
            output_file_id="",  # Anthropic doesn't use file IDs
            error_file_id="",
            raw=data,
        )

    def retrieve_results(self, batch_id: str, status: BatchStatus) -> list[BatchResult]:
        url = f"{self._base_url}/v1/messages/batches/{batch_id}/results"
        resp = http_requests.get(
            url, headers=_headers(self._api_key), timeout=max(self._timeout, 120),
        )
        resp.raise_for_status()

        results: list[BatchResult] = []
        for line in resp.text.splitlines():
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

            result_obj = record.get("result") or {}
            result_type = str(result_obj.get("type", ""))

            if result_type == "succeeded":
                message = result_obj.get("message") or {}
                results.append(BatchResult(
                    custom_id=custom_id,
                    success=True,
                    raw_text=_extract_message_text(message),
                    error="",
                ))
            elif result_type == "errored":
                error_info = result_obj.get("error") or {}
                inner = error_info.get("error") or error_info
                error_msg = str(inner.get("message", "") or json.dumps(inner))
                results.append(BatchResult(
                    custom_id=custom_id,
                    success=False,
                    raw_text="",
                    error=error_msg[:500],
                ))
            else:
                # canceled or expired
                results.append(BatchResult(
                    custom_id=custom_id,
                    success=False,
                    raw_text="",
                    error=f"batch_request_{result_type}",
                ))
        return results
