"""OpenAI-compatible chat completions backend wrapper."""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI


class OpenAIChatClient:
    """Single OpenAI-compatible chat completions client."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        timeout_seconds: int = 60,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
        rate_limit_per_minute: float = 0.0,
    ) -> None:
        if not api_key:
            raise ValueError("Missing API key. Set OPENAI_API_KEY in .env.")

        self.api_key = api_key
        self.base_url = (base_url or "").rstrip("/")
        self.timeout_seconds = timeout_seconds
        kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout_seconds,
            "max_retries": 0,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url

        self.client = OpenAI(**kwargs)
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self._use_requests_transport = bool(self.base_url)
        self.rate_limit_per_minute = float(rate_limit_per_minute or 0.0)
        self._min_interval_seconds = (60.0 / self.rate_limit_per_minute) if self.rate_limit_per_minute > 0 else 0.0
        self._last_request_ts = 0.0
        self._rate_lock = threading.Lock()

    def _apply_rate_limit(self) -> None:
        if self._min_interval_seconds <= 0:
            return
        with self._rate_lock:
            now = time.monotonic()
            wait_s = self._min_interval_seconds - (now - self._last_request_ts)
            if wait_s > 0:
                time.sleep(wait_s)
            self._last_request_ts = time.monotonic()

    def _chat_completion_via_requests(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        strict_json: bool,
    ) -> Dict[str, Any]:
        if not self.base_url:
            raise RuntimeError("Requests transport requires base_url")

        endpoint = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if strict_json:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"HTTP request failed: {exc}") from exc

        if response.status_code >= 400 and strict_json:
            # Some OpenAI-compatible providers reject response_format.
            payload.pop("response_format", None)
            try:
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException as exc:
                raise RuntimeError(f"HTTP request failed after fallback: {exc}") from exc

        if response.status_code >= 400:
            body_preview = (response.text or "")[:400]
            raise RuntimeError(f"HTTP {response.status_code}: {body_preview}")

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            body_preview = (response.text or "")[:400]
            raise RuntimeError(f"Invalid JSON response: {body_preview}") from exc

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("No choices returned by chat completion endpoint")

        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        usage = data.get("usage") or {}
        return {
            "text": content,
            "id": data.get("id", ""),
            "model": data.get("model", model),
            "usage": usage,
        }

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 500,
        strict_json: bool = True,
    ) -> Dict[str, Any]:
        """Call chat completions with retry logic.

        Returns dict with text + raw response metadata.
        """
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self._apply_rate_limit()
                if self._use_requests_transport:
                    return self._chat_completion_via_requests(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        strict_json=strict_json,
                    )

                req: Dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if strict_json:
                    req["response_format"] = {"type": "json_object"}

                try:
                    response = self.client.chat.completions.create(**req)
                except Exception:
                    # Some OpenAI-compatible providers do not support response_format.
                    if strict_json and "response_format" in req:
                        req.pop("response_format", None)
                        response = self.client.chat.completions.create(**req)
                    else:
                        raise

                content = response.choices[0].message.content or ""
                return {
                    "text": content,
                    "id": response.id,
                    "model": response.model,
                    "usage": response.usage.model_dump() if response.usage else {},
                }
            except Exception as exc:  # pragma: no cover - external API behavior
                last_err = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * (2 ** (attempt - 1)))

        raise RuntimeError(f"OpenAI chat completion failed after retries: {last_err}")
