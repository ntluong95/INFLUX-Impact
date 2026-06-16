"""Provider-agnostic batch API interface for headline classification.

Each provider (OpenAI, Anthropic) implements the BatchProvider protocol.
The rest of the pipeline only interacts with this interface, never with
provider-specific request or response formats.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# Normalized batch statuses used by the rest of the pipeline
# ---------------------------------------------------------------------------

ACTIVE_STATUSES = {"submitted", "validating", "in_progress", "finalizing"}
TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


@dataclass(frozen=True)
class BatchResult:
    """Normalized result for a single request inside a completed batch."""

    custom_id: str
    success: bool
    raw_text: str  # The assistant's text content (empty on error)
    error: str  # Human-readable error description (empty on success)


@dataclass(frozen=True)
class BatchStatus:
    """Normalized status snapshot returned by poll_batch."""

    batch_id: str
    status: str  # One of ACTIVE_STATUSES | TERMINAL_STATUSES
    output_file_id: str  # Provider-specific (OpenAI file id, empty for Anthropic)
    error_file_id: str  # Provider-specific (OpenAI error file id, empty for Anthropic)
    raw: dict[str, Any]  # Full provider response for debugging


class BatchProvider(Protocol):
    """Protocol that OpenAI and Anthropic batch backends must implement."""

    @property
    def provider_name(self) -> str:
        """Return 'openai' or 'anthropic'."""
        ...

    def build_batch_request(
        self,
        custom_id: str,
        system_prompt: str,
        user_prompt: str,
        model: str,
        max_tokens: int,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a single request object in the provider's native format."""
        ...

    def submit_batch(
        self,
        requests: list[dict[str, Any]],
        metadata: dict[str, str],
        jsonl_path: Path | None = None,
    ) -> tuple[str, str]:
        """Submit a batch and return (batch_id, initial_status)."""
        ...

    def poll_batch(self, batch_id: str) -> BatchStatus:
        """Check current status of a previously submitted batch."""
        ...

    def retrieve_results(self, batch_id: str, status: BatchStatus) -> list[BatchResult]:
        """Download and normalize results for a completed/ended batch."""
        ...


def write_request_jsonl(requests: list[dict[str, Any]], jsonl_path: Path) -> None:
    """Write provider-native request objects as JSONL (used for audit trail)."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(req, ensure_ascii=False) for req in requests]
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
