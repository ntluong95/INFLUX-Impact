from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from epp_emergence_analysis_config import PROJECT_ROOT
from epp_emergence_analysis_excel import load_json, write_json


def load_openai_api_key() -> str:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        load_dotenv = None
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            f"OPENAI_API_KEY not found in {PROJECT_ROOT / '.env'} or the environment. "
            "Install python-dotenv if you need to load the key from .env."
        )
    return str(os.environ["OPENAI_API_KEY"])


def client_for(openai_base_url: str) -> Any:
    load_openai_api_key()
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The OpenAI SDK is required for submit/poll actions. "
            "Install dependencies with: python3 -m pip install openai python-dotenv pandas openpyxl requests. "
            "If you already created the project venv, run with: "
            'PYTHON="/tmp/influx-epp-venv/bin/python"'
        ) from exc
    return OpenAI(base_url=openai_base_url)


def request_line(custom_id: str, body: dict[str, Any]) -> dict[str, Any]:
    return {"custom_id": custom_id, "method": "POST", "url": "/v1/responses", "body": body}


def submit_batch(run_dir: Path, openai_base_url: str, jsonl_path: Path, label: str) -> Path:
    client = client_for(openai_base_url)
    with jsonl_path.open("rb") as handle:
        file_obj = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )
    path = run_dir / "openai_batch" / f"{label}_batch.json"
    write_json(batch.model_dump(mode="json"), path)
    print(f"{label} batch submitted: {batch.id}")
    print(f"{label} batch metadata: {path}")
    return path


def retrieve_batch(run_dir: Path, openai_base_url: str, label: str) -> dict[str, Any]:
    batch_info = load_json(run_dir / "openai_batch" / f"{label}_batch.json")
    batch = client_for(openai_base_url).batches.retrieve(batch_info["id"])
    data = batch.model_dump(mode="json")
    write_json(data, run_dir / "openai_batch" / f"{label}_batch.json")
    print(f"{label} batch {data['id']} status: {data['status']}")
    return data


def download_output(openai_base_url: str, file_id: str) -> list[dict[str, Any]]:
    api_key = load_openai_api_key()
    try:
        import requests
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The requests package is required to download Batch API output files. "
            "Install dependencies with: python3 -m pip install openai python-dotenv pandas openpyxl requests. "
            "If you already created the project venv, run with: "
            'PYTHON="/tmp/influx-epp-venv/bin/python"'
        ) from exc
    url = f"{openai_base_url.rstrip('/')}/files/{file_id}/content"
    response = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=600)
    response.raise_for_status()
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def response_body(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("error"):
        raise RuntimeError(f"Batch item {record.get('custom_id')} failed: {record['error']}")
    body = (record.get("response") or {}).get("body")
    if not isinstance(body, dict):
        raise RuntimeError(f"Batch item {record.get('custom_id')} did not contain a response body.")
    return body
