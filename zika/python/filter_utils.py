from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

import hashlib


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def getenv_nonempty(key: str) -> str | None:
    value = os.getenv(key)
    if value is None:
        return None
    value = value.strip()
    return value or None


def load_config(config_path: Path) -> dict[str, Any]:
    env_path = config_path.parent / ".env"
    load_dotenv(env_path, override=False)
    env_file_values: dict[str, str] = {}
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env_file_values[key.strip()] = value.strip()

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    hf = cfg.setdefault("headline_filter", {})
    openai_cfg = hf.setdefault("openai", {})
    if key := (getenv_nonempty("OPENAI_API_KEY") or env_file_values.get("OPENAI_API_KEY", "")):
        openai_cfg["api_key"] = key
    if model := (getenv_nonempty("OPENAI_MODEL") or env_file_values.get("OPENAI_MODEL", "")):
        openai_cfg["model"] = model
    if base := (getenv_nonempty("OPENAI_BASE_URL") or env_file_values.get("OPENAI_BASE_URL", "")):
        openai_cfg["base_url"] = base
    if batch_enabled := (
        getenv_nonempty("OPENAI_BATCH_ENABLED")
        or env_file_values.get("OPENAI_BATCH_ENABLED", "")
    ):
        openai_cfg["use_batch"] = batch_enabled.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    if batch_window := (
        getenv_nonempty("OPENAI_BATCH_COMPLETION_WINDOW")
        or env_file_values.get("OPENAI_BATCH_COMPLETION_WINDOW", "")
    ):
        openai_cfg["batch_completion_window"] = batch_window
    if batch_poll_seconds := (
        getenv_nonempty("OPENAI_BATCH_POLL_SECONDS")
        or env_file_values.get("OPENAI_BATCH_POLL_SECONDS", "")
    ):
        openai_cfg["batch_poll_seconds"] = int(batch_poll_seconds)

    local_cfg = hf.setdefault("local", {})
    if provider := (
        getenv_nonempty("LOCAL_LLM_PROVIDER")
        or env_file_values.get("LOCAL_LLM_PROVIDER", "")
    ):
        local_cfg["provider"] = provider
    if base := (getenv_nonempty("LOCAL_LLM_BASE_URL") or env_file_values.get("LOCAL_LLM_BASE_URL", "")):
        local_cfg["base_url"] = base
    if model := (getenv_nonempty("LOCAL_LLM_MODEL") or env_file_values.get("LOCAL_LLM_MODEL", "")):
        local_cfg["model"] = model

    return cfg


def setup_logger(log_path: Path) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger("zika_filter")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def build_prompt(title: str, source_hint: str, pubdate: str) -> str:
    return f"""Classify this Google News headline for relevance to Zika.

Return JSON only with keys:
- label: relevant | irrelevant | unsure
- confidence: number from 0 to 1
- rationale: short rationale under 20 words

Rules:
- relevant: clearly about Zika virus, Zika disease, ZIKV, outbreaks, cases, spread, risk, policy, or health impacts related to Zika
- irrelevant: Zika is incidental, metaphorical, spammy, ambiguous, or not about the pathogen or disease
- unsure: not enough information from the headline alone

Headline: {title or "[missing]"}
Source: {source_hint or "[missing]"}
Published: {pubdate or "[missing]"}"""


def extract_json_object(raw_text: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"```json|```", "", raw_text).strip()
    match = re.search(r"(?s)\{.*\}", cleaned)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def parse_llm_label(raw_text: str, default_confidence: float = 0.5) -> dict[str, Any]:
    obj = extract_json_object(raw_text)
    if not obj:
        return {
            "label": "unsure",
            "confidence": default_confidence,
            "rationale": "",
            "parse_error": "invalid_json",
        }

    label = str(obj.get("label", "unsure")).lower().strip()
    if label not in {"relevant", "irrelevant", "unsure"}:
        label = "unsure"
        parse_error = "invalid_label"
    else:
        parse_error = ""

    try:
        confidence = float(obj.get("confidence", default_confidence))
    except TypeError, ValueError:
        confidence = default_confidence

    confidence = max(0.0, min(1.0, confidence))
    rationale = re.sub(r"\s+", " ", str(obj.get("rationale", ""))).strip()[:240]

    return {
        "label": label,
        "confidence": confidence,
        "rationale": rationale,
        "parse_error": parse_error,
    }


def normalized_base_url(value: str | None, default: str = "") -> str:
    base_url = (value or "").strip()
    if not base_url:
        base_url = default
    return base_url.rstrip("/")


def summarize_http_error(exc: requests.HTTPError, limit: int = 500) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)

    status_code = getattr(response, "status_code", "unknown")
    body = ""
    try:
        body = response.text.strip()
    except Exception:
        body = ""

    if body:
        compact = re.sub(r"\s+", " ", body)
        return f"HTTP {status_code}: {compact[:limit]}"
    return f"HTTP {status_code}: {exc}"


def response_needs_more_tokens(raw_text: str) -> bool:
    try:
        data = json.loads(raw_text)
    except Exception:
        return False

    if data.get("done_reason") == "length":
        message = data.get("message") or {}
        if (
            isinstance(message, dict)
            and not str(message.get("content", "")).strip()
            and (
                str(message.get("thinking", "")).strip()
                or str(message.get("reasoning", "")).strip()
            )
        ):
            return True

    choices = data.get("choices") or []
    if choices:
        choice = choices[0] or {}
        message = choice.get("message") or {}
        if (
            choice.get("finish_reason") == "length"
            and not str(message.get("content", "")).strip()
            and (
                str(message.get("thinking", "")).strip()
                or str(message.get("reasoning", "")).strip()
            )
        ):
            return True

    return False


def ollama_available_models(base_url: str, timeout_seconds: int = 30) -> list[str]:
    url = f"{normalized_base_url(base_url)}/api/tags"
    resp = requests.get(url, timeout=timeout_seconds)
    resp.raise_for_status()
    data = resp.json()
    return [str(model.get("name", "")).strip() for model in data.get("models", [])]


def call_openai_chat(system: str, user: str, cfg: dict[str, Any]) -> tuple[str, str]:
    headers = openai_headers(cfg)
    base_url = openai_api_base_url(cfg)
    url = f"{base_url}/chat/completions"
    payload = {
        "model": cfg.get("model", "gpt-5-nano"),
        "temperature": cfg.get("temperature", 0),
        "max_tokens": cfg.get("max_tokens", 120),
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }

    resp = requests.post(
        url, headers=headers, json=payload, timeout=cfg.get("timeout_seconds", 60)
    )
    resp.raise_for_status()
    data = resp.json()
    return resp.text, data["choices"][0]["message"].get("content", "")


def openai_api_base_url(cfg: dict[str, Any]) -> str:
    base_url = normalized_base_url(cfg.get("base_url"), "https://api.openai.com/v1")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    return base_url


def openai_headers(cfg: dict[str, Any]) -> dict[str, str]:
    api_key = cfg.get("api_key", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def upload_openai_batch_file(jsonl_path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    url = f"{openai_api_base_url(cfg)}/files"
    headers = {"Authorization": openai_headers(cfg)["Authorization"]}
    with jsonl_path.open("rb") as handle:
        resp = requests.post(
            url,
            headers=headers,
            data={"purpose": "batch"},
            files={"file": (jsonl_path.name, handle, "application/jsonl")},
            timeout=cfg.get("timeout_seconds", 60),
        )
    resp.raise_for_status()
    return resp.json()


def create_openai_batch(
    input_file_id: str,
    cfg: dict[str, Any],
    endpoint: str = "/v1/chat/completions",
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    url = f"{openai_api_base_url(cfg)}/batches"
    payload: dict[str, Any] = {
        "input_file_id": input_file_id,
        "endpoint": endpoint,
        "completion_window": cfg.get("batch_completion_window", "24h"),
    }
    if metadata:
        payload["metadata"] = metadata
    resp = requests.post(
        url,
        headers=openai_headers(cfg),
        json=payload,
        timeout=cfg.get("timeout_seconds", 60),
    )
    resp.raise_for_status()
    return resp.json()


def retrieve_openai_batch(batch_id: str, cfg: dict[str, Any]) -> dict[str, Any]:
    url = f"{openai_api_base_url(cfg)}/batches/{batch_id}"
    resp = requests.get(
        url,
        headers=openai_headers(cfg),
        timeout=cfg.get("timeout_seconds", 60),
    )
    resp.raise_for_status()
    return resp.json()


def download_openai_file_content(file_id: str, cfg: dict[str, Any]) -> str:
    url = f"{openai_api_base_url(cfg)}/files/{file_id}/content"
    resp = requests.get(
        url,
        headers=openai_headers(cfg),
        timeout=cfg.get("timeout_seconds", 60),
    )
    resp.raise_for_status()
    return resp.text


def parse_openai_style_content(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content", ""))


def post_local_ollama_generate(
    base_url: str, system: str, user: str, cfg: dict[str, Any]
) -> tuple[str, str]:
    url = f"{base_url}/api/generate"
    payload = {
        "model": cfg.get("model", "deepseek-r1:14b"),
        "system": system,
        "prompt": user,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": cfg.get("temperature", 0),
            "num_predict": cfg.get("max_tokens", 120),
        },
    }
    resp = requests.post(url, json=payload, timeout=cfg.get("timeout_seconds", 120))
    resp.raise_for_status()
    data = resp.json()
    return resp.text, str(data.get("response", ""))


def post_local_ollama_chat(
    base_url: str, system: str, user: str, cfg: dict[str, Any]
) -> tuple[str, str]:
    url = f"{base_url}/api/chat"
    payload = {
        "model": cfg.get("model", "deepseek-r1:14b"),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": cfg.get("temperature", 0),
            "num_predict": cfg.get("max_tokens", 120),
        },
    }
    resp = requests.post(url, json=payload, timeout=cfg.get("timeout_seconds", 120))
    resp.raise_for_status()
    data = resp.json()
    message = data.get("message") or {}
    return resp.text, str(message.get("content", ""))


def post_local_openai_compatible(
    base_url: str, system: str, user: str, cfg: dict[str, Any]
) -> tuple[str, str]:
    url = f"{base_url}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": cfg.get("model", "deepseek-r1:14b"),
        "temperature": cfg.get("temperature", 0),
        "max_tokens": cfg.get("max_tokens", 120),
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    resp = requests.post(
        url, headers=headers, json=payload, timeout=cfg.get("timeout_seconds", 120)
    )
    resp.raise_for_status()
    data = resp.json()
    return resp.text, parse_openai_style_content(data)


def call_local_chat(system: str, user: str, cfg: dict[str, Any]) -> tuple[str, str]:
    base_url = normalized_base_url(cfg.get("base_url"))
    if not base_url:
        raise ValueError("LOCAL_LLM_BASE_URL is not set.")

    provider = str(cfg.get("provider", "ollama")).strip().lower()
    attempts: list[tuple[str, Any]] = []
    if provider == "openai_compatible":
        attempts = [("openai_compatible", post_local_openai_compatible)]
    else:
        attempts = [
            ("ollama_chat", post_local_ollama_chat),
            ("ollama_generate", post_local_ollama_generate),
            ("openai_compatible", post_local_openai_compatible),
        ]

    errors: list[str] = []
    for label, request_fn in attempts:
        try:
            raw_text, content = request_fn(base_url, system, user, cfg)
            parsed = parse_llm_label(content, 0.5)
            if parsed["parse_error"] and response_needs_more_tokens(raw_text):
                retry_cfg = dict(cfg)
                retry_cfg["max_tokens"] = max(
                    int(cfg.get("max_tokens", 120)) * 2,
                    400,
                )
                raw_text, content = request_fn(base_url, system, user, retry_cfg)
                parsed = parse_llm_label(content, 0.5)
            if parsed["parse_error"]:
                errors.append(f"{label}: {parsed['parse_error']}")
                continue
            return raw_text, content
        except requests.HTTPError as exc:
            status_code = getattr(exc.response, "status_code", None)
            response_text = ""
            if exc.response is not None:
                try:
                    response_text = exc.response.text.strip()
                except Exception:
                    response_text = ""
            if status_code == 404 and "model" in response_text.lower() and "not found" in response_text.lower():
                raise ValueError(response_text)
            if status_code == 404:
                errors.append(f"{label}: 404")
                continue
            if status_code is not None:
                raise ValueError(f"{label}: {summarize_http_error(exc)}") from exc
            raise ValueError(f"{label}: {exc}") from exc
        except requests.RequestException as exc:
            errors.append(f"{label}: {exc}")
            continue

    raise ValueError(
        "Local LLM request failed across endpoints: " + "; ".join(errors or ["unknown"])
    )


class SbertScorer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is not installed. Please install it."
            )
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        if not text_a or not text_b:
            return float("nan")
        emb_a = self.model.encode(text_a, convert_to_numpy=True)
        emb_b = self.model.encode(text_b, convert_to_numpy=True)
        denom = np.linalg.norm(emb_a) * np.linalg.norm(emb_b)
        if denom == 0:
            return float("nan")
        return float(np.dot(emb_a, emb_b) / denom)


def derive_ensemble_action(
    openai_label: str, local_label: str, similarity: float, threshold: float
) -> str:
    if openai_label == "irrelevant" and local_label == "irrelevant":
        return "drop"
    if (
        openai_label == "relevant"
        and local_label == "relevant"
        and not pd.isna(similarity)
        and similarity >= threshold
    ):
        return "keep"
    return "review"


def deterministic_validation_sample(
    df: pd.DataFrame, sample_size: int, seed: int = 42
) -> pd.DataFrame:
    if df.empty or sample_size <= 0:
        return pd.DataFrame()

    df = df.copy()
    df["agreement_bucket"] = np.where(
        df["openai_label"] == df["local_label"], "agree", "disagree"
    )
    df["sample_stratum"] = df["final_action"] + "__" + df["agreement_bucket"]

    # Hash record ids for stable sorting
    def stable_hash(val: str) -> str:
        return hashlib.sha256(val.encode("utf-8")).hexdigest()

    df["sample_order_key"] = df["record_id"].astype(str).apply(stable_hash)

    # Oversample disagreement
    counts = df["sample_stratum"].value_counts().to_frame("available_n").reset_index()
    # Boost weight for disagree strata
    counts["weight"] = np.where(
        counts["sample_stratum"].str.contains("disagree"), 2.0, 1.0
    )
    counts["weighted_n"] = counts["available_n"] * counts["weight"]
    total_weight = counts["weighted_n"].sum()

    if total_weight > 0:
        counts["raw_target"] = sample_size * counts["weighted_n"] / total_weight
    else:
        counts["raw_target"] = 0

    counts["target_n"] = np.floor(counts["raw_target"]).astype(int)
    counts["Fractional"] = counts["raw_target"] - counts["target_n"]

    remaining = sample_size - int(counts["target_n"].sum())
    if remaining > 0:
        counts = counts.sort_values(
            by=["Fractional", "sample_stratum"], ascending=[False, True]
        ).reset_index(drop=True)
        for i in range(min(remaining, len(counts))):
            counts.loc[i, "target_n"] = counts.loc[i, "target_n"] + 1

    sampled_dfs = []
    for _, row in counts.iterrows():
        stratum_df = df[df["sample_stratum"] == row["sample_stratum"]]
        if not stratum_df.empty:
            sampled_dfs.append(
                stratum_df.sort_values("sample_order_key").head(row["target_n"])
            )

    sampled_df = pd.concat(sampled_dfs) if sampled_dfs else pd.DataFrame()
    return sampled_df.drop(
        columns=["agreement_bucket", "sample_stratum", "sample_order_key"]
    ).sort_values("record_id")


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]
) -> dict[str, float]:
    y_true_clean = []
    y_pred_clean = []
    for t, p in zip(y_true, y_pred):
        if t in labels and p in labels:
            y_true_clean.append(t)
            y_pred_clean.append(p)

    if not y_true_clean:
        return {
            "n": 0,
            "accuracy": np.nan,
            "macro_f1": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "specificity": np.nan,
            "balanced_accuracy": np.nan,
            "mcc": np.nan,
        }

    acc = accuracy_score(y_true_clean, y_pred_clean)
    macro_f1 = f1_score(
        y_true_clean, y_pred_clean, average="macro", labels=labels, zero_division=0
    )
    precision = precision_score(
        y_true_clean, y_pred_clean, average="macro", labels=labels, zero_division=0
    )
    recall = recall_score(
        y_true_clean, y_pred_clean, average="macro", labels=labels, zero_division=0
    )
    bal_acc = balanced_accuracy_score(y_true_clean, y_pred_clean)
    mcc = matthews_corrcoef(y_true_clean, y_pred_clean)

    # Specificity macro approx
    cm = confusion_matrix(y_true_clean, y_pred_clean, labels=labels)
    specs = []
    for i in range(len(labels)):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        if tn + fp > 0:
            specs.append(tn / (tn + fp))

    spec = np.mean(specs) if specs else np.nan

    return {
        "n": len(y_true_clean),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "precision": precision,
        "recall": recall,
        "specificity": spec,
        "balanced_accuracy": bal_acc,
        "mcc": mcc,
    }
