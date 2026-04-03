"""Headline relevance batch classification.

Supports both OpenAI and Anthropic batch APIs through a provider-agnostic
interface. The provider is selected by the `classification.provider` config
field ('openai' or 'anthropic'). Only one provider is used per run.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.batch_provider import (
    ACTIVE_STATUSES,
    TERMINAL_STATUSES,
    BatchProvider,
    BatchResult,
    write_request_jsonl,
)
from src.utils.common import (
    estimate_prompt_tokens,
    normalize_whitespace,
    setup_logger,
    text_or_empty,
    utc_now_iso,
    write_dataframe_atomic,
)
from src.utils.config import create_batch_provider, project_paths
from src.utils.project import DatasetKey


STATE_COLUMNS = [
    "record_id",
    "dataset_key",
    "pathogen_domain",
    "language_code",
    "search_string",
    "rss_title",
    "rss_description_text",
    "rss_pubdate",
    "source_hint",
    "source_url",
    "source_domain",
    "google_news_redirect_url",
    "batch_custom_id",
    "batch_name",
    "batch_id",
    "batch_provider",
    "batch_state",
    "batch_submitted_at",
    "headline_label",
    "headline_confidence",
    "headline_rationale",
    "headline_parse_error",
    "classification_error",
    "classified_at",
    "final_action",
]

BATCH_REGISTRY_COLUMNS = [
    "dataset_key",
    "batch_name",
    "batch_provider",
    "request_jsonl_path",
    "request_count",
    "estimated_prompt_tokens",
    "input_file_id",
    "batch_id",
    "status",
    "created_at",
    "completed_at",
    "hydrated_at",
    "output_file_id",
    "error_file_id",
]


def build_system_prompt(dataset: DatasetKey) -> str:
    domain_scope = {
        "human": "human pathogen or disease",
        "animal": "animal pathogen, pest, outbreak, or livestock/wildlife disease",
        "plant": "plant pathogen, pest, outbreak, or crop disease",
    }[dataset.pathogen_domain]
    return f"""You are a strict classifier for pathogen-news headline relevance.

Return JSON only with keys:
- label: relevant | irrelevant | unsure
- confidence: number from 0 to 1
- rationale: short rationale under 24 words

Rules:
- relevant: clearly about the searched {domain_scope}, including outbreaks, spread, control, surveillance, response, impacts, or consequences.
- irrelevant: homonym, metaphor, finance, sports, entertainment, spam, or a different topic.
- unsure: ambiguous from headline and summary alone.
- Use conservative confidence when the signal is weak.
"""


def build_user_prompt(row: pd.Series) -> str:
    return f"""Classify this Google News item.

Pathogen domain: {row.get("pathogen_domain", "[missing]")}
Language: {row.get("language_code", "[missing]")}
Matched search string: {row.get("search_string", "[missing]")}
Headline: {row.get("rss_title", "[missing]")}
Summary: {row.get("rss_description_text", "[missing]")}
Source: {row.get("source_hint", "[missing]")}
Source domain: {row.get("source_domain", "[missing]")}
Published: {row.get("rss_pubdate", "[missing]")}
"""


def parse_headline_label(raw_text: str) -> dict[str, Any]:
    """Parse JSON classification from the model's text response."""
    cleaned = raw_text.strip()
    if not cleaned:
        return {"label": "unsure", "confidence": 0.0, "rationale": "", "parse_error": "empty_response"}

    # Try to extract JSON object from text
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    try:
        parsed = json.loads(cleaned)
    except Exception:
        return {"label": "unsure", "confidence": 0.0, "rationale": "", "parse_error": "invalid_json"}

    label = str(parsed.get("label", "unsure")).strip().lower()
    parse_error = ""
    if label not in {"relevant", "irrelevant", "unsure"}:
        label = "unsure"
        parse_error = "invalid_label"

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
        parse_error = parse_error or "invalid_confidence"

    confidence = max(0.0, min(1.0, confidence))
    rationale = normalize_whitespace(str(parsed.get("rationale", "")))[:280]
    return {"label": label, "confidence": confidence, "rationale": rationale, "parse_error": parse_error}


def label_to_action(label: str) -> str:
    if label == "relevant":
        return "keep"
    if label == "irrelevant":
        return "drop"
    return "review"


def load_registry(registry_csv: Path) -> pd.DataFrame:
    if not registry_csv.exists():
        return pd.DataFrame(columns=BATCH_REGISTRY_COLUMNS)
    df = pd.read_csv(registry_csv, low_memory=False)
    for column in BATCH_REGISTRY_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df[BATCH_REGISTRY_COLUMNS].fillna("")


def _migrate_legacy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename legacy openai_* columns to generic batch_* columns."""
    renames = {
        "openai_custom_id": "batch_custom_id",
        "openai_batch_name": "batch_name",
        "openai_batch_id": "batch_id",
    }
    for old, new in renames.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
        elif old in df.columns:
            df.drop(columns=[old], inplace=True, errors="ignore")
    if "batch_provider" not in df.columns:
        df["batch_provider"] = ""
    return df


def load_or_initialize_state(input_csv: Path, state_csv: Path, dataset: DatasetKey) -> pd.DataFrame:
    input_df = pd.read_csv(input_csv, low_memory=False)
    work = input_df.copy()
    work["dataset_key"] = dataset.stem
    work["pathogen_domain"] = dataset.pathogen_domain
    work["language_code"] = dataset.language_code

    if state_csv.exists():
        existing = pd.read_csv(state_csv, low_memory=False)
        existing = _migrate_legacy_columns(existing)
        merged = work.merge(existing, on="record_id", how="left", suffixes=("", "_existing"))
        for column in STATE_COLUMNS:
            existing_column = f"{column}_existing"
            if column in work.columns and column != "record_id":
                continue
            if existing_column not in merged.columns:
                merged[column] = "" if column not in work.columns else merged[column]
                continue
            if column in work.columns:
                merged[column] = merged[column]
            else:
                merged[column] = merged[existing_column]
            merged.drop(columns=[existing_column], inplace=True, errors="ignore")
        work = merged

    for column in STATE_COLUMNS:
        if column not in work.columns:
            work[column] = ""
    return work[STATE_COLUMNS].fillna("")


def rows_needing_submission(state_df: pd.DataFrame) -> list[int]:
    pending_rows: list[int] = []
    for idx, row in state_df.iterrows():
        if text_or_empty(row.get("headline_label")):
            continue
        batch_state = text_or_empty(row.get("batch_state")).strip().lower()
        if batch_state in ACTIVE_STATUSES:
            continue
        pending_rows.append(idx)
    return pending_rows


def split_request_chunks(
    request_rows: list[dict[str, Any]],
    *,
    max_requests: int,
    max_bytes: int,
    max_prompt_tokens: int,
) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_bytes = 0
    current_prompt_tokens = 0

    for row in request_rows:
        line = json.dumps(row["request"], ensure_ascii=False)
        line_bytes = len(line.encode("utf-8")) + 1
        line_tokens = int(row["estimated_prompt_tokens"])
        would_overflow = current and (
            len(current) >= max_requests
            or current_bytes + line_bytes > max_bytes
            or (max_prompt_tokens > 0 and current_prompt_tokens + line_tokens > max_prompt_tokens)
        )
        if would_overflow:
            chunks.append(current)
            current = []
            current_bytes = 0
            current_prompt_tokens = 0

        current.append(row)
        current_bytes += line_bytes
        current_prompt_tokens += line_tokens

    if current:
        chunks.append(current)
    return chunks


def build_request_rows(
    state_df: pd.DataFrame,
    row_indexes: list[int],
    dataset: DatasetKey,
    provider: BatchProvider,
    provider_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    model = str(provider_cfg.get("model", ""))
    max_tokens = int(provider_cfg.get("max_tokens", 160))
    extra = {"temperature": float(provider_cfg.get("temperature", 0.0))}
    request_rows: list[dict[str, Any]] = []

    for idx in row_indexes:
        row = state_df.loc[idx]
        system_prompt = build_system_prompt(dataset)
        user_prompt = build_user_prompt(row)
        custom_id = f"{dataset.stem}:{row['record_id']}"
        request = provider.build_batch_request(
            custom_id=custom_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            max_tokens=max_tokens,
            extra=extra,
        )
        request_rows.append({
            "idx": idx,
            "custom_id": custom_id,
            "estimated_prompt_tokens": estimate_prompt_tokens(f"{system_prompt}\n{user_prompt}"),
            "request": request,
        })
    return request_rows


def active_enqueued_prompt_tokens(registry_df: pd.DataFrame) -> int:
    if registry_df.empty:
        return 0
    active = registry_df.loc[
        ~registry_df["status"].fillna("").astype(str).str.lower().isin(TERMINAL_STATUSES)
    ].copy()
    if active.empty:
        return 0
    return int(pd.to_numeric(active["estimated_prompt_tokens"], errors="coerce").fillna(0).sum())


def hydrate_batches(
    dataset: DatasetKey,
    config: dict[str, Any],
    state_df: pd.DataFrame,
    registry_df: pd.DataFrame,
    logger: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = project_paths(config)

    for registry_idx, registry_row in registry_df.iterrows():
        batch_id = text_or_empty(registry_row.get("batch_id"))
        if not batch_id:
            continue

        # Create provider matching the one that submitted this batch
        row_provider_name = text_or_empty(registry_row.get("batch_provider")) or "openai"
        provider = create_batch_provider(config, provider_override=row_provider_name)
        batch_status = provider.poll_batch(batch_id)
        status = batch_status.status

        registry_df.loc[registry_idx, "status"] = status
        registry_df.loc[registry_idx, "output_file_id"] = batch_status.output_file_id
        registry_df.loc[registry_idx, "error_file_id"] = batch_status.error_file_id
        if status == "completed" and not text_or_empty(registry_df.loc[registry_idx, "completed_at"]):
            registry_df.loc[registry_idx, "completed_at"] = utc_now_iso()

        state_mask = state_df["batch_id"].fillna("").astype(str) == batch_id
        if not state_mask.any():
            continue

        if status in ACTIVE_STATUSES:
            state_df.loc[state_mask, "batch_state"] = status
            continue

        if status != "completed":
            state_df.loc[state_mask, "batch_state"] = status or "unknown"
            state_df.loc[state_mask, "classification_error"] = (
                state_df.loc[state_mask, "classification_error"].replace("", f"batch_{status or 'unknown'}")
            )
            continue

        # Retrieve and apply results
        results = provider.retrieve_results(batch_id, batch_status)

        # Save raw results for audit
        result_path = paths.batch_results_dir(dataset) / f"{registry_row['batch_name']}_output.jsonl"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with result_path.open("w", encoding="utf-8") as f:
            for r in results:
                json.dump({"custom_id": r.custom_id, "success": r.success, "raw_text": r.raw_text, "error": r.error}, f, ensure_ascii=False)
                f.write("\n")

        result_map: dict[str, BatchResult] = {r.custom_id: r for r in results}

        for state_idx in state_df.index[state_mask].tolist():
            custom_id = text_or_empty(state_df.loc[state_idx, "batch_custom_id"])
            result = result_map.get(custom_id)
            if not result:
                continue

            if not result.success:
                state_df.loc[state_idx, "classification_error"] = normalize_whitespace(result.error)[:500]
                state_df.loc[state_idx, "batch_state"] = "error"
                continue

            parsed = parse_headline_label(result.raw_text)
            state_df.loc[state_idx, "headline_label"] = parsed["label"]
            state_df.loc[state_idx, "headline_confidence"] = parsed["confidence"]
            state_df.loc[state_idx, "headline_rationale"] = parsed["rationale"]
            state_df.loc[state_idx, "headline_parse_error"] = parsed["parse_error"]
            state_df.loc[state_idx, "classification_error"] = ""
            state_df.loc[state_idx, "final_action"] = label_to_action(parsed["label"])
            state_df.loc[state_idx, "classified_at"] = utc_now_iso()
            state_df.loc[state_idx, "batch_state"] = "hydrated"

        registry_df.loc[registry_idx, "hydrated_at"] = utc_now_iso()
        logger.info(
            "Hydrated %s batch %s for %s status=%s",
            row_provider_name,
            registry_row["batch_name"],
            dataset.stem,
            status,
        )

    return state_df, registry_df


def submit_batches(
    dataset: DatasetKey,
    config: dict[str, Any],
    state_df: pd.DataFrame,
    registry_df: pd.DataFrame,
    logger: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    provider = create_batch_provider(config)
    provider_name = provider.provider_name
    provider_cfg = config["classification"].get(provider_name, {})
    batch_cfg = provider_cfg.get("batch", {})

    pending_indexes = rows_needing_submission(state_df)
    if not pending_indexes:
        return state_df, registry_df

    request_rows = build_request_rows(state_df, pending_indexes, dataset, provider, provider_cfg)
    request_chunks = split_request_chunks(
        request_rows,
        max_requests=int(batch_cfg.get("max_requests_per_batch", 50000)),
        max_bytes=int(batch_cfg.get("max_input_file_bytes", 190_000_000)),
        max_prompt_tokens=int(batch_cfg.get("max_estimated_prompt_tokens_per_batch", 0)),
    )
    if not request_chunks:
        return state_df, registry_df

    paths = project_paths(config)
    paths.ensure_parent_dirs(dataset)

    max_pending_prompt_tokens = int(batch_cfg.get("max_estimated_enqueued_prompt_tokens", 0))
    max_pending_batches = int(batch_cfg.get("max_pending_batches_per_dataset", 0))
    pending_batch_count = (
        int((~registry_df["status"].fillna("").astype(str).str.lower().isin(TERMINAL_STATUSES)).sum())
        if not registry_df.empty
        else 0
    )

    next_batch_number = len(registry_df) + 1
    for offset, request_chunk in enumerate(request_chunks):
        estimated_chunk_tokens = int(sum(item["estimated_prompt_tokens"] for item in request_chunk))
        if max_pending_prompt_tokens > 0:
            existing_tokens = active_enqueued_prompt_tokens(registry_df)
            if existing_tokens + estimated_chunk_tokens > max_pending_prompt_tokens:
                logger.info(
                    "Stopped submitting new batches for %s: enqueued prompt tokens would exceed cap (%s).",
                    dataset.stem, max_pending_prompt_tokens,
                )
                break
        if max_pending_batches > 0 and pending_batch_count >= max_pending_batches:
            logger.info(
                "Stopped submitting new batches for %s: pending batch count reached cap (%s).",
                dataset.stem, max_pending_batches,
            )
            break

        batch_name = f"{dataset.stem}_headline_batch_{next_batch_number + offset:04d}"
        jsonl_path = paths.batch_requests_dir(dataset) / f"{batch_name}.jsonl"
        native_requests = [item["request"] for item in request_chunk]

        metadata = {
            "batch_name": batch_name,
            "dataset_key": dataset.stem,
            "pathogen_domain": dataset.pathogen_domain,
            "language_code": dataset.language_code,
            "stage": "headline_filter",
            "provider": provider_name,
        }

        batch_id, initial_status = provider.submit_batch(
            native_requests, metadata, jsonl_path=jsonl_path,
        )
        pending_batch_count += 1

        registry_row = {
            "dataset_key": dataset.stem,
            "batch_name": batch_name,
            "batch_provider": provider_name,
            "request_jsonl_path": str(jsonl_path.relative_to(paths.repo_root)),
            "request_count": len(request_chunk),
            "estimated_prompt_tokens": estimated_chunk_tokens,
            "input_file_id": "",
            "batch_id": batch_id,
            "status": initial_status,
            "created_at": utc_now_iso(),
            "completed_at": "",
            "hydrated_at": "",
            "output_file_id": "",
            "error_file_id": "",
        }
        registry_df = pd.concat([registry_df, pd.DataFrame([registry_row])], ignore_index=True)

        for item in request_chunk:
            state_df.loc[item["idx"], "batch_custom_id"] = item["custom_id"]
            state_df.loc[item["idx"], "batch_name"] = batch_name
            state_df.loc[item["idx"], "batch_id"] = batch_id
            state_df.loc[item["idx"], "batch_provider"] = provider_name
            state_df.loc[item["idx"], "batch_state"] = initial_status
            state_df.loc[item["idx"], "batch_submitted_at"] = utc_now_iso()

        logger.info(
            "Submitted %s batch %s for %s request_count=%s batch_id=%s",
            provider_name, batch_name, dataset.stem, len(request_chunk), batch_id,
        )

    return state_df, registry_df


def run_headline_batch_filtering(
    dataset: DatasetKey,
    config: dict[str, Any],
    force: bool,
    logger: Any,
) -> dict[str, Any]:
    paths = project_paths(config)
    input_csv = paths.prefiltered_headlines_csv(dataset)
    state_csv = paths.classified_headlines_csv(dataset)
    registry_csv = paths.batch_registry_csv(dataset)

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Missing prefiltered headline input for {dataset.stem}: {input_csv}"
        )

    if force and state_csv.exists():
        state_csv.unlink()
    if force and registry_csv.exists():
        registry_csv.unlink()

    state_df = load_or_initialize_state(input_csv, state_csv, dataset)
    registry_df = load_registry(registry_csv)
    state_df, registry_df = hydrate_batches(dataset, config, state_df, registry_df, logger)
    state_df, registry_df = submit_batches(dataset, config, state_df, registry_df, logger)

    write_dataframe_atomic(state_df[STATE_COLUMNS].fillna(""), state_csv)
    write_dataframe_atomic(registry_df[BATCH_REGISTRY_COLUMNS].fillna(""), registry_csv)

    pending = int(state_df["headline_label"].fillna("").astype(str).eq("").sum())
    ready = pending == 0 and not state_df.empty
    logger.info(
        "Headline filtering state for %s: ready=%s pending_rows=%s",
        dataset.stem, ready, pending,
    )
    return {
        "dataset_key": dataset.stem,
        "ready": ready,
        "pending_rows": pending,
        "state_csv": str(state_csv),
        "registry_csv": str(registry_csv),
    }
