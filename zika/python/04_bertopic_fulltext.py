from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from bertopic import BERTopic
from dotenv import load_dotenv
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zika.python.scrape_utils import ensure_dir, write_dataframe  # noqa: E402


DOCUMENT_TOPIC_COLUMNS = [
    "record_id",
    "domain",
    "final_url",
    "rss_title",
    "word_count",
    "language",
    "model_word_count",
    "topic_id",
    "topic_name",
    "topic_keywords",
    "is_outlier",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BERTopic on the Zika full-text subset."
    )
    parser.add_argument("--config", default="zika/config/zika.yaml")
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_config(config_path: Path) -> dict[str, Any]:
    load_dotenv(config_path.parent / ".env", override=False)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def setup_logger(log_path: Path) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger("zika_bertopic")
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


def normalize_language(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    return text


def language_allowed(
    language: str, allowed_prefixes: list[str], allow_blank_language: bool
) -> bool:
    if not language:
        return allow_blank_language
    return any(language.startswith(prefix) for prefix in allowed_prefixes)


def truncate_words(text: str, max_words: int) -> str:
    cleaned = " ".join(str(text or "").split())
    if max_words <= 0:
        return cleaned
    words = cleaned.split()
    return " ".join(words[:max_words])


def format_elapsed(seconds: float) -> str:
    total_seconds = int(round(seconds))
    minutes, secs = divmod(total_seconds, 60)
    hours, mins = divmod(minutes, 60)
    if hours:
        return f"{hours}h {mins:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def prepare_documents(df: pd.DataFrame, topic_cfg: dict[str, Any]) -> pd.DataFrame:
    allowed_prefixes = [
        str(prefix).strip().lower()
        for prefix in topic_cfg.get("allowed_language_prefixes", ["en"])
        if str(prefix).strip()
    ]
    allow_blank_language = bool(topic_cfg.get("allow_blank_language", True))
    min_word_count = int(topic_cfg.get("min_word_count", 120))
    max_words_per_document = int(topic_cfg.get("max_words_per_document", 800))
    max_documents = int(topic_cfg.get("max_documents", 0))
    seed = int(topic_cfg.get("seed", 42))

    prepared = df.copy()
    prepared["language_norm"] = prepared.get("language", "").map(normalize_language)
    prepared["word_count"] = pd.to_numeric(
        prepared.get("word_count", np.nan), errors="coerce"
    ).fillna(0)
    prepared["full_text"] = prepared.get("full_text", "").fillna("").astype(str)
    prepared["rss_title"] = prepared.get("rss_title", "").fillna("").astype(str)

    prepared = prepared[
        prepared["full_text"].str.strip().ne("")
        & (prepared["word_count"] >= min_word_count)
        & prepared["language_norm"].map(
            lambda value: language_allowed(
                value, allowed_prefixes=allowed_prefixes, allow_blank_language=allow_blank_language
            )
        )
    ].copy()

    prepared["model_text"] = prepared.apply(
        lambda row: "\n\n".join(
            part
            for part in [
                str(row.get("rss_title", "")).strip(),
                truncate_words(row.get("full_text", ""), max_words_per_document),
            ]
            if part
        ),
        axis=1,
    )
    prepared["model_word_count"] = prepared["model_text"].map(
        lambda value: len(str(value).split())
    )
    prepared = prepared[prepared["model_word_count"] > 0].copy()

    if max_documents > 0 and len(prepared) > max_documents:
        prepared = prepared.sample(n=max_documents, random_state=seed).copy()

    prepared = prepared.sort_values("record_id").reset_index(drop=True)
    return prepared


def topic_keywords_for(topic_model: BERTopic, topic_id: int) -> str:
    if topic_id == -1:
        return "outlier"
    topic_terms = topic_model.get_topic(topic_id) or []
    return ", ".join(term for term, _score in topic_terms[:5])


def serialize_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    serialized = df.copy()
    for column in serialized.columns:
        if serialized[column].dtype != "object":
            continue
        serialized[column] = serialized[column].map(
            lambda value: " | ".join(map(str, value))
            if isinstance(value, list)
            else value
        )
    return serialized


def build_summary_payload(
    prepared_df: pd.DataFrame,
    source_rows: int,
    topic_info_df: pd.DataFrame,
    topic_cfg: dict[str, Any],
    runtime_seconds: float,
) -> dict[str, Any]:
    non_outlier = topic_info_df[topic_info_df["Topic"] != -1].copy()
    top_topics: list[dict[str, Any]] = []
    if not non_outlier.empty:
        for record in non_outlier.head(10).to_dict(orient="records"):
            top_topics.append(
                {
                    "topic_id": int(record.get("Topic", -1)),
                    "count": int(record.get("Count", 0)),
                    "name": str(record.get("Name", "")),
                    "representation": [
                        str(item)
                        for item in (record.get("Representation") or [])
                    ],
                }
            )

    outlier_row = topic_info_df[topic_info_df["Topic"] == -1]
    outlier_documents = (
        int(outlier_row["Count"].iloc[0]) if not outlier_row.empty else 0
    )

    return {
        "generated_at": utc_now_iso(),
        "documents_input": int(source_rows),
        "documents_modeled": int(len(prepared_df)),
        "documents_excluded": int(source_rows - len(prepared_df)),
        "unique_topics_excluding_outlier": int(len(non_outlier)),
        "outlier_documents": outlier_documents,
        "runtime_seconds": round(runtime_seconds, 3),
        "runtime_human": format_elapsed(runtime_seconds),
        "config": {
            "min_word_count": int(topic_cfg.get("min_word_count", 120)),
            "max_words_per_document": int(topic_cfg.get("max_words_per_document", 800)),
            "embedding_model": str(topic_cfg.get("embedding_model", "all-MiniLM-L6-v2")),
            "min_topic_size": int(topic_cfg.get("min_topic_size", 12)),
            "ngram_range": [
                int(topic_cfg.get("ngram_min", 1)),
                int(topic_cfg.get("ngram_max", 2)),
            ],
            "umap_neighbors": int(topic_cfg.get("umap_neighbors", 15)),
            "umap_components": int(topic_cfg.get("umap_components", 5)),
            "seed": int(topic_cfg.get("seed", 42)),
        },
        "top_topics": top_topics,
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config_path = (repo_root / args.config).resolve()
    zika_root = config_path.parent.parent
    cfg = load_config(config_path)
    topic_cfg = cfg.get("topic_model", {})

    log_path = zika_root / "logs" / "04_bertopic_fulltext.log"
    logger = setup_logger(log_path)

    input_csv = zika_root / str(topic_cfg.get("input_csv", "data/final/zika_fulltext.csv"))
    output_doc_csv = zika_root / "data" / "final" / "zika_bertopic_document_topics.csv"
    output_doc_parquet = (
        zika_root / "data" / "final" / "zika_bertopic_document_topics.parquet"
    )
    output_topic_info_csv = (
        zika_root / "data" / "final" / "zika_bertopic_topic_info.csv"
    )
    output_summary_json = (
        zika_root / "data" / "final" / "zika_bertopic_summary.json"
    )

    if not input_csv.exists():
        raise FileNotFoundError(f"Missing Stage 3 full-text input: {input_csv}")

    source_df = pd.read_csv(input_csv)
    prepared_df = prepare_documents(source_df, topic_cfg)

    logger.info(
        "BERTopic starting with %s source rows and %s modeled documents after filters",
        len(source_df),
        len(prepared_df),
    )

    if prepared_df.empty:
        raise RuntimeError("No documents remain after BERTopic input filtering.")

    seed = int(topic_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    embedding_model_name = str(topic_cfg.get("embedding_model", "all-MiniLM-L6-v2"))
    embedding_model = SentenceTransformer(embedding_model_name)

    vectorizer_model = CountVectorizer(
        stop_words=str(topic_cfg.get("vectorizer_stop_words", "english")),
        min_df=max(1, int(topic_cfg.get("min_df", 2))),
        ngram_range=(
            int(topic_cfg.get("ngram_min", 1)),
            int(topic_cfg.get("ngram_max", 2)),
        ),
    )
    umap_model = UMAP(
        n_neighbors=int(topic_cfg.get("umap_neighbors", 15)),
        n_components=int(topic_cfg.get("umap_components", 5)),
        min_dist=float(topic_cfg.get("umap_min_dist", 0.0)),
        metric="cosine",
        random_state=seed,
    )
    min_topic_size = int(topic_cfg.get("min_topic_size", 12))
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        top_n_words=int(topic_cfg.get("top_n_words", 10)),
        min_topic_size=min_topic_size,
        calculate_probabilities=False,
        verbose=True,
    )

    start_time = time.perf_counter()
    topics, _probabilities = topic_model.fit_transform(
        prepared_df["model_text"].tolist()
    )
    runtime_seconds = time.perf_counter() - start_time

    topic_info_df = topic_model.get_topic_info()
    topic_name_lookup = {
        int(row["Topic"]): str(row.get("Name", ""))
        for row in topic_info_df.to_dict(orient="records")
    }

    document_topics_df = prepared_df[
        [
            "record_id",
            "domain",
            "final_url",
            "rss_title",
            "word_count",
            "language",
            "model_word_count",
        ]
    ].copy()
    document_topics_df["topic_id"] = topics
    document_topics_df["topic_name"] = [
        topic_name_lookup.get(int(topic_id), "")
        for topic_id in document_topics_df["topic_id"].tolist()
    ]
    document_topics_df["topic_keywords"] = [
        topic_keywords_for(topic_model, int(topic_id))
        for topic_id in document_topics_df["topic_id"].tolist()
    ]
    document_topics_df["is_outlier"] = document_topics_df["topic_id"].eq(-1)
    document_topics_df = document_topics_df[DOCUMENT_TOPIC_COLUMNS].copy()

    topic_info_out = serialize_object_columns(topic_info_df)
    summary_payload = build_summary_payload(
        prepared_df=prepared_df,
        source_rows=len(source_df),
        topic_info_df=topic_info_df,
        topic_cfg=topic_cfg,
        runtime_seconds=runtime_seconds,
    )

    write_dataframe(document_topics_df, output_doc_csv, output_doc_parquet)
    ensure_dir(output_topic_info_csv.parent)
    topic_info_out.to_csv(output_topic_info_csv, index=False)
    output_summary_json.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(
        "BERTopic complete: modeled=%s topics=%s outliers=%s runtime=%s",
        len(prepared_df),
        summary_payload["unique_topics_excluding_outlier"],
        summary_payload["outlier_documents"],
        summary_payload["runtime_human"],
    )


if __name__ == "__main__":
    main()
