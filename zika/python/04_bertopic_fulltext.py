from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from dotenv import load_dotenv
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
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
    "embedding_model",
    "analysis_group",
    "analysis_variant",
    "topic_id",
    "topic_name",
    "topic_keywords",
    "is_outlier",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BERTopic analyses on the Zika full-text subset."
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


def slugify_model_name(model_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", model_name.strip())
    cleaned = cleaned.strip("_").lower()
    return cleaned or "model"


def normalize_language(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


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
    prepared["domain"] = prepared.get("domain", "").fillna("").astype(str)
    prepared["final_url"] = prepared.get("final_url", "").fillna("").astype(str)

    prepared = prepared[
        prepared["full_text"].str.strip().ne("")
        & (prepared["word_count"] >= min_word_count)
        & prepared["language_norm"].map(
            lambda value: language_allowed(
                value,
                allowed_prefixes=allowed_prefixes,
                allow_blank_language=allow_blank_language,
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


def topic_distribution_stats(topic_info_df: pd.DataFrame) -> dict[str, Any]:
    non_outlier = topic_info_df[topic_info_df["Topic"] != -1].copy()
    if non_outlier.empty:
        return {
            "largest_topic_id": None,
            "largest_topic_count": 0,
            "largest_topic_share": 0.0,
            "topic_entropy": 0.0,
            "normalized_topic_entropy": 0.0,
        }

    counts = non_outlier["Count"].astype(int).to_numpy()
    shares = counts / counts.sum()
    entropy = float(-np.sum(shares * np.log(shares)))
    normalized_entropy = (
        float(entropy / math.log(len(shares))) if len(shares) > 1 else 0.0
    )
    largest_idx = int(np.argmax(counts))

    return {
        "largest_topic_id": int(non_outlier.iloc[largest_idx]["Topic"]),
        "largest_topic_count": int(counts[largest_idx]),
        "largest_topic_share": float(shares[largest_idx]),
        "topic_entropy": entropy,
        "normalized_topic_entropy": normalized_entropy,
    }


def topic_matrix_with_ids(topic_model: BERTopic, topic_info_df: pd.DataFrame) -> tuple[list[int], Any]:
    topic_ids = (
        topic_info_df.loc[topic_info_df["Topic"] != -1, "Topic"]
        .astype(int)
        .sort_values()
        .tolist()
    )
    matrix = getattr(topic_model, "c_tf_idf_", None)
    return topic_ids, matrix


def lexical_distinctiveness_stats(
    topic_model: BERTopic, topic_info_df: pd.DataFrame
) -> dict[str, Any]:
    topic_ids, matrix = topic_matrix_with_ids(topic_model, topic_info_df)
    if matrix is None or not topic_ids:
        return {
            "mean_topic_ctfidf_cosine": None,
            "max_topic_ctfidf_cosine": None,
            "unique_representation_terms": 0,
        }

    unique_terms: set[str] = set()
    if "Representation" in topic_info_df.columns:
        for value in topic_info_df["Representation"].tolist():
            if isinstance(value, list):
                unique_terms.update(str(item) for item in value if str(item).strip())
            elif pd.notna(value):
                text = str(value).strip()
                if text:
                    unique_terms.update(
                        item.strip() for item in text.split("|") if item.strip()
                    )

    if len(topic_ids) <= 1:
        return {
            "mean_topic_ctfidf_cosine": None,
            "max_topic_ctfidf_cosine": None,
            "unique_representation_terms": len(unique_terms),
        }

    similarity = cosine_similarity(matrix)
    upper = similarity[np.triu_indices_from(similarity, k=1)]
    return {
        "mean_topic_ctfidf_cosine": round(float(np.mean(upper)), 6),
        "max_topic_ctfidf_cosine": round(float(np.max(upper)), 6),
        "unique_representation_terms": int(len(unique_terms)),
    }


def build_summary_payload(
    prepared_df: pd.DataFrame,
    source_rows: int,
    topic_model: BERTopic,
    topic_info_df: pd.DataFrame,
    topic_cfg: dict[str, Any],
    runtime_seconds: float,
    embedding_model_name: str,
    *,
    analysis_group: str,
    analysis_variant: str,
    component_settings: dict[str, Any],
    notes: str = "",
    stage_timings: dict[str, float] | None = None,
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
                        str(item) for item in (record.get("Representation") or [])
                    ],
                }
            )

    outlier_row = topic_info_df[topic_info_df["Topic"] == -1]
    outlier_documents = (
        int(outlier_row["Count"].iloc[0]) if not outlier_row.empty else 0
    )
    stats = topic_distribution_stats(topic_info_df)
    lexical_stats = lexical_distinctiveness_stats(topic_model, topic_info_df)

    return {
        "generated_at": utc_now_iso(),
        "analysis_group": analysis_group,
        "analysis_variant": analysis_variant,
        "embedding_model": embedding_model_name,
        "documents_input": int(source_rows),
        "documents_modeled": int(len(prepared_df)),
        "documents_excluded": int(source_rows - len(prepared_df)),
        "unique_topics_excluding_outlier": int(len(non_outlier)),
        "outlier_documents": outlier_documents,
        "runtime_seconds": round(runtime_seconds, 3),
        "runtime_human": format_elapsed(runtime_seconds),
        "largest_topic_id": stats["largest_topic_id"],
        "largest_topic_count": stats["largest_topic_count"],
        "largest_topic_share": round(stats["largest_topic_share"], 6),
        "topic_entropy": round(stats["topic_entropy"], 6),
        "normalized_topic_entropy": round(stats["normalized_topic_entropy"], 6),
        "mean_topic_ctfidf_cosine": lexical_stats["mean_topic_ctfidf_cosine"],
        "max_topic_ctfidf_cosine": lexical_stats["max_topic_ctfidf_cosine"],
        "unique_representation_terms": lexical_stats["unique_representation_terms"],
        "component_settings": component_settings,
        "notes": notes,
        "stage_timings": {
            key: round(float(value), 3) for key, value in (stage_timings or {}).items()
        },
        "config": {
            "min_word_count": int(topic_cfg.get("min_word_count", 120)),
            "max_words_per_document": int(topic_cfg.get("max_words_per_document", 800)),
            "top_n_words": int(topic_cfg.get("top_n_words", 10)),
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


def build_vectorizer_model(
    topic_cfg: dict[str, Any],
    *,
    variant: str,
    vocabulary: dict[str, int] | None = None,
) -> CountVectorizer:
    if variant == "basic_unigram":
        return CountVectorizer(
            stop_words=None,
            min_df=1,
            ngram_range=(1, 1),
            vocabulary=vocabulary,
        )
    if variant == "tuned_bigram":
        return CountVectorizer(
            stop_words=str(topic_cfg.get("vectorizer_stop_words", "english")),
            min_df=max(1, int(topic_cfg.get("min_df", 2))),
            ngram_range=(
                int(topic_cfg.get("ngram_min", 1)),
                int(topic_cfg.get("ngram_max", 2)),
            ),
            vocabulary=vocabulary,
        )
    raise ValueError(f"Unsupported vectorizer variant: {variant}")


def build_ctfidf_model(variant: str) -> ClassTfidfTransformer:
    if variant == "standard":
        return ClassTfidfTransformer()
    if variant == "reduce_frequent":
        return ClassTfidfTransformer(reduce_frequent_words=True)
    if variant == "bm25_reduce_frequent":
        return ClassTfidfTransformer(
            bm25_weighting=True,
            reduce_frequent_words=True,
        )
    raise ValueError(f"Unsupported c-TF-IDF variant: {variant}")


def build_representation_model(
    topic_cfg: dict[str, Any], variant: str
) -> Any | None:
    top_n_words = int(topic_cfg.get("top_n_words", 10))
    if variant == "ctfidf_only":
        return None
    if variant == "keybert_inspired":
        return KeyBERTInspired(
            top_n_words=top_n_words,
            random_state=int(topic_cfg.get("seed", 42)),
        )
    if variant == "mmr_diverse":
        return MaximalMarginalRelevance(
            diversity=float(topic_cfg.get("representation_mmr_diversity", 0.3)),
            top_n_words=top_n_words,
        )
    raise ValueError(f"Unsupported representation variant: {variant}")


def build_dimensionality_model(
    topic_cfg: dict[str, Any], *, variant: str, seed: int
) -> Any:
    components = int(topic_cfg.get("umap_components", 5))
    if variant == "pca":
        return PCA(
            n_components=int(topic_cfg.get("pca_components", components)),
            random_state=seed,
        )
    if variant == "umap":
        return UMAP(
            n_neighbors=int(topic_cfg.get("umap_neighbors", 15)),
            n_components=components,
            min_dist=float(topic_cfg.get("umap_min_dist", 0.0)),
            metric="cosine",
            random_state=seed,
        )
    raise ValueError(f"Unsupported dimensionality reduction variant: {variant}")


def build_cluster_model(
    topic_cfg: dict[str, Any],
    *,
    variant: str,
    min_topic_size: int,
    seed: int,
    kmeans_clusters: int | None = None,
) -> Any:
    if variant == "kmeans":
        clusters = int(
            kmeans_clusters
            or topic_cfg.get("kmeans_clusters", max(2, min_topic_size // 2))
        )
        return KMeans(
            n_clusters=max(2, clusters),
            random_state=seed,
            n_init=10,
        )
    if variant == "hdbscan":
        return HDBSCAN(
            min_cluster_size=min_topic_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
    raise ValueError(f"Unsupported clustering variant: {variant}")


def topic_rows_from_info(
    topic_info_df: pd.DataFrame,
    *,
    analysis_group: str,
    analysis_variant: str,
    embedding_model: str,
) -> pd.DataFrame:
    topic_rows = serialize_object_columns(topic_info_df).copy()
    topic_rows.insert(0, "embedding_model", embedding_model)
    topic_rows.insert(0, "analysis_variant", analysis_variant)
    topic_rows.insert(0, "analysis_group", analysis_group)
    return topic_rows


def document_topics_frame(
    prepared_df: pd.DataFrame,
    topic_model: BERTopic,
    topics: list[int],
    *,
    embedding_model: str,
    analysis_group: str,
    analysis_variant: str,
) -> pd.DataFrame:
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
    document_topics_df["embedding_model"] = embedding_model
    document_topics_df["analysis_group"] = analysis_group
    document_topics_df["analysis_variant"] = analysis_variant
    document_topics_df["topic_id"] = [int(topic_id) for topic_id in topics]
    document_topics_df["topic_name"] = [
        topic_name_lookup.get(int(topic_id), "")
        for topic_id in document_topics_df["topic_id"].tolist()
    ]
    document_topics_df["topic_keywords"] = [
        topic_keywords_for(topic_model, int(topic_id))
        for topic_id in document_topics_df["topic_id"].tolist()
    ]
    document_topics_df["is_outlier"] = document_topics_df["topic_id"].eq(-1)
    return document_topics_df[DOCUMENT_TOPIC_COLUMNS].copy()


def encode_documents(
    documents: list[str],
    embedding_model_name: str,
    logger: logging.Logger,
) -> tuple[SentenceTransformer, np.ndarray, float]:
    logger.info("Encoding %s documents with '%s'", len(documents), embedding_model_name)
    start_time = time.perf_counter()
    embedding_model = SentenceTransformer(embedding_model_name)
    embeddings = embedding_model.encode(
        documents,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    runtime_seconds = time.perf_counter() - start_time
    logger.info(
        "Embedding complete for '%s' in %s",
        embedding_model_name,
        format_elapsed(runtime_seconds),
    )
    return embedding_model, embeddings, runtime_seconds


def fit_full_model_run(
    prepared_df: pd.DataFrame,
    source_rows: int,
    topic_cfg: dict[str, Any],
    embedding_model_name: str,
    embedding_backend: SentenceTransformer,
    embeddings: np.ndarray,
    *,
    analysis_group: str,
    analysis_variant: str,
    dim_variant: str,
    cluster_variant: str,
    vectorizer_variant: str,
    ctfidf_variant: str,
    representation_variant: str,
    use_shared_vocabulary: bool,
    seed: int,
    logger: logging.Logger,
    embedding_seconds: float = 0.0,
    notes: str = "",
    kmeans_clusters: int | None = None,
) -> dict[str, Any]:
    min_topic_size = int(topic_cfg.get("min_topic_size", 12))
    vectorizer_model = build_vectorizer_model(
        topic_cfg,
        variant=vectorizer_variant,
        vocabulary=None,
    )
    ctfidf_model = build_ctfidf_model(ctfidf_variant)
    representation_model = build_representation_model(topic_cfg, representation_variant)
    umap_model = build_dimensionality_model(topic_cfg, variant=dim_variant, seed=seed)
    hdbscan_model = build_cluster_model(
        topic_cfg,
        variant=cluster_variant,
        min_topic_size=min_topic_size,
        seed=seed,
        kmeans_clusters=kmeans_clusters,
    )

    topic_model = BERTopic(
        embedding_model=embedding_backend,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        top_n_words=int(topic_cfg.get("top_n_words", 10)),
        min_topic_size=min_topic_size,
        calculate_probabilities=False,
        verbose=True,
    )

    logger.info(
        "BERTopic analysis '%s/%s' started for '%s'",
        analysis_group,
        analysis_variant,
        embedding_model_name,
    )
    fit_start = time.perf_counter()
    topics, _probabilities = topic_model.fit_transform(
        prepared_df["model_text"].tolist(),
        embeddings=embeddings,
    )
    fit_seconds = time.perf_counter() - fit_start

    component_settings = {
        "dimensionality_reduction": dim_variant,
        "clustering": cluster_variant,
        "vectorizer": vectorizer_variant,
        "ctfidf": ctfidf_variant,
        "representation": representation_variant,
        "shared_vocabulary": bool(use_shared_vocabulary),
        "document_assignments_fixed": False,
    }
    if cluster_variant == "kmeans":
        component_settings["kmeans_clusters"] = int(
            kmeans_clusters or topic_cfg.get("kmeans_clusters", 0)
        )

    summary_payload = build_summary_payload(
        prepared_df=prepared_df,
        source_rows=source_rows,
        topic_model=topic_model,
        topic_info_df=topic_model.get_topic_info(),
        topic_cfg=topic_cfg,
        runtime_seconds=embedding_seconds + fit_seconds,
        embedding_model_name=embedding_model_name,
        analysis_group=analysis_group,
        analysis_variant=analysis_variant,
        component_settings=component_settings,
        notes=notes,
        stage_timings={
            "embedding_encode": embedding_seconds,
            "bertopic_fit_transform": fit_seconds,
        },
    )
    document_topics_df = document_topics_frame(
        prepared_df,
        topic_model,
        topics=[int(topic) for topic in topics],
        embedding_model=embedding_model_name,
        analysis_group=analysis_group,
        analysis_variant=analysis_variant,
    )
    topic_info_df = topic_model.get_topic_info()

    logger.info(
        "BERTopic analysis '%s/%s' complete: topics=%s outliers=%s runtime=%s",
        analysis_group,
        analysis_variant,
        summary_payload["unique_topics_excluding_outlier"],
        summary_payload["outlier_documents"],
        summary_payload["runtime_human"],
    )
    return {
        "embedding_model": embedding_model_name,
        "slug": slugify_model_name(embedding_model_name),
        "analysis_group": analysis_group,
        "analysis_variant": analysis_variant,
        "topic_model": topic_model,
        "topics": [int(topic) for topic in topics],
        "document_topics_df": document_topics_df,
        "topic_info_df": topic_info_df,
        "topic_info_out": serialize_object_columns(topic_info_df),
        "topic_rows_df": topic_rows_from_info(
            topic_info_df,
            analysis_group=analysis_group,
            analysis_variant=analysis_variant,
            embedding_model=embedding_model_name,
        ),
        "summary_payload": summary_payload,
        "summary_row": {
            "analysis_group": analysis_group,
            "analysis_variant": analysis_variant,
            "embedding_model": embedding_model_name,
            "documents_input": summary_payload["documents_input"],
            "documents_modeled": summary_payload["documents_modeled"],
            "documents_excluded": summary_payload["documents_excluded"],
            "unique_topics_excluding_outlier": summary_payload[
                "unique_topics_excluding_outlier"
            ],
            "outlier_documents": summary_payload["outlier_documents"],
            "largest_topic_count": summary_payload["largest_topic_count"],
            "largest_topic_share": summary_payload["largest_topic_share"],
            "normalized_topic_entropy": summary_payload["normalized_topic_entropy"],
            "mean_topic_ctfidf_cosine": summary_payload["mean_topic_ctfidf_cosine"],
            "max_topic_ctfidf_cosine": summary_payload["max_topic_ctfidf_cosine"],
            "unique_representation_terms": summary_payload[
                "unique_representation_terms"
            ],
            "runtime_seconds": summary_payload["runtime_seconds"],
            "runtime_human": summary_payload["runtime_human"],
            "embedding_seconds": round(float(embedding_seconds), 3),
            "analysis_seconds": round(float(fit_seconds), 3),
            "analysis_human": format_elapsed(fit_seconds),
            "notes": notes,
            **component_settings,
        },
        "stage_timing_rows": [
            {
                "analysis_group": analysis_group,
                "analysis_variant": analysis_variant,
                "embedding_model": embedding_model_name,
                "stage_name": "embedding_encode",
                "runtime_seconds": round(float(embedding_seconds), 3),
                "runtime_human": format_elapsed(embedding_seconds),
            },
            {
                "analysis_group": analysis_group,
                "analysis_variant": analysis_variant,
                "embedding_model": embedding_model_name,
                "stage_name": "bertopic_fit_transform",
                "runtime_seconds": round(float(fit_seconds), 3),
                "runtime_human": format_elapsed(fit_seconds),
            },
        ],
    }


def run_lexical_analysis(
    base_run: dict[str, Any],
    prepared_df: pd.DataFrame,
    source_rows: int,
    topic_cfg: dict[str, Any],
    *,
    analysis_group: str,
    analysis_variant: str,
    vectorizer_variant: str,
    ctfidf_variant: str,
    representation_variant: str,
    notes: str,
) -> dict[str, Any]:
    topic_model = base_run["topic_model"]
    topics = [int(topic) for topic in base_run["topics"]]
    start_time = time.perf_counter()
    topic_model.update_topics(
        prepared_df["model_text"].tolist(),
        topics=topics,
        vectorizer_model=build_vectorizer_model(topic_cfg, variant=vectorizer_variant),
        ctfidf_model=build_ctfidf_model(ctfidf_variant),
        representation_model=build_representation_model(topic_cfg, representation_variant),
    )
    update_seconds = time.perf_counter() - start_time

    component_settings = {
        "dimensionality_reduction": "umap",
        "clustering": "hdbscan",
        "vectorizer": vectorizer_variant,
        "ctfidf": ctfidf_variant,
        "representation": representation_variant,
        "shared_vocabulary": False,
        "document_assignments_fixed": True,
    }
    topic_info_df = topic_model.get_topic_info()
    summary_payload = build_summary_payload(
        prepared_df=prepared_df,
        source_rows=source_rows,
        topic_model=topic_model,
        topic_info_df=topic_info_df,
        topic_cfg=topic_cfg,
        runtime_seconds=update_seconds,
        embedding_model_name=base_run["embedding_model"],
        analysis_group=analysis_group,
        analysis_variant=analysis_variant,
        component_settings=component_settings,
        notes=notes,
        stage_timings={"update_topics": update_seconds},
    )
    document_topics_df = document_topics_frame(
        prepared_df,
        topic_model,
        topics=topics,
        embedding_model=base_run["embedding_model"],
        analysis_group=analysis_group,
        analysis_variant=analysis_variant,
    )

    return {
        "analysis_group": analysis_group,
        "analysis_variant": analysis_variant,
        "embedding_model": base_run["embedding_model"],
        "document_topics_df": document_topics_df,
        "topic_info_df": topic_info_df,
        "topic_rows_df": topic_rows_from_info(
            topic_info_df,
            analysis_group=analysis_group,
            analysis_variant=analysis_variant,
            embedding_model=base_run["embedding_model"],
        ),
        "summary_payload": summary_payload,
        "summary_row": {
            "analysis_group": analysis_group,
            "analysis_variant": analysis_variant,
            "embedding_model": base_run["embedding_model"],
            "documents_input": summary_payload["documents_input"],
            "documents_modeled": summary_payload["documents_modeled"],
            "documents_excluded": summary_payload["documents_excluded"],
            "unique_topics_excluding_outlier": summary_payload[
                "unique_topics_excluding_outlier"
            ],
            "outlier_documents": summary_payload["outlier_documents"],
            "largest_topic_count": summary_payload["largest_topic_count"],
            "largest_topic_share": summary_payload["largest_topic_share"],
            "normalized_topic_entropy": summary_payload["normalized_topic_entropy"],
            "mean_topic_ctfidf_cosine": summary_payload["mean_topic_ctfidf_cosine"],
            "max_topic_ctfidf_cosine": summary_payload["max_topic_ctfidf_cosine"],
            "unique_representation_terms": summary_payload[
                "unique_representation_terms"
            ],
            "runtime_seconds": summary_payload["runtime_seconds"],
            "runtime_human": summary_payload["runtime_human"],
            "embedding_seconds": 0.0,
            "analysis_seconds": summary_payload["runtime_seconds"],
            "analysis_human": summary_payload["runtime_human"],
            "notes": notes,
            **component_settings,
        },
        "stage_timing_rows": [
            {
                "analysis_group": analysis_group,
                "analysis_variant": analysis_variant,
                "embedding_model": base_run["embedding_model"],
                "stage_name": "update_topics",
                "runtime_seconds": round(float(update_seconds), 3),
                "runtime_human": format_elapsed(update_seconds),
            }
        ],
    }


def write_model_outputs(
    zika_root: Path, model_run: dict[str, Any], is_primary: bool
) -> None:
    slug = str(model_run["slug"])
    doc_csv = zika_root / "data" / "final" / f"zika_bertopic_{slug}_document_topics.csv"
    doc_parquet = (
        zika_root / "data" / "final" / f"zika_bertopic_{slug}_document_topics.parquet"
    )
    topic_csv = zika_root / "data" / "final" / f"zika_bertopic_{slug}_topic_info.csv"
    summary_json = zika_root / "data" / "final" / f"zika_bertopic_{slug}_summary.json"

    write_dataframe(model_run["document_topics_df"], doc_csv, doc_parquet)
    model_run["topic_info_out"].to_csv(topic_csv, index=False)
    summary_json.write_text(
        json.dumps(model_run["summary_payload"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if is_primary:
        write_dataframe(
            model_run["document_topics_df"],
            zika_root / "data" / "final" / "zika_bertopic_document_topics.csv",
            zika_root / "data" / "final" / "zika_bertopic_document_topics.parquet",
        )
        model_run["topic_info_out"].to_csv(
            zika_root / "data" / "final" / "zika_bertopic_topic_info.csv",
            index=False,
        )
        (zika_root / "data" / "final" / "zika_bertopic_summary.json").write_text(
            json.dumps(model_run["summary_payload"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def build_topic_pseudo_documents(
    run: dict[str, Any], prepared_df: pd.DataFrame
) -> pd.DataFrame:
    topic_lookup = (
        run["topic_info_df"][["Topic", "Name"]]
        .rename(columns={"Topic": "topic_id", "Name": "topic_name"})
        .copy()
    )
    return (
        run["document_topics_df"][["record_id", "topic_id"]]
        .merge(
            prepared_df[["record_id", "model_text"]],
            on="record_id",
            how="inner",
        )
        .query("topic_id != -1")
        .groupby("topic_id", as_index=False)
        .agg(topic_text=("model_text", " ".join))
        .merge(topic_lookup, on="topic_id", how="left")
        .sort_values("topic_id")
        .reset_index(drop=True)
    )


def build_topic_match_rows(
    source_run: dict[str, Any],
    target_run: dict[str, Any],
    prepared_df: pd.DataFrame,
    topic_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    source_docs = build_topic_pseudo_documents(source_run, prepared_df)
    target_docs = build_topic_pseudo_documents(target_run, prepared_df)
    if source_docs.empty or target_docs.empty:
        return []

    vectorizer = CountVectorizer(
        stop_words=str(topic_cfg.get("vectorizer_stop_words", "english")),
        min_df=1,
        ngram_range=(
            int(topic_cfg.get("ngram_min", 1)),
            int(topic_cfg.get("ngram_max", 2)),
        ),
    )
    combined_docs = pd.concat(
        [
            source_docs[["topic_text"]],
            target_docs[["topic_text"]],
        ],
        ignore_index=True,
    )
    counts = vectorizer.fit_transform(combined_docs["topic_text"].tolist())
    ctfidf = ClassTfidfTransformer(reduce_frequent_words=True).fit_transform(counts)

    source_matrix = ctfidf[: len(source_docs)]
    target_matrix = ctfidf[len(source_docs) :]
    similarity = cosine_similarity(source_matrix, target_matrix)

    rows: list[dict[str, Any]] = []
    for source_idx, source_record in source_docs.reset_index(drop=True).iterrows():
        best_idx = int(np.argmax(similarity[source_idx]))
        target_record = target_docs.iloc[best_idx]
        rows.append(
            {
                "source_model": source_run["embedding_model"],
                "target_model": target_run["embedding_model"],
                "source_topic_id": int(source_record["topic_id"]),
                "source_topic_count": int(
                    source_run["topic_info_df"]
                    .set_index("Topic")
                    .loc[int(source_record["topic_id"]), "Count"]
                ),
                "source_topic_name": str(source_record["topic_name"]),
                "target_topic_id": int(target_record["topic_id"]),
                "target_topic_count": int(
                    target_run["topic_info_df"]
                    .set_index("Topic")
                    .loc[int(target_record["topic_id"]), "Count"]
                ),
                "target_topic_name": str(target_record["topic_name"]),
                "ctfidf_cosine_similarity": round(
                    float(similarity[source_idx, best_idx]), 6
                ),
            }
        )
    return rows


def build_model_comparison(
    model_runs: list[dict[str, Any]],
    prepared_df: pd.DataFrame,
    topic_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    for run in model_runs:
        payload = run["summary_payload"]
        summary_rows.append(
            {
                "embedding_model": run["embedding_model"],
                "documents_modeled": payload["documents_modeled"],
                "unique_topics_excluding_outlier": payload[
                    "unique_topics_excluding_outlier"
                ],
                "outlier_documents": payload["outlier_documents"],
                "largest_topic_count": payload["largest_topic_count"],
                "largest_topic_share": payload["largest_topic_share"],
                "normalized_topic_entropy": payload["normalized_topic_entropy"],
                "mean_topic_ctfidf_cosine": payload["mean_topic_ctfidf_cosine"],
                "max_topic_ctfidf_cosine": payload["max_topic_ctfidf_cosine"],
                "unique_representation_terms": payload[
                    "unique_representation_terms"
                ],
                "runtime_seconds": payload["runtime_seconds"],
                "runtime_human": payload["runtime_human"],
                "embedding_seconds": run["summary_row"]["embedding_seconds"],
                "analysis_seconds": run["summary_row"]["analysis_seconds"],
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    pairwise_metrics: list[dict[str, Any]] = []
    topic_match_rows: list[dict[str, Any]] = []
    for i, left_run in enumerate(model_runs):
        for right_run in model_runs[i + 1 :]:
            left_docs = left_run["document_topics_df"][
                ["record_id", "topic_id", "is_outlier"]
            ].rename(
                columns={
                    "topic_id": "topic_id_left",
                    "is_outlier": "is_outlier_left",
                }
            )
            right_docs = right_run["document_topics_df"][
                ["record_id", "topic_id", "is_outlier"]
            ].rename(
                columns={
                    "topic_id": "topic_id_right",
                    "is_outlier": "is_outlier_right",
                }
            )
            merged = left_docs.merge(right_docs, on="record_id", how="inner")

            ari = adjusted_rand_score(
                merged["topic_id_left"], merged["topic_id_right"]
            )
            nmi = normalized_mutual_info_score(
                merged["topic_id_left"], merged["topic_id_right"]
            )
            non_outlier = merged[
                ~merged["is_outlier_left"] & ~merged["is_outlier_right"]
            ].copy()
            if len(non_outlier) > 1:
                ari_non_outlier = adjusted_rand_score(
                    non_outlier["topic_id_left"], non_outlier["topic_id_right"]
                )
                nmi_non_outlier = normalized_mutual_info_score(
                    non_outlier["topic_id_left"], non_outlier["topic_id_right"]
                )
            else:
                ari_non_outlier = np.nan
                nmi_non_outlier = np.nan

            pairwise_metrics.append(
                {
                    "model_left": left_run["embedding_model"],
                    "model_right": right_run["embedding_model"],
                    "paired_documents": int(len(merged)),
                    "adjusted_rand_index": round(float(ari), 6),
                    "normalized_mutual_info": round(float(nmi), 6),
                    "paired_non_outlier_documents": int(len(non_outlier)),
                    "adjusted_rand_index_non_outlier": (
                        round(float(ari_non_outlier), 6)
                        if not pd.isna(ari_non_outlier)
                        else None
                    ),
                    "normalized_mutual_info_non_outlier": (
                        round(float(nmi_non_outlier), 6)
                        if not pd.isna(nmi_non_outlier)
                        else None
                    ),
                }
            )
            topic_match_rows.extend(
                build_topic_match_rows(left_run, right_run, prepared_df, topic_cfg)
            )
            topic_match_rows.extend(
                build_topic_match_rows(right_run, left_run, prepared_df, topic_cfg)
            )

    comparison_payload = {
        "generated_at": utc_now_iso(),
        "models": [run["embedding_model"] for run in model_runs],
        "summary_rows": summary_rows,
        "pairwise_metrics": pairwise_metrics,
    }
    topic_match_df = pd.DataFrame(topic_match_rows)
    return summary_df, comparison_payload, topic_match_df


def embedding_models_from_config(topic_cfg: dict[str, Any]) -> list[str]:
    configured = topic_cfg.get("embedding_models")
    if isinstance(configured, list) and configured:
        values = [str(value).strip() for value in configured if str(value).strip()]
        if values:
            return values
    fallback = str(topic_cfg.get("embedding_model", "all-MiniLM-L6-v2")).strip()
    return [fallback]


def module_variants_from_config(topic_cfg: dict[str, Any], key: str, default: list[str]) -> list[str]:
    configured = topic_cfg.get(key)
    if isinstance(configured, list) and configured:
        values = [str(value).strip() for value in configured if str(value).strip()]
        if values:
            return values
    return default


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config_path = (repo_root / args.config).resolve()
    zika_root = config_path.parent.parent
    cfg = load_config(config_path)
    topic_cfg = cfg.get("topic_model", {})

    log_path = zika_root / "logs" / "04_bertopic_fulltext.log"
    logger = setup_logger(log_path)

    input_csv = zika_root / str(
        topic_cfg.get("input_csv", "data/final/zika_fulltext.csv")
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

    embedding_models = embedding_models_from_config(topic_cfg)
    primary_embedding_model = str(
        topic_cfg.get("primary_embedding_model", embedding_models[0])
    ).strip()
    if primary_embedding_model not in embedding_models:
        embedding_models = [primary_embedding_model, *embedding_models]

    embedding_cache: dict[str, tuple[SentenceTransformer, np.ndarray, float]] = {}
    model_runs: list[dict[str, Any]] = []
    for embedding_model_name in embedding_models:
        embedding_backend, embeddings, embedding_seconds = encode_documents(
            prepared_df["model_text"].tolist(),
            embedding_model_name,
            logger,
        )
        embedding_cache[embedding_model_name] = (
            embedding_backend,
            embeddings,
            embedding_seconds,
        )
        model_runs.append(
            fit_full_model_run(
                prepared_df=prepared_df,
                source_rows=len(source_df),
                topic_cfg=topic_cfg,
                embedding_model_name=embedding_model_name,
                embedding_backend=embedding_backend,
                embeddings=embeddings,
                analysis_group="embedding_model",
                analysis_variant=embedding_model_name,
                dim_variant="umap",
                cluster_variant="hdbscan",
                vectorizer_variant="tuned_bigram",
                ctfidf_variant="reduce_frequent",
                representation_variant="keybert_inspired",
                use_shared_vocabulary=False,
                seed=seed,
                logger=logger,
                embedding_seconds=embedding_seconds,
                notes="Full BERTopic pipeline with shared vectorizer settings for cross-embedding comparison.",
            )
        )

    for run in model_runs:
        write_model_outputs(
            zika_root=zika_root,
            model_run=run,
            is_primary=run["embedding_model"] == primary_embedding_model,
        )

    comparison_df, comparison_payload, topic_match_df = build_model_comparison(
        model_runs,
        prepared_df,
        topic_cfg,
    )
    comparison_csv = zika_root / "data" / "final" / "zika_bertopic_model_comparison.csv"
    comparison_json = (
        zika_root / "data" / "final" / "zika_bertopic_model_comparison.json"
    )
    topic_match_csv = zika_root / "data" / "final" / "zika_bertopic_topic_matches.csv"

    ensure_dir(comparison_csv.parent)
    comparison_df.to_csv(comparison_csv, index=False)
    comparison_json.write_text(
        json.dumps(comparison_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    topic_match_df.to_csv(topic_match_csv, index=False)

    primary_run = next(
        run for run in model_runs if run["embedding_model"] == primary_embedding_model
    )
    primary_embedding_backend, primary_embeddings, _primary_embedding_seconds = embedding_cache[
        primary_embedding_model
    ]
    baseline_topic_count = int(
        primary_run["summary_payload"]["unique_topics_excluding_outlier"]
    )
    kmeans_clusters = int(topic_cfg.get("kmeans_clusters", max(2, baseline_topic_count)))

    analysis_runs: list[dict[str, Any]] = [
        {
            "summary_row": {
                **primary_run["summary_row"],
                "analysis_group": "dimensionality_reduction",
                "analysis_variant": "umap",
            },
            "topic_rows_df": primary_run["topic_rows_df"].assign(
                analysis_group="dimensionality_reduction",
                analysis_variant="umap",
            ),
            "stage_timing_rows": [
                {
                    **row,
                    "analysis_group": "dimensionality_reduction",
                    "analysis_variant": "umap",
                }
                for row in primary_run["stage_timing_rows"]
            ],
        },
        {
            "summary_row": {
                **primary_run["summary_row"],
                "analysis_group": "clustering",
                "analysis_variant": "hdbscan",
            },
            "topic_rows_df": primary_run["topic_rows_df"].assign(
                analysis_group="clustering",
                analysis_variant="hdbscan",
            ),
            "stage_timing_rows": [
                {
                    **row,
                    "analysis_group": "clustering",
                    "analysis_variant": "hdbscan",
                }
                for row in primary_run["stage_timing_rows"]
            ],
        },
    ]

    for dim_variant in module_variants_from_config(
        topic_cfg,
        "dimensionality_reduction_variants",
        ["umap", "pca"],
    ):
        if dim_variant == "umap":
            continue
        analysis_runs.append(
            fit_full_model_run(
                prepared_df=prepared_df,
                source_rows=len(source_df),
                topic_cfg=topic_cfg,
                embedding_model_name=primary_embedding_model,
                embedding_backend=primary_embedding_backend,
                embeddings=primary_embeddings,
                analysis_group="dimensionality_reduction",
                analysis_variant=dim_variant,
                dim_variant=dim_variant,
                cluster_variant="hdbscan",
                vectorizer_variant="tuned_bigram",
                ctfidf_variant="reduce_frequent",
                representation_variant="keybert_inspired",
                use_shared_vocabulary=False,
                seed=seed,
                logger=logger,
                embedding_seconds=0.0,
                notes="Primary embedding reused; only dimensionality reduction changes.",
            )
        )

    for cluster_variant in module_variants_from_config(
        topic_cfg,
        "clustering_variants",
        ["hdbscan", "kmeans"],
    ):
        if cluster_variant == "hdbscan":
            continue
        analysis_runs.append(
            fit_full_model_run(
                prepared_df=prepared_df,
                source_rows=len(source_df),
                topic_cfg=topic_cfg,
                embedding_model_name=primary_embedding_model,
                embedding_backend=primary_embedding_backend,
                embeddings=primary_embeddings,
                analysis_group="clustering",
                analysis_variant=cluster_variant,
                dim_variant="umap",
                cluster_variant=cluster_variant,
                vectorizer_variant="tuned_bigram",
                ctfidf_variant="reduce_frequent",
                representation_variant="keybert_inspired",
                use_shared_vocabulary=False,
                seed=seed,
                logger=logger,
                embedding_seconds=0.0,
                notes="Primary embedding reused; only clustering changes.",
                kmeans_clusters=kmeans_clusters,
            )
        )

    for vectorizer_variant in module_variants_from_config(
        topic_cfg,
        "vectorizer_variants",
        ["basic_unigram", "tuned_bigram"],
    ):
        analysis_runs.append(
            run_lexical_analysis(
                base_run=primary_run,
                prepared_df=prepared_df,
                source_rows=len(source_df),
                topic_cfg=topic_cfg,
                analysis_group="vectorizer",
                analysis_variant=vectorizer_variant,
                vectorizer_variant=vectorizer_variant,
                ctfidf_variant="reduce_frequent",
                representation_variant="keybert_inspired",
                notes="MiniLM topic assignments fixed; update_topics reruns only lexical stages.",
            )
        )

    for ctfidf_variant in module_variants_from_config(
        topic_cfg,
        "ctfidf_variants",
        ["standard", "reduce_frequent", "bm25_reduce_frequent"],
    ):
        analysis_runs.append(
            run_lexical_analysis(
                base_run=primary_run,
                prepared_df=prepared_df,
                source_rows=len(source_df),
                topic_cfg=topic_cfg,
                analysis_group="ctfidf",
                analysis_variant=ctfidf_variant,
                vectorizer_variant="tuned_bigram",
                ctfidf_variant=ctfidf_variant,
                representation_variant="keybert_inspired",
                notes="MiniLM topic assignments fixed; c-TF-IDF settings updated on the same clusters.",
            )
        )

    for representation_variant in module_variants_from_config(
        topic_cfg,
        "representation_variants",
        ["ctfidf_only", "keybert_inspired", "mmr_diverse"],
    ):
        analysis_runs.append(
            run_lexical_analysis(
                base_run=primary_run,
                prepared_df=prepared_df,
                source_rows=len(source_df),
                topic_cfg=topic_cfg,
                analysis_group="representation",
                analysis_variant=representation_variant,
                vectorizer_variant="tuned_bigram",
                ctfidf_variant="reduce_frequent",
                representation_variant=representation_variant,
                notes="MiniLM topic assignments fixed; only representation labeling changes.",
            )
        )

    modular_summary_df = pd.DataFrame(
        [run["summary_row"] for run in analysis_runs]
    ).sort_values(["analysis_group", "analysis_variant"])
    modular_topics_df = pd.concat(
        [run["topic_rows_df"] for run in analysis_runs],
        ignore_index=True,
    )
    modular_stage_timings_df = pd.DataFrame(
        [row for run in analysis_runs for row in run["stage_timing_rows"]]
    )

    modular_summary_json = {
        "generated_at": utc_now_iso(),
        "primary_embedding_model": primary_embedding_model,
        "kmeans_clusters": kmeans_clusters,
        "rows": modular_summary_df.to_dict(orient="records"),
    }

    modular_summary_csv = (
        zika_root / "data" / "final" / "zika_bertopic_modular_analysis.csv"
    )
    modular_summary_json_path = (
        zika_root / "data" / "final" / "zika_bertopic_modular_analysis.json"
    )
    modular_topics_csv = (
        zika_root / "data" / "final" / "zika_bertopic_modular_topics.csv"
    )
    modular_timing_csv = (
        zika_root / "data" / "final" / "zika_bertopic_stage_timings.csv"
    )
    modular_summary_df.to_csv(modular_summary_csv, index=False)
    modular_summary_json_path.write_text(
        json.dumps(modular_summary_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    modular_topics_df.to_csv(modular_topics_csv, index=False)
    modular_stage_timings_df.to_csv(modular_timing_csv, index=False)

    logger.info(
        "BERTopic modular analysis complete across groups: %s",
        ", ".join(sorted(modular_summary_df["analysis_group"].unique().tolist())),
    )


if __name__ == "__main__":
    main()
