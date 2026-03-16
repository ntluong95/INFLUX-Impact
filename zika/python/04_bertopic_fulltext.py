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
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from dotenv import load_dotenv
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
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


def build_shared_vocabulary(
    prepared_df: pd.DataFrame, topic_cfg: dict[str, Any]
) -> dict[str, int]:
    vectorizer = CountVectorizer(
        stop_words=str(topic_cfg.get("vectorizer_stop_words", "english")),
        min_df=max(1, int(topic_cfg.get("min_df", 2))),
        ngram_range=(
            int(topic_cfg.get("ngram_min", 1)),
            int(topic_cfg.get("ngram_max", 2)),
        ),
    )
    vectorizer.fit(prepared_df["model_text"].tolist())
    return dict(vectorizer.vocabulary_)


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


def build_summary_payload(
    prepared_df: pd.DataFrame,
    source_rows: int,
    topic_info_df: pd.DataFrame,
    topic_cfg: dict[str, Any],
    runtime_seconds: float,
    embedding_model_name: str,
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

    return {
        "generated_at": utc_now_iso(),
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
        "config": {
            "min_word_count": int(topic_cfg.get("min_word_count", 120)),
            "max_words_per_document": int(topic_cfg.get("max_words_per_document", 800)),
            "embedding_model": embedding_model_name,
            "min_topic_size": int(topic_cfg.get("min_topic_size", 12)),
            "ngram_range": [
                int(topic_cfg.get("ngram_min", 1)),
                int(topic_cfg.get("ngram_max", 2)),
            ],
            "umap_neighbors": int(topic_cfg.get("umap_neighbors", 15)),
            "umap_components": int(topic_cfg.get("umap_components", 5)),
            "reduce_frequent_words": bool(
                topic_cfg.get("reduce_frequent_words", True)
            ),
            "use_keybert_inspired": bool(
                topic_cfg.get("use_keybert_inspired", True)
            ),
            "seed": int(topic_cfg.get("seed", 42)),
        },
        "top_topics": top_topics,
    }


def build_topic_model(
    embedding_model_name: str,
    vocabulary: dict[str, int],
    topic_cfg: dict[str, Any],
    seed: int,
) -> BERTopic:
    embedding_model = SentenceTransformer(embedding_model_name)
    vectorizer_model = CountVectorizer(
        stop_words=str(topic_cfg.get("vectorizer_stop_words", "english")),
        min_df=max(1, int(topic_cfg.get("min_df", 2))),
        ngram_range=(
            int(topic_cfg.get("ngram_min", 1)),
            int(topic_cfg.get("ngram_max", 2)),
        ),
        vocabulary=vocabulary,
    )
    ctfidf_model = ClassTfidfTransformer(
        reduce_frequent_words=bool(topic_cfg.get("reduce_frequent_words", True))
    )
    representation_model = (
        KeyBERTInspired()
        if bool(topic_cfg.get("use_keybert_inspired", True))
        else None
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

    return BERTopic(
        embedding_model=embedding_model,
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


def fit_model_run(
    prepared_df: pd.DataFrame,
    source_rows: int,
    topic_cfg: dict[str, Any],
    embedding_model_name: str,
    vocabulary: dict[str, int],
    seed: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    logger.info(
        "BERTopic fitting started for embedding model '%s' on %s documents",
        embedding_model_name,
        len(prepared_df),
    )
    topic_model = build_topic_model(
        embedding_model_name=embedding_model_name,
        vocabulary=vocabulary,
        topic_cfg=topic_cfg,
        seed=seed,
    )
    start_time = time.perf_counter()
    topics, _probabilities = topic_model.fit_transform(prepared_df["model_text"].tolist())
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
    document_topics_df["embedding_model"] = embedding_model_name
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

    summary_payload = build_summary_payload(
        prepared_df=prepared_df,
        source_rows=source_rows,
        topic_info_df=topic_info_df,
        topic_cfg=topic_cfg,
        runtime_seconds=runtime_seconds,
        embedding_model_name=embedding_model_name,
    )
    logger.info(
        "BERTopic fitting complete for '%s': modeled=%s topics=%s outliers=%s runtime=%s",
        embedding_model_name,
        len(prepared_df),
        summary_payload["unique_topics_excluding_outlier"],
        summary_payload["outlier_documents"],
        summary_payload["runtime_human"],
    )

    return {
        "embedding_model": embedding_model_name,
        "slug": slugify_model_name(embedding_model_name),
        "topic_model": topic_model,
        "document_topics_df": document_topics_df,
        "topic_info_df": topic_info_df,
        "topic_info_out": serialize_object_columns(topic_info_df),
        "summary_payload": summary_payload,
    }


def write_model_outputs(
    zika_root: Path, model_run: dict[str, Any], is_primary: bool
) -> None:
    slug = str(model_run["slug"])
    base_prefix = zika_root / "data" / "final" / f"zika_bertopic_{slug}"

    write_dataframe(
        model_run["document_topics_df"],
        base_prefix.with_name(f"zika_bertopic_{slug}_document_topics.csv"),
        base_prefix.with_name(f"zika_bertopic_{slug}_document_topics.parquet"),
    )
    model_run["topic_info_out"].to_csv(
        base_prefix.with_name(f"zika_bertopic_{slug}_topic_info.csv"),
        index=False,
    )
    base_prefix.with_name(f"zika_bertopic_{slug}_summary.json").write_text(
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


def topic_matrix_with_ids(model_run: dict[str, Any]) -> tuple[list[int], Any]:
    topic_info_df = model_run["topic_info_df"]
    topic_ids = (
        topic_info_df.loc[topic_info_df["Topic"] != -1, "Topic"]
        .astype(int)
        .sort_values()
        .tolist()
    )
    matrix = model_run["topic_model"].c_tf_idf_
    return topic_ids, matrix


def build_topic_match_rows(
    source_run: dict[str, Any], target_run: dict[str, Any]
) -> list[dict[str, Any]]:
    source_ids, source_matrix = topic_matrix_with_ids(source_run)
    target_ids, target_matrix = topic_matrix_with_ids(target_run)
    if not source_ids or not target_ids:
        return []

    similarity = cosine_similarity(source_matrix, target_matrix)
    source_info = source_run["topic_info_df"].set_index("Topic")
    target_info = target_run["topic_info_df"].set_index("Topic")

    rows: list[dict[str, Any]] = []
    for source_idx, source_topic_id in enumerate(source_ids):
        best_idx = int(np.argmax(similarity[source_idx]))
        target_topic_id = int(target_ids[best_idx])
        rows.append(
            {
                "source_model": source_run["embedding_model"],
                "target_model": target_run["embedding_model"],
                "source_topic_id": source_topic_id,
                "source_topic_count": int(source_info.loc[source_topic_id, "Count"]),
                "source_topic_name": str(source_info.loc[source_topic_id, "Name"]),
                "target_topic_id": target_topic_id,
                "target_topic_count": int(target_info.loc[target_topic_id, "Count"]),
                "target_topic_name": str(target_info.loc[target_topic_id, "Name"]),
                "ctfidf_cosine_similarity": round(
                    float(similarity[source_idx, best_idx]), 6
                ),
            }
        )
    return rows


def build_model_comparison(
    model_runs: list[dict[str, Any]],
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
                "runtime_seconds": payload["runtime_seconds"],
                "runtime_human": payload["runtime_human"],
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
            topic_match_rows.extend(build_topic_match_rows(left_run, right_run))
            topic_match_rows.extend(build_topic_match_rows(right_run, left_run))

    comparison_payload = {
        "generated_at": utc_now_iso(),
        "models": [run["embedding_model"] for run in model_runs],
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

    vocabulary = build_shared_vocabulary(prepared_df, topic_cfg)
    logger.info(
        "Shared vocabulary built with %s terms for cross-model BERTopic comparison",
        len(vocabulary),
    )

    model_runs: list[dict[str, Any]] = []
    for embedding_model_name in embedding_models:
        model_runs.append(
            fit_model_run(
                prepared_df=prepared_df,
                source_rows=len(source_df),
                topic_cfg=topic_cfg,
                embedding_model_name=embedding_model_name,
                vocabulary=vocabulary,
                seed=seed,
                logger=logger,
            )
        )

    for run in model_runs:
        write_model_outputs(
            zika_root=zika_root,
            model_run=run,
            is_primary=run["embedding_model"] == primary_embedding_model,
        )

    comparison_df, comparison_payload, topic_match_df = build_model_comparison(
        model_runs
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

    logger.info(
        "BERTopic comparison complete across %s models: %s",
        len(model_runs),
        ", ".join(run["embedding_model"] for run in model_runs),
    )


if __name__ == "__main__":
    main()
