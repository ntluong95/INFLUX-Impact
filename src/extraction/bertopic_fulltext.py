from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.common import normalize_whitespace, setup_logger, utc_now_iso, write_dataframe_atomic
from src.utils.config import load_project_config, project_paths
from src.utils.project import DatasetKey, resolve_datasets, resolve_languages, resolve_pathogen_domains


DOCUMENT_TOPIC_COLUMNS = [
    "record_id",
    "dataset_key",
    "pathogen_domain",
    "language_code",
    "domain",
    "final_url",
    "rss_title",
    "word_count",
    "article_language",
    "model_word_count",
    "embedding_model",
    "topic_id",
    "topic_name",
    "topic_keywords",
    "is_outlier",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BERTopic over pathogen full-text articles."
    )
    parser.add_argument("--config", default="src/config/pipeline.yaml")
    parser.add_argument("--pathogen-domains", default="all")
    parser.add_argument("--languages", default="all")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


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


def topic_keywords_for(topic_model: BERTopic, topic_id: int) -> str:
    if topic_id == -1:
        return "outlier"
    topic_terms = topic_model.get_topic(topic_id) or []
    return ", ".join(term for term, _score in topic_terms[:6])


def prepare_documents(df: pd.DataFrame, topic_cfg: dict[str, Any]) -> pd.DataFrame:
    prepared = df.copy()
    prepared["full_text"] = prepared.get("full_text", "").fillna("").astype(str)
    prepared["rss_title"] = prepared.get("rss_title", "").fillna("").astype(str)
    prepared["word_count"] = pd.to_numeric(prepared.get("word_count", 0), errors="coerce").fillna(0)
    prepared["article_language"] = prepared.get("article_language", "").fillna("").astype(str)

    min_word_count = int(topic_cfg.get("min_word_count", 120))
    max_words_per_document = int(topic_cfg.get("max_words_per_document", 900))
    prepared = prepared.loc[
        prepared["full_text"].str.strip().ne("")
        & (prepared["word_count"] >= min_word_count)
    ].copy()

    def model_text(row: pd.Series) -> str:
        full_text = normalize_whitespace(row["full_text"])
        words = full_text.split()
        clipped = " ".join(words[:max_words_per_document])
        return "\n\n".join(part for part in [normalize_whitespace(row["rss_title"]), clipped] if part)

    prepared["model_text"] = prepared.apply(model_text, axis=1)
    prepared["model_word_count"] = prepared["model_text"].map(lambda value: len(str(value).split()))
    prepared = prepared.loc[prepared["model_word_count"] > 0].copy()
    prepared = prepared.sort_values("record_id").reset_index(drop=True)
    return prepared


def load_fulltext_table(dataset: DatasetKey, paths: Any) -> pd.DataFrame:
    parquet_path = paths.fulltext_parquet(dataset)
    csv_path = paths.fulltext_csv(dataset)
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path, low_memory=False)
    raise FileNotFoundError(f"Missing full-text input for {dataset.stem}: {csv_path}")


def topic_summary(
    dataset: DatasetKey,
    prepared_df: pd.DataFrame,
    topic_info_df: pd.DataFrame,
    runtime_seconds: float,
    embedding_model: str,
    stage_timings: dict[str, float],
    notes: str = "",
) -> dict[str, Any]:
    non_outlier = topic_info_df.loc[topic_info_df["Topic"] != -1].copy()
    outlier_documents = (
        int(topic_info_df.loc[topic_info_df["Topic"] == -1, "Count"].iloc[0])
        if any(topic_info_df["Topic"] == -1)
        else 0
    )
    if non_outlier.empty:
        largest_topic_share = 0.0
    else:
        largest_topic_share = float(non_outlier["Count"].max()) / float(non_outlier["Count"].sum())
    return {
        "generated_at": utc_now_iso(),
        "dataset_key": dataset.stem,
        "pathogen_domain": dataset.pathogen_domain,
        "language_code": dataset.language_code,
        "embedding_model": embedding_model,
        "documents_modeled": int(len(prepared_df)),
        "unique_topics_excluding_outlier": int(len(non_outlier)),
        "outlier_documents": outlier_documents,
        "largest_topic_share": round(largest_topic_share, 6),
        "runtime_seconds": round(runtime_seconds, 3),
        "runtime_human": format_elapsed(runtime_seconds),
        "stage_timings": {key: round(float(value), 3) for key, value in stage_timings.items()},
        "notes": notes,
    }


def document_topics_frame(
    dataset: DatasetKey,
    prepared_df: pd.DataFrame,
    topic_model: BERTopic,
    topics: list[int],
    embedding_model: str,
) -> pd.DataFrame:
    topic_info_df = topic_model.get_topic_info()
    topic_name_lookup = {
        int(record["Topic"]): str(record.get("Name", ""))
        for record in topic_info_df.to_dict(orient="records")
    }
    document_topics_df = prepared_df[
        [
            "record_id",
            "domain",
            "final_url",
            "rss_title",
            "word_count",
            "article_language",
            "model_word_count",
        ]
    ].copy()
    document_topics_df.insert(1, "dataset_key", dataset.stem)
    document_topics_df.insert(2, "pathogen_domain", dataset.pathogen_domain)
    document_topics_df.insert(3, "language_code", dataset.language_code)
    document_topics_df["embedding_model"] = embedding_model
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


def fit_dataset(
    dataset: DatasetKey,
    config: dict[str, Any],
    force: bool,
    logger: Any,
) -> None:
    paths = project_paths(config)
    topic_cfg = config["bertopic"]
    output_dir = paths.bertopic_dir(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)

    document_topics_csv = output_dir / f"{dataset.stem}_document_topics.csv"
    topic_info_csv = output_dir / f"{dataset.stem}_topic_info.csv"
    summary_json = output_dir / f"{dataset.stem}_summary.json"
    projection_csv = output_dir / f"{dataset.stem}_projection.csv"
    timings_csv = output_dir / f"{dataset.stem}_stage_timings.csv"

    if summary_json.exists() and not force:
        logger.info("BERTopic outputs already exist for %s: %s", dataset.stem, summary_json)
        return

    fulltext_df = load_fulltext_table(dataset, paths)
    prepared_df = prepare_documents(fulltext_df, topic_cfg)
    min_documents = int(topic_cfg.get("min_documents", 20))
    if len(prepared_df) < min_documents:
        empty_topics = pd.DataFrame(columns=DOCUMENT_TOPIC_COLUMNS)
        empty_topic_info = pd.DataFrame(columns=["Topic", "Count", "Name", "Representation"])
        empty_projection = pd.DataFrame(columns=["record_id", "topic_id", "x", "y"])
        write_dataframe_atomic(empty_topics, document_topics_csv)
        write_dataframe_atomic(empty_topic_info, topic_info_csv)
        write_dataframe_atomic(empty_projection, projection_csv)
        write_dataframe_atomic(
            pd.DataFrame(
                [{"stage_name": "skipped", "runtime_seconds": 0.0, "runtime_human": "0s"}]
            ),
            timings_csv,
        )
        summary_json.write_text(
            json.dumps(
                topic_summary(
                    dataset,
                    prepared_df,
                    empty_topic_info,
                    runtime_seconds=0.0,
                    embedding_model="",
                    stage_timings={"skipped": 0.0},
                    notes=f"Skipped because only {len(prepared_df)} documents met the modeling threshold.",
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Skipping BERTopic for %s because documents_modeled=%s < min_documents=%s", dataset.stem, len(prepared_df), min_documents)
        return

    embedding_models = topic_cfg.get("embedding_models_by_domain", {})
    embedding_model_name = str(
        embedding_models.get(dataset.pathogen_domain, "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    )
    documents = prepared_df["model_text"].tolist()

    encode_start = time.perf_counter()
    sentence_model = SentenceTransformer(embedding_model_name)
    embeddings = sentence_model.encode(documents, show_progress_bar=False, convert_to_numpy=True)
    embedding_seconds = time.perf_counter() - encode_start

    language_stop_words = topic_cfg.get("language_stop_words", {})
    stop_words = language_stop_words.get(dataset.language_code)
    vectorizer = CountVectorizer(
        stop_words=stop_words if stop_words else None,
        min_df=max(1, int(topic_cfg.get("min_df", 3))),
        ngram_range=(int(topic_cfg.get("ngram_min", 1)), int(topic_cfg.get("ngram_max", 2))),
    )
    ctfidf_model = ClassTfidfTransformer(
        bm25_weighting=bool(topic_cfg.get("bm25_weighting", True)),
        reduce_frequent_words=bool(topic_cfg.get("reduce_frequent_words", True)),
    )
    representation_model = MaximalMarginalRelevance(
        diversity=float(topic_cfg.get("representation_mmr_diversity", 0.5)),
        top_n_words=int(topic_cfg.get("top_n_words", 10)),
    )
    umap_model = UMAP(
        n_neighbors=int(topic_cfg.get("umap_neighbors", 12)),
        n_components=int(topic_cfg.get("umap_components", 5)),
        min_dist=float(topic_cfg.get("umap_min_dist", 0.05)),
        metric="cosine",
        random_state=int(topic_cfg.get("seed", 42)),
    )

    cluster_strategy = str(topic_cfg.get("cluster_model", "kmeans")).strip().lower()
    if cluster_strategy == "hdbscan":
        cluster_model = HDBSCAN(
            min_cluster_size=max(2, int(topic_cfg.get("min_topic_size", 12))),
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
    else:
        requested_clusters = int(topic_cfg.get("n_clusters", 10))
        cluster_model = KMeans(
            n_clusters=max(2, min(requested_clusters, max(2, len(prepared_df) // 3))),
            random_state=int(topic_cfg.get("seed", 42)),
            n_init=10,
        )

    topic_model = BERTopic(
        embedding_model=sentence_model,
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        top_n_words=int(topic_cfg.get("top_n_words", 10)),
        min_topic_size=max(2, int(topic_cfg.get("min_topic_size", 12))),
        calculate_probabilities=False,
        verbose=False,
    )

    fit_start = time.perf_counter()
    topics, _probabilities = topic_model.fit_transform(documents, embeddings)
    fit_seconds = time.perf_counter() - fit_start

    topic_info_df = topic_model.get_topic_info()
    projection_start = time.perf_counter()
    projection = UMAP(
        n_neighbors=int(topic_cfg.get("projection_neighbors", 10)),
        n_components=2,
        min_dist=float(topic_cfg.get("projection_min_dist", 0.05)),
        metric="cosine",
        random_state=int(topic_cfg.get("seed", 42)),
    ).fit_transform(embeddings)
    projection_seconds = time.perf_counter() - projection_start

    document_topics_df = document_topics_frame(
        dataset,
        prepared_df,
        topic_model,
        topics,
        embedding_model_name,
    )
    topic_info_out = serialize_object_columns(topic_info_df)
    projection_df = prepared_df[["record_id"]].copy()
    projection_df["topic_id"] = [int(topic_id) for topic_id in topics]
    projection_df["x"] = projection[:, 0]
    projection_df["y"] = projection[:, 1]

    total_runtime_seconds = embedding_seconds + fit_seconds + projection_seconds
    stage_timings = {
        "embedding_encode": embedding_seconds,
        "bertopic_fit_transform": fit_seconds,
        "projection_umap_2d": projection_seconds,
    }
    summary_payload = topic_summary(
        dataset,
        prepared_df,
        topic_info_df,
        total_runtime_seconds,
        embedding_model_name,
        stage_timings,
        notes=(
            "Impact-oriented topic exploration on full-text articles. "
            "Embedding models are configurable per pathogen domain."
        ),
    )

    write_dataframe_atomic(document_topics_df, document_topics_csv)
    write_dataframe_atomic(topic_info_out, topic_info_csv)
    write_dataframe_atomic(projection_df, projection_csv)
    write_dataframe_atomic(
        pd.DataFrame(
            [
                {
                    "stage_name": key,
                    "runtime_seconds": round(float(value), 3),
                    "runtime_human": format_elapsed(value),
                }
                for key, value in stage_timings.items()
            ]
        ),
        timings_csv,
    )
    summary_json.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "BERTopic complete for %s: documents_modeled=%s topics=%s runtime=%s",
        dataset.stem,
        len(prepared_df),
        len(topic_info_df.loc[topic_info_df["Topic"] != -1]),
        format_elapsed(total_runtime_seconds),
    )


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    paths = project_paths(config)
    logger = setup_logger(
        "pathogen_bertopic",
        paths.logs_dir() / "bertopic.log",
    )
    datasets = resolve_datasets(
        resolve_pathogen_domains(args.pathogen_domains),
        resolve_languages(args.languages),
    )
    for dataset in datasets:
        fit_dataset(dataset, config, args.force, logger)


if __name__ == "__main__":
    main()
