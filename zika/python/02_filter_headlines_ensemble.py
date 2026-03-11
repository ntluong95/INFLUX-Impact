from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

import krippendorff

warnings.filterwarnings("ignore", message=".*doesn't match a supported version.*")

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from filter_utils import (
    SbertScorer,
    build_prompt,
    call_local_chat,
    call_openai_chat,
    classification_metrics,
    derive_ensemble_action,
    deterministic_validation_sample,
    load_config,
    normalized_base_url,
    ollama_available_models,
    parse_llm_label,
    setup_logger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter headlines via OpenAI + local LLM ensemble."
    )
    parser.add_argument("--config", default="zika/config/zika.yaml")
    return parser.parse_args()


def write_outputs(
    df: pd.DataFrame, csv_path: Path, parquet_path: Path | None = None
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    if parquet_path:
        df.to_parquet(parquet_path, index=False)


def metric_row(
    group: str, metric: str, value: float | int | str, subgroup: str = ""
) -> dict[str, Any]:
    return {
        "metric_group": group,
        "metric_name": metric,
        "subgroup": subgroup,
        "value": value,
    }


VALID_LABELS = {"relevant", "irrelevant", "unsure"}
VALID_ACTIONS = {"keep", "drop", "review"}


def nonempty_mask(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip() != ""


def valid_label_mask(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).isin(sorted(VALID_LABELS))


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def json_safe(value: Any) -> Any:
    return None if pd.isna(value) else value


def completed_score_mask(df: pd.DataFrame) -> pd.Series:
    return (
        df["final_action"].fillna("").astype(str).isin(sorted(VALID_ACTIONS))
        & valid_label_mask(df["openai_label"])
        & valid_label_mask(df["local_label"])
        & ~nonempty_mask(df["openai_error"])
        & ~nonempty_mask(df["local_error"])
    )


def sanitize_scored_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    df = df.copy()
    changed_rows = pd.Series(False, index=df.index)

    for prefix in ["openai", "local"]:
        err_mask = nonempty_mask(df[f"{prefix}_error"])
        if err_mask.any():
            changed_rows = changed_rows | err_mask
            df.loc[err_mask, f"{prefix}_label"] = ""
            df.loc[err_mask, f"{prefix}_confidence"] = np.nan
            df.loc[err_mask, f"{prefix}_rationale"] = ""

    invalid_ensemble_mask = (
        nonempty_mask(df["openai_error"])
        | nonempty_mask(df["local_error"])
        | ~valid_label_mask(df["openai_label"])
        | ~valid_label_mask(df["local_label"])
    )
    if invalid_ensemble_mask.any():
        changed_rows = changed_rows | invalid_ensemble_mask
        df.loc[invalid_ensemble_mask, "rationale_similarity"] = np.nan
        df.loc[invalid_ensemble_mask, "rationale_similarity_method"] = ""
        df.loc[invalid_ensemble_mask, "final_action"] = ""
        df.loc[invalid_ensemble_mask, "ensemble_label"] = ""

    return df, int(changed_rows.sum())


def refresh_validation_assets(
    base_df: pd.DataFrame,
    cfg: dict[str, Any],
    validation_csv: Path,
    metrics_json: Path,
    metrics_csv: Path,
    logger,
) -> None:
    val_size = int(cfg.get("validation", {}).get("sample_size", 120))
    val_seed = int(cfg.get("validation", {}).get("seed", 42))

    scored_df = base_df[completed_score_mask(base_df)].copy()

    if validation_csv.exists():
        existing_val = pd.read_csv(validation_csv, dtype=str)
    else:
        existing_val = pd.DataFrame(
            columns=["record_id", "gold_label", "review_notes", "validated_at"]
        )

    for col in ["gold_label", "review_notes", "validated_at"]:
        if col not in existing_val.columns:
            existing_val[col] = ""

    val_cols = [
        "record_id",
        "rss_title",
        "source_hint",
        "rss_pubdate",
        "google_news_redirect_url",
        "openai_label",
        "openai_confidence",
        "openai_rationale",
        "local_label",
        "local_confidence",
        "local_rationale",
        "rationale_similarity",
        "final_action",
        "ensemble_label",
        "gold_label",
        "review_notes",
        "validated_at",
    ]

    val_sample = deterministic_validation_sample(scored_df, val_size, val_seed)
    if not val_sample.empty and not existing_val.empty:
        val_sample = val_sample.merge(
            existing_val[["record_id", "gold_label", "review_notes", "validated_at"]],
            on="record_id",
            how="left",
        )
    elif not val_sample.empty:
        val_sample["gold_label"] = ""
        val_sample["review_notes"] = ""
        val_sample["validated_at"] = ""

    validation_csv.parent.mkdir(parents=True, exist_ok=True)
    existing_has_labels = (
        not existing_val.empty
        and existing_val["gold_label"].fillna("").astype(str).str.strip().ne("").any()
    )
    if not val_sample.empty:
        val_sample = val_sample[[c for c in val_cols if c in val_sample.columns]]
        val_sample.to_csv(validation_csv, index=False)
    elif not existing_has_labels:
        pd.DataFrame(columns=val_cols).to_csv(validation_csv, index=False)

    labeled_val = (
        val_sample[val_sample["gold_label"].isin(["relevant", "irrelevant", "unsure"])]
        if not val_sample.empty
        else pd.DataFrame()
    )

    m_rows = []
    m_rows.append(metric_row("counts", "total_rows", len(base_df)))
    m_rows.append(metric_row("counts", "completed_rows", len(scored_df)))
    m_rows.append(metric_row("counts", "incomplete_rows", len(base_df) - len(scored_df)))
    m_rows.append(metric_row("counts", "openai_error_rows", int(nonempty_mask(base_df["openai_error"]).sum())))
    m_rows.append(metric_row("counts", "local_error_rows", int(nonempty_mask(base_df["local_error"]).sum())))

    if len(scored_df) > 0:
        o_lbl = scored_df["openai_label"].values
        l_lbl = scored_df["local_label"].values
        agree_mask = pd.Series(o_lbl) == pd.Series(l_lbl)
        raw_agree = agree_mask.sum() / len(scored_df)
        disagree_rate = 1.0 - raw_agree
        unsure_mask = (pd.Series(o_lbl) == "unsure") | (pd.Series(l_lbl) == "unsure")
        share_any_unsure = unsure_mask.sum() / len(scored_df)
    else:
        raw_agree = np.nan
        disagree_rate = np.nan
        share_any_unsure = np.nan

    m_rows.append(metric_row("agreement", "raw_agreement_rate", raw_agree))
    m_rows.append(metric_row("agreement", "disagreement_rate", disagree_rate))
    m_rows.append(metric_row("agreement", "share_any_unsure", share_any_unsure))

    kappa = np.nan
    krip_alpha = np.nan
    if len(scored_df) > 0:
        try:
            kappa = cohen_kappa_score(scored_df["openai_label"], scored_df["local_label"])
            m_rows.append(metric_row("agreement", "cohen_kappa", kappa))
        except Exception:
            pass

        if krippendorff is not None:
            try:
                mapping = {"relevant": 2, "irrelevant": 0, "unsure": 1}
                arr = np.array(
                    [
                        scored_df["openai_label"].map(mapping).values,
                        scored_df["local_label"].map(mapping).values,
                    ]
                )
                krip_alpha = krippendorff.alpha(
                    reliability_data=arr, level_of_measurement="ordinal"
                )
                m_rows.append(metric_row("agreement", "krippendorff_alpha", krip_alpha))
            except Exception:
                pass

    for val, count in scored_df["openai_label"].value_counts().items():
        m_rows.append(
            metric_row(
                "openai_distribution", "label_share", count / len(scored_df), str(val)
            )
        )
    for val, count in scored_df["local_label"].value_counts().items():
        m_rows.append(
            metric_row(
                "local_distribution", "label_share", count / len(scored_df), str(val)
            )
        )
    for val, count in scored_df["final_action"].value_counts().items():
        m_rows.append(
            metric_row(
                "ensemble_distribution", "action_share", count / len(scored_df), str(val)
            )
        )

    action_counts = scored_df["final_action"].value_counts()
    review_rate = (
        action_counts.get("review", 0) / len(scored_df) if len(scored_df) > 0 else np.nan
    )
    retention_rate = (
        action_counts.get("keep", 0) / len(scored_df) if len(scored_df) > 0 else np.nan
    )
    m_rows.append(metric_row("metrics", "review_rate", review_rate))
    m_rows.append(metric_row("metrics", "retention_rate", retention_rate))

    if not labeled_val.empty:
        m_rows.append(metric_row("validation", "labeled_rows", len(labeled_val)))
        y_true = labeled_val["gold_label"].values
        for sys_name, y_pred in [
            ("openai", labeled_val["openai_label"].values),
            ("local", labeled_val["local_label"].values),
            ("ensemble", labeled_val["ensemble_label"].values),
        ]:
            metrics = classification_metrics(y_true, y_pred, ["relevant", "irrelevant", "unsure"])
            for m_name, m_val in metrics.items():
                if m_name != "n":
                    m_rows.append(metric_row(f"validation_{sys_name}", m_name, m_val))

            try:
                cm = confusion_matrix(
                    y_true, y_pred, labels=["relevant", "irrelevant", "unsure"]
                )
                m_rows.append(
                    metric_row(
                        f"validation_{sys_name}",
                        "confusion_matrix",
                        json.dumps(cm.tolist()),
                    )
                )
            except Exception:
                pass
    else:
        m_rows.append(metric_row("validation", "labeled_rows", 0))

    metrics_df = pd.DataFrame(m_rows)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_csv, index=False)

    metrics_payload = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "total_rows": len(base_df),
        "completed_rows": len(scored_df),
        "incomplete_rows": len(base_df) - len(scored_df),
        "errors": {
            "openai_error_rows": int(nonempty_mask(base_df["openai_error"]).sum()),
            "local_error_rows": int(nonempty_mask(base_df["local_error"]).sum()),
        },
        "agreement": {
            "raw_agreement_rate": json_safe(raw_agree),
            "disagreement_rate": json_safe(disagree_rate),
            "share_any_unsure": json_safe(share_any_unsure),
            "cohen_kappa": json_safe(kappa),
            "krippendorff_alpha": json_safe(krip_alpha),
        },
        "distributions": {
            "openai": scored_df["openai_label"].value_counts().to_dict(),
            "local": scored_df["local_label"].value_counts().to_dict(),
            "ensemble_action": scored_df["final_action"].value_counts().to_dict(),
        },
        "validation": {
            "sample_path": str(validation_csv),
            "labeled_rows": len(labeled_val),
            "review_rate": json_safe(review_rate),
            "retention_rate": json_safe(retention_rate),
        },
    }

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    logger.info(f"Validation sample written to {validation_csv}")
    logger.info(f"Metrics written to {metrics_json} and {metrics_csv}")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config_path = (repo_root / args.config).resolve()
    zika_root = config_path.parent.parent
    cfg = load_config(config_path)

    log_path = zika_root / "logs" / "02_filter_headlines_ensemble.log"
    logger = setup_logger(log_path)

    input_csv = zika_root / "data" / "intermediate" / "zika_rss_raw.csv"
    output_csv = zika_root / "data" / "intermediate" / "zika_headlines_scored.csv"
    output_parquet = (
        zika_root / "data" / "intermediate" / "zika_headlines_scored.parquet"
    )
    validation_csv = (
        zika_root / "data" / "validation" / "zika_headline_validation_sample.csv"
    )
    metrics_json = zika_root / "data" / "validation" / "zika_headline_metrics.json"
    metrics_csv = zika_root / "data" / "validation" / "zika_headline_metrics.csv"

    if not input_csv.exists():
        logger.error(f"Missing input CSV: {input_csv}. Run stage 1 first.")
        sys.exit(1)

    hf_cfg = cfg.get("headline_filter", {})
    openai_cfg = hf_cfg.get("openai", {})
    local_cfg = hf_cfg.get("local", {})

    base_df = pd.read_csv(input_csv, dtype=str)

    new_cols = {
        "openai_model": openai_cfg.get("model", ""),
        "openai_raw_response": "",
        "openai_label": "",
        "openai_confidence": np.nan,
        "openai_rationale": "",
        "openai_error": "",
        "local_model": local_cfg.get("model", ""),
        "local_raw_response": "",
        "local_label": "",
        "local_confidence": np.nan,
        "local_rationale": "",
        "local_error": "",
        "rationale_similarity": np.nan,
        "rationale_similarity_method": "",
        "final_action": "",
        "ensemble_label": "",
        "processed_at": "",
    }

    for col, default_val in new_cols.items():
        if col not in base_df.columns:
            base_df[col] = default_val

    if output_csv.exists():
        existing_df = pd.read_csv(output_csv, dtype=str)
        # Restore specific typed columns
        for col in ["openai_confidence", "local_confidence", "rationale_similarity"]:
            if col in existing_df.columns:
                existing_df[col] = pd.to_numeric(existing_df[col], errors="coerce")

        # Merge overlapping records safely
        base_df.set_index("record_id", inplace=True)
        existing_df.set_index("record_id", inplace=True)
        base_df.update(existing_df)
        base_df.reset_index(inplace=True)

        for col in ["openai_confidence", "local_confidence", "rationale_similarity"]:
            base_df[col] = pd.to_numeric(base_df[col], errors="coerce")

    checkpoint_every = int(hf_cfg.get("checkpoint_every", 10))
    conf_default = float(hf_cfg.get("confidence_default", 0.5))
    sim_threshold = float(hf_cfg.get("similarity_threshold", 0.75))

    system_prompt = (
        "You are a strict headline classifier for Zika relevance. "
        "Return JSON only with keys 'label', 'confidence', and 'rationale'. "
        "Use label values only: 'relevant', 'irrelevant', 'unsure'. "
        "Confidence must be between 0 and 1. Rationale must be under 20 words."
    )

    base_df, sanitized_rows = sanitize_scored_frame(base_df)
    if sanitized_rows:
        logger.warning(
            "Sanitized %s rows with provider errors or incomplete ensemble outputs before scoring.",
            sanitized_rows,
        )
        write_outputs(base_df, output_csv, output_parquet)
        refresh_validation_assets(
            base_df, cfg, validation_csv, metrics_json, metrics_csv, logger
        )

    def checkpoint_and_exit(message: str) -> None:
        write_outputs(base_df, output_csv, output_parquet)
        refresh_validation_assets(
            base_df, cfg, validation_csv, metrics_json, metrics_csv, logger
        )
        logger.error(message)
        sys.exit(1)

    incomplete_indices = base_df[~completed_score_mask(base_df)].index.tolist()

    logger.info(
        f"Stage 2 starting with {len(base_df)} headlines; {len(incomplete_indices)} rows need scoring"
    )

    if incomplete_indices and not openai_cfg.get("api_key"):
        checkpoint_and_exit(
            "Missing OPENAI_API_KEY. Stage 2 requires OpenAI + local LLM scoring for incomplete rows."
        )

    if incomplete_indices and (
        not local_cfg.get("base_url") or not local_cfg.get("model")
    ):
        checkpoint_and_exit(
            "Missing LOCAL_LLM_BASE_URL or LOCAL_LLM_MODEL for incomplete rows."
        )

    if incomplete_indices and str(local_cfg.get("provider", "ollama")).lower() == "ollama":
        try:
            available_models = ollama_available_models(
                normalized_base_url(local_cfg.get("base_url")),
                int(local_cfg.get("timeout_seconds", 30)),
            )
        except Exception as exc:
            checkpoint_and_exit(f"Unable to inspect local Ollama models: {exc}")

        configured_model = str(local_cfg.get("model", "")).strip()
        if configured_model not in available_models:
            model_preview = ", ".join(sorted(available_models)[:10]) or "[none detected]"
            checkpoint_and_exit(
                "Configured LOCAL_LLM_MODEL '%s' is not available in Ollama. Available models: %s"
                % (configured_model, model_preview)
            )

        try:
            _, local_probe = call_local_chat(
                system_prompt,
                build_prompt("Zika outbreak spreads in Brazil", "Reuters", "2015-01-01"),
                local_cfg,
            )
            local_probe_parsed = parse_llm_label(local_probe, conf_default)
            if local_probe_parsed["parse_error"]:
                raise ValueError(local_probe_parsed["parse_error"])
        except Exception as exc:
            checkpoint_and_exit(
                f"Local LLM preflight failed for model '{configured_model}': {exc}"
            )

    sbert_model_name = hf_cfg.get("sbert_model", "all-MiniLM-L6-v2")
    if incomplete_indices:
        try:
            sbert = SbertScorer(model_name=sbert_model_name)
        except ImportError:
            checkpoint_and_exit(
                "sentence-transformers not installed. Install via pip."
            )
    else:
        sbert = None

    for i, idx in enumerate(incomplete_indices):
        row = base_df.loc[idx]
        user_prompt = build_prompt(
            row["rss_title"], row["source_hint"], row["rss_pubdate"]
        )

        openai_complete = (
            str(row.get("openai_label", "")).strip() in VALID_LABELS
            and str(row.get("openai_error", "")).strip() == ""
        )
        if openai_complete:
            o_raw = "" if pd.isna(row["openai_raw_response"]) else str(row["openai_raw_response"])
            o_parsed = {
                "label": str(row["openai_label"]).strip(),
                "confidence": safe_float(row["openai_confidence"], conf_default),
                "rationale": "" if pd.isna(row["openai_rationale"]) else str(row["openai_rationale"]).strip(),
                "parse_error": "",
            }
        else:
            try:
                o_raw, o_content = call_openai_chat(system_prompt, user_prompt, openai_cfg)
            except Exception as exc:
                base_df.at[idx, "openai_error"] = str(exc)
                base_df, _ = sanitize_scored_frame(base_df)
                checkpoint_and_exit(
                    f"OpenAI scoring failed for record_id={row['record_id']}: {exc}"
                )

            o_parsed = parse_llm_label(o_content, conf_default)
            if o_parsed["parse_error"]:
                base_df.at[idx, "openai_raw_response"] = o_raw or o_content
                base_df.at[idx, "openai_error"] = f"parse_error:{o_parsed['parse_error']}"
                base_df, _ = sanitize_scored_frame(base_df)
                checkpoint_and_exit(
                    f"OpenAI returned non-parseable JSON for record_id={row['record_id']}: {o_parsed['parse_error']}"
                )

        local_complete = (
            str(row.get("local_label", "")).strip() in VALID_LABELS
            and str(row.get("local_error", "")).strip() == ""
        )
        if local_complete:
            l_raw = "" if pd.isna(row["local_raw_response"]) else str(row["local_raw_response"])
            l_parsed = {
                "label": str(row["local_label"]).strip(),
                "confidence": safe_float(row["local_confidence"], conf_default),
                "rationale": "" if pd.isna(row["local_rationale"]) else str(row["local_rationale"]).strip(),
                "parse_error": "",
            }
        else:
            try:
                l_raw, l_content = call_local_chat(system_prompt, user_prompt, local_cfg)
            except Exception as exc:
                base_df.at[idx, "local_error"] = str(exc)
                base_df, _ = sanitize_scored_frame(base_df)
                checkpoint_and_exit(
                    f"Local LLM scoring failed for record_id={row['record_id']}: {exc}"
                )

            l_parsed = parse_llm_label(l_content, conf_default)
            if l_parsed["parse_error"]:
                base_df.at[idx, "local_raw_response"] = l_raw or l_content
                base_df.at[idx, "local_error"] = f"parse_error:{l_parsed['parse_error']}"
                base_df, _ = sanitize_scored_frame(base_df)
                checkpoint_and_exit(
                    f"Local LLM returned non-parseable JSON for record_id={row['record_id']}: {l_parsed['parse_error']}"
                )

        # Similarity
        sim_val = np.nan
        sim_method = "not_applicable"

        if o_parsed["label"] == "relevant" and l_parsed["label"] == "relevant":
            if sbert:
                try:
                    sim_val = sbert.compute_similarity(
                        o_parsed["rationale"], l_parsed["rationale"]
                    )
                    sim_method = f"sbert_{sbert_model_name}_cosine"
                except Exception as e:
                    sim_method = f"sbert_error: {str(e)}"

        final_action = derive_ensemble_action(
            o_parsed["label"], l_parsed["label"], sim_val, sim_threshold
        )
        ens_label = (
            "relevant"
            if final_action == "keep"
            else ("irrelevant" if final_action == "drop" else "unsure")
        )

        # Update df
        base_df.at[idx, "openai_raw_response"] = o_raw
        base_df.at[idx, "openai_label"] = o_parsed["label"]
        base_df.at[idx, "openai_confidence"] = o_parsed["confidence"]
        base_df.at[idx, "openai_rationale"] = o_parsed["rationale"]
        base_df.at[idx, "openai_error"] = ""

        base_df.at[idx, "local_raw_response"] = l_raw
        base_df.at[idx, "local_label"] = l_parsed["label"]
        base_df.at[idx, "local_confidence"] = l_parsed["confidence"]
        base_df.at[idx, "local_rationale"] = l_parsed["rationale"]
        base_df.at[idx, "local_error"] = ""

        base_df.at[idx, "rationale_similarity"] = sim_val
        base_df.at[idx, "rationale_similarity_method"] = sim_method
        base_df.at[idx, "final_action"] = final_action
        base_df.at[idx, "ensemble_label"] = ens_label
        base_df.at[idx, "processed_at"] = pd.Timestamp.now("UTC").isoformat()

        logger.info(
            f"Scored {i + 1}/{len(incomplete_indices)} record_id={row['record_id']} openai={o_parsed['label']} local={l_parsed['label']} action={final_action}"
        )

        if (i + 1) % checkpoint_every == 0 or (i + 1) == len(incomplete_indices):
            write_outputs(base_df, output_csv, output_parquet)
            logger.info(f"Checkpoint written after {i + 1} rows scored")

    if not incomplete_indices:
        logger.info(
            "No incomplete rows detected; refreshing diagnostics and validation assets only."
        )
    write_outputs(base_df, output_csv, output_parquet)
    refresh_validation_assets(base_df, cfg, validation_csv, metrics_json, metrics_csv, logger)

    logger.info(f"Scored output written to {output_csv} and {output_parquet}")


if __name__ == "__main__":
    main()
