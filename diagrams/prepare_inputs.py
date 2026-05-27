#!/usr/bin/env python3
"""Generate JSON input specs for PaperVizAgent methodology diagrams.

Each spec contains fields expected by PaperVizAgent:
  method_content, figure_caption, figure_type, style_notes

Content is derived from the methodology described in Methodology Impact-revised.md.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS_DIR = ROOT / "diagrams" / "inputs"


def main() -> None:
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)

    specs = {
        "pipeline_overview": {
            "method_content": (
                "Corpus: multilingual news articles on emerging pests and pathogens across three domains "
                "(human, animal, plant) and four languages (en, fr, es, pt). Retrieve Google News RSS at scale, "
                "apply deterministic cleaning and de-duplication, and remove blocked or non-news domains. "
                "Then run Step 3 ensemble relevance filtering with K=3 independent headline classifiers and retain "
                "only the relevant subset R, shrinking the corpus from N articles to N_R. Kept articles proceed to "
                "full-text extraction and then BERTopic for impact-oriented topic discovery."
            ),
            "figure_caption": (
                "Overview of the INFLUX Impact pipeline for pathogen-news cascading impact analysis"
            ),
            "figure_type": "horizontal methodology pipeline",
            "style_notes": (
                "Left-to-right flow with colored stage boxes, clear arrows, domain/language intake at the left, "
                "and a funnel annotation showing N to N_R before full-text extraction and BERTopic."
            ),
        },
        "ensemble_filtering": {
            "method_content": (
                "For each article a_i and target EPP h, run K=3 independent relevance classifiers. "
                "Each classifier outputs a binary label y_hat_ik, a probability p_ik in [0,1], and evidence spans e_ik. "
                "Aggregate with weighted probability pooling p_bar_i = sum_k w_k p_ik, where sum_k w_k = 1. "
                "Classify an article as relevant only when p_bar_i >= theta and the agreement rate "
                "Agree_i = (1/K) sum_k 1(y_hat_ik = 1) >= tau. After the decision, consolidate supporting evidence spans. "
                "Tune w_k, theta, and tau on a human-labeled validation set S to maximize precision subject to a minimum recall."
            ),
            "figure_caption": (
                "Ensemble LLM relevance filtering architecture with weighted probability pooling and dual-threshold decision"
            ),
            "figure_type": "parallel ensemble decision workflow",
            "style_notes": (
                "Show three parallel classifier branches converging into an aggregation block, include the key formulas "
                "inside or below the aggregation and decision stages, and close with evidence consolidation plus a "
                "validation feedback loop."
            ),
        },
        "uig_discovery": {
            "method_content": (
                "Step 4 operates only on the relevant set R. Draw B bootstrap samples S_b of size n=1000 with replacement. "
                "Within each batch, run K independent Propose passes that each generate at most 25 Unique Impact Groups "
                "(UIGs) plus article assignments and evidence. Harmonize the run-level UIGs with semantic embedding "
                "clustering into canonical UIG clusters. Adjudicate with Support_bu = #runs_containing_u / K and "
                "Coverage_bu = #articles_assigned_to_u / n; retain UIGs with Support_bu >= tau_u and, if needed, keep "
                "the top 25 by Score_bu = alpha * Support_bu + (1 - alpha) * Coverage_bu. Run a constrained judge pass "
                "for final batch labels, then align batch UIGs across bootstrap samples into a global taxonomy and retain "
                "robust global UIGs with selection probability pi_g = #{b : g in U_b} / B >= pi*."
            ),
            "figure_caption": (
                "Impact extraction via Propose-Harmonize-Adjudicate within bootstrap batches and cross-batch UIG synthesis"
            ),
            "figure_type": "bootstrap taxonomy discovery workflow",
            "style_notes": (
                "Use a three-level vertical composition: bootstrap sampling at the top, within-batch "
                "Propose/Harmonize/Adjudicate and judge pass in the middle, and cross-batch global taxonomy synthesis "
                "at the bottom."
            ),
        },
    }

    for name, payload in specs.items():
        out_path = INPUTS_DIR / f"{name}.json"
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"  wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
