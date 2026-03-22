
You are an expert Python ML/data engineer. I’m on a MacBook Pro (Apple silicon) with 16GB RAM. I have an OpenAI API key stored in a local .env file (do NOT hardcode secrets). Use OpenAI’s API as the default inference backend for all LLM calls.

Please adapt the existing **Zika pipeline** in `zika` folder and implement its functionality within the **`src`** folder, reusing as much of the established Zika codebase as possible while respecting the new project structure and scope.

## Project goal

The overall aim is to build a news-processing pipeline for **human, animal, and plant pathogens**, focused on identifying and exploring **cascading impacts on social-ecological systems**. The Zika workflow already retrieves Google RSS articles, filters relevance, retrieves full text, and applies BERTopic. I want the same overall logic extended and reorganized for the broader pathogen project in `src`.

## Main differences from the Zika folder

### 1. Scope of diseases

- The **`zika`** folder focuses only on **Zika**.
- The **`src`** folder must support **three pathogen domains**:
    - **human**
    - **animal**
    - **plant**

### 2. Code organization

- In the **Zika** folder, scripts are relatively flat.
- In the **`src`** folder, scripts must follow the existing modular structure with subfolders such as:
    - `rss_retrieval`
    - `classification`
    - `fulltext_retrieval`
    - `extraction`
    - `utils`
- Please place each adapted script into the correct subfolder and keep the codebase organized and maintainable.

### 3. Google RSS retrieval

The RSS retrieval logic from the Zika workflow should be generalized for the new project.

#### Input

- Each species/domain should have its own `.csv` input file:
    - human diseases
    - animal diseases
    - plant diseases
- In each CSV, use the column:
    - `search_string`

#### Search settings

- Search timeframe:
    - from **2005-01-01** to **2025-12-31**
- Languages:
    - **English**
    - **French**
    - **Spanish**
    - **Portuguese**

#### Output

- Retrieval outputs must be saved **separately by species and language**
- Expected output files:
    - `human_en`
    - `human_fr`
    - `human_es`
    - `human_pt`
    - `animal_en`
    - `animal_fr`
    - `animal_es`
    - `animal_pt`
    - `plant_en`
    - `plant_fr`
    - `plant_es`
    - `plant_pt`

So there should be **12 separate output files** for RSS retrieval.

### 4. Domain filtering before AI headline filtering

Before running AI-based relevance filtering on headlines:

- Import `data/domains_removed.csv`
- Use the column:
    - `is_news_outlet`
- Keep only rows where `is_news_outlet == "No"`
- Use that resulting domain list to filter the retrieved headlines before headline classification

### 5. Headline relevance filtering

The Zika workflow used an **ensemble** approach for headline filtering. In `src`, replace that with:

- **OpenAI `gpt-5-nano` Batch API** only
- The scripts should be designed to prepare **multiple batches** as needed

Please account for Batch API constraints:

- A single batch may contain up to **50,000 requests**
- A single batch input file may be up to **200 MB**
- Batch creation rate limit is **2,000 batches per hour**
- Batch API rate limits are separate from normal per-model rate limits
- Also consider the per-model limit on **enqueued prompt tokens**

### 6. Batch naming

Batch submission names should be informative so I can immediately tell:

- which **species** is being processed
- which **language** is being processed

Use a naming convention similar to the RSS retrieval step.

### 7. Full-text retrieval

The full-text retrieval stage should remain functionally the same as in the Zika pipeline, but adapted to the new `src` structure.

#### Output

- Save outputs into the same **12 separate species-language files**
- Keep naming consistent with previous stages

### 8. BERTopic stage

Adapt `04_bertopic` for the broader pathogen project.

#### Requirements

- BERTopic should operate on **full-text articles**
- It may use **different embedding models** depending on the pathogen domain:
    - human disease news
    - animal disease news
    - plant disease news

#### Objective

The purpose of this stage is **not** final extraction yet. Instead, BERTopic should provide an **initial overview of major impact-related themes** in news articles, especially topics related to the **cascading impacts of emerging pests and pathogens on social-ecological systems**.

This topic exploration should help inform a later impact-extraction pipeline that will use an ensemble of two AIs, but for now, you do **not** need to build that later extraction step.

#### Conceptual inspiration

You may take inspiration from this paper, especially its framing of outbreak response phases, but do **not** need to replicate it exactly:

- [https://www.sciencedirect.com/science/article/pii/S2949856224000230](https://www.sciencedirect.com/science/article/pii/S2949856224000230)

Use it only as conceptual guidance for thinking about possible impact dimensions, response phases, and topic framing.

## What I want from you

Please do the following:

1. Review the existing Zika code and identify which parts can be reused directly and which must be generalized.
2. Implement the adapted pipeline inside `src`, preserving the modular folder structure.
3. Create or refactor scripts for:
    - RSS retrieval
    - domain filtering
    - AI headline filtering with `gpt-5-nano` Batch API
    - full-text retrieval
    - BERTopic on full text
4. Ensure outputs are consistently separated into the 12 species-language combinations.
5. Keep the code clean, modular, and easy to extend.
6. Every steps should have the checkpoint and resumable

## Additional expectations

- Reuse existing Zika logic where sensible instead of rewriting everything from scratch.
- Make file naming and batch naming explicit and traceable.
- Add comments where adaptation decisions are important.
- Favor maintainable code over quick hacks.
