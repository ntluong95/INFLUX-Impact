REPO_ROOT := $(shell cd "$(CURDIR)/.." && pwd)

export RSS_CHUNK_SIZE
export VALIDATION_SAMPLE_SIZE
export FULLTEXT_RATE_LIMIT
export TIMEOUT_SECONDS
export OPENAI_API_KEY
export OPENAI_BASE_URL
export OPENAI_MODEL
export LOCAL_LLM_BASE_URL
export LOCAL_LLM_MODEL

.PHONY: zika-rss zika-filter zika-fulltext zika-all

zika-rss:
	cd "$(REPO_ROOT)" && Rscript zika/R/01_retrieve_google_rss.R

zika-filter:
	cd "$(REPO_ROOT)" && Rscript zika/R/02_filter_headlines_ensemble.R

zika-fulltext:
	cd "$(REPO_ROOT)" && uv run python zika/python/03_retrieve_fulltext.py

zika-all: zika-rss zika-filter zika-fulltext
