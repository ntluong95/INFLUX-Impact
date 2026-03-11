REPO_ROOT := $(shell cd "$(CURDIR)/.." && pwd)

export RSS_CHUNK_SIZE
export VALIDATION_SAMPLE_SIZE
export FULLTEXT_RATE_LIMIT
export TIMEOUT_SECONDS
export GOOGLE_NEWS_LANG
export GOOGLE_NEWS_COUNTRY
export GOOGLE_NEWS_PROXY_BACKEND
export GOOGLE_NEWS_HTTP_PROXY
export GOOGLE_NEWS_HTTPS_PROXY
export SCRAPING_BEE_API_KEY
export OPENAI_API_KEY
export OPENAI_BASE_URL
export OPENAI_MODEL
export LOCAL_LLM_PROVIDER
export LOCAL_LLM_BASE_URL
export LOCAL_LLM_MODEL

.PHONY: zika-rss zika-rss-fetch zika-rss-parse zika-filter zika-fulltext zika-report zika-all

zika-rss-fetch:
	cd "$(REPO_ROOT)" && uv run python zika/python/01_retrieve_google_rss.py

zika-rss-parse:
	cd "$(REPO_ROOT)" && uv run python zika/python/01_parse_google_rss.py

zika-rss: zika-rss-fetch zika-rss-parse

zika-filter:
	cd "$(REPO_ROOT)" && uv run python zika/python/run_stage2_with_patched_ollama.py

zika-fulltext:
	cd "$(REPO_ROOT)" && uv run python zika/python/03_retrieve_fulltext.py

zika-report:
	cd "$(REPO_ROOT)" && sh -lc 'command -v quarto >/dev/null 2>&1 || { echo "Quarto CLI not found. Install Quarto to render zika/report/zika_pipeline_report.qmd."; exit 1; }; quarto render zika/report/zika_pipeline_report.qmd'

zika-all: zika-rss zika-filter zika-fulltext
