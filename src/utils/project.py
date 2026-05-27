from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.common import ensure_dir, normalize_whitespace


SUPPORTED_PATHOGEN_DOMAINS = ("human", "animal", "plant")
SUPPORTED_LANGUAGES = ("en", "fr", "es", "pt")

DEFAULT_LANGUAGE_SPECS: dict[str, dict[str, Any]] = {
    "en": {
        "locales": [
            {
                "id": "en_us",
                "lang": "en",
                "country": "US",
                "accept_language": "en-US,en-GB,en-CA,en-AU,en-NZ,en;q=0.9",
            },
            {
                "id": "en_gb",
                "lang": "en",
                "country": "GB",
                "accept_language": "en-GB,en-US,en-IE,en-CA,en-AU,en;q=0.9",
            },
            {
                "id": "en_ca",
                "lang": "en",
                "country": "CA",
                "accept_language": "en-CA,en-US,en-GB,en;q=0.9",
            },
        ],
    },
    "fr": {
        "locales": [
            {
                "id": "fr_fr",
                "lang": "fr",
                "country": "FR",
                "accept_language": "fr-FR,fr-CA,fr-BE,fr-CH,fr;q=0.9,en;q=0.3",
            },
            {
                "id": "fr_ca",
                "lang": "fr",
                "country": "CA",
                "accept_language": "fr-CA,fr-FR,fr-BE,fr-CH,fr;q=0.9,en;q=0.3",
            },
            {
                "id": "fr_be",
                "lang": "fr",
                "country": "BE",
                "accept_language": "fr-BE,fr-FR,fr-CH,fr-CA,fr;q=0.9,en;q=0.3",
            },
        ],
    },
    "es": {
        "locales": [
            {
                "id": "es_es",
                "lang": "es",
                "country": "ES",
                "accept_language": "es-ES,es-419,es-MX,es-AR,es-CO,es-CL,es-PE,es;q=0.9,en;q=0.3",
            },
            {
                "id": "es_mx",
                "lang": "es",
                "country": "MX",
                "accept_language": "es-MX,es-419,es-ES,es-AR,es-CO,es;q=0.9,en;q=0.3",
            },
            {
                "id": "es_ar",
                "lang": "es",
                "country": "AR",
                "accept_language": "es-AR,es-419,es-ES,es-MX,es-CO,es;q=0.9,en;q=0.3",
            },
        ],
    },
    "pt": {
        "locales": [
            {
                "id": "pt_br",
                "lang": "pt-BR",
                "country": "BR",
                "accept_language": "pt-BR,pt-PT,pt-AO,pt-MZ,pt;q=0.9,en;q=0.3",
            },
            {
                "id": "pt_pt",
                "lang": "pt-PT",
                "country": "PT",
                "accept_language": "pt-PT,pt-BR,pt-AO,pt-MZ,pt;q=0.9,en;q=0.3",
            },
        ],
    },
}


@dataclass(frozen=True)
class DatasetKey:
    pathogen_domain: str
    language_code: str

    def __post_init__(self) -> None:
        if self.pathogen_domain not in SUPPORTED_PATHOGEN_DOMAINS:
            raise ValueError(f"Unsupported pathogen domain: {self.pathogen_domain}")
        if self.language_code not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language code: {self.language_code}")

    @property
    def stem(self) -> str:
        return f"{self.pathogen_domain}_{self.language_code}"


@dataclass(frozen=True)
class ProjectPaths:
    repo_root: Path
    data_root: Path

    def logs_dir(self) -> Path:
        return self.data_root / "logs"

    def rss_raw_dir(self, dataset: DatasetKey) -> Path:
        return self.data_root / "raw" / "rss" / dataset.stem

    def rss_manifest_path(self, dataset: DatasetKey) -> Path:
        return self.rss_raw_dir(dataset) / f"{dataset.stem}_manifest.csv"

    def rss_output_csv(self, dataset: DatasetKey) -> Path:
        return self.data_root / "intermediate" / "rss" / f"{dataset.stem}.csv"

    def rss_output_parquet(self, dataset: DatasetKey) -> Path:
        return self.data_root / "intermediate" / "rss" / f"{dataset.stem}.parquet"

    def rss_metrics_csv(self, dataset: DatasetKey) -> Path:
        return self.data_root / "intermediate" / "rss" / f"{dataset.stem}_metrics.csv"

    def rss_scan_summary_csv(self, dataset: DatasetKey) -> Path:
        return self.data_root / "intermediate" / "rss" / f"{dataset.stem}_scan_summary.csv"

    def prefiltered_headlines_csv(self, dataset: DatasetKey) -> Path:
        return (
            self.data_root
            / "intermediate"
            / "classification"
            / f"{dataset.stem}_headlines_prefiltered.csv"
        )

    def prefiltered_headlines_metrics_csv(self, dataset: DatasetKey) -> Path:
        return (
            self.data_root
            / "intermediate"
            / "classification"
            / f"{dataset.stem}_domain_filter_metrics.csv"
        )

    def classified_headlines_csv(self, dataset: DatasetKey) -> Path:
        return (
            self.data_root
            / "intermediate"
            / "classification"
            / f"{dataset.stem}_headlines_classified.csv"
        )

    def batch_dir(self, dataset: DatasetKey) -> Path:
        return self.data_root / "intermediate" / "batches" / dataset.stem

    def batch_registry_csv(self, dataset: DatasetKey) -> Path:
        return self.batch_dir(dataset) / f"{dataset.stem}_batches.csv"

    def batch_requests_dir(self, dataset: DatasetKey) -> Path:
        return self.batch_dir(dataset) / "requests"

    def batch_results_dir(self, dataset: DatasetKey) -> Path:
        return self.batch_dir(dataset) / "results"

    def fulltext_csv(self, dataset: DatasetKey) -> Path:
        return self.data_root / "final" / "fulltext" / f"{dataset.stem}.csv"

    def fulltext_parquet(self, dataset: DatasetKey) -> Path:
        return self.data_root / "final" / "fulltext" / f"{dataset.stem}.parquet"

    def fulltext_failed_csv(self, dataset: DatasetKey) -> Path:
        return self.data_root / "final" / "fulltext" / f"{dataset.stem}_failed.csv"

    def bertopic_dir(self, dataset: DatasetKey) -> Path:
        return self.data_root / "final" / "bertopic" / dataset.stem

    def ensure_parent_dirs(self, dataset: DatasetKey) -> None:
        for path in [
            self.logs_dir(),
            self.rss_raw_dir(dataset),
            self.rss_output_csv(dataset).parent,
            self.prefiltered_headlines_csv(dataset).parent,
            self.batch_dir(dataset),
            self.batch_requests_dir(dataset),
            self.batch_results_dir(dataset),
            self.fulltext_csv(dataset).parent,
            self.bertopic_dir(dataset),
        ]:
            ensure_dir(path)


def resolve_pathogen_domains(value: str) -> list[str]:
    normalized = normalize_whitespace(value).lower()
    if normalized in {"", "all"}:
        return list(SUPPORTED_PATHOGEN_DOMAINS)
    domains = [item.strip() for item in normalized.split(",") if item.strip()]
    invalid = [domain for domain in domains if domain not in SUPPORTED_PATHOGEN_DOMAINS]
    if invalid:
        raise ValueError(f"Unsupported pathogen domains: {', '.join(sorted(invalid))}")
    return domains


def resolve_languages(value: str) -> list[str]:
    normalized = normalize_whitespace(value).lower()
    if normalized in {"", "all"}:
        return list(SUPPORTED_LANGUAGES)
    languages = [item.strip() for item in normalized.split(",") if item.strip()]
    invalid = [code for code in languages if code not in SUPPORTED_LANGUAGES]
    if invalid:
        raise ValueError(f"Unsupported language codes: {', '.join(sorted(invalid))}")
    return languages


def resolve_datasets(pathogen_domains: list[str], languages: list[str]) -> list[DatasetKey]:
    return [
        DatasetKey(pathogen_domain=domain, language_code=language)
        for domain in pathogen_domains
        for language in languages
    ]


def load_search_strings(csv_path: Path, language_code: str | None = None) -> list[str]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing search input CSV: {csv_path}. Expected a search string column."
        )

    suffix = csv_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(csv_path)
    else:
        df = pd.read_csv(csv_path, low_memory=False)
    preferred_columns: list[str] = []
    if language_code:
        preferred_columns.append(f"search_string_{normalize_whitespace(language_code).lower()}")
    preferred_columns.extend(["search_string", "search_string_en"])

    search_column = next((column for column in preferred_columns if column in df.columns), None)
    if search_column is None:
        raise ValueError(
            f"{csv_path} must contain one of these columns: {', '.join(preferred_columns)}."
        )

    values = [
        normalize_whitespace(value)
        for value in df[search_column].fillna("").astype(str).tolist()
    ]
    deduped: list[str] = []
    seen = set()
    for value in values:
        lowered = value.lower()
        if not value or lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(value)
    if not deduped:
        raise ValueError(f"{csv_path} contains no usable search strings.")
    return deduped


def language_spec_for(config: dict[str, Any], language_code: str) -> dict[str, Any]:
    """Return the locale specifications for a language code.

    Locales define the Google News country editions to query. Each locale
    specifies the lang, country, and Accept-Language header used for RSS
    retrieval. Multiple locales per language ensure geographic coverage
    (e.g. en_us, en_gb, en_ca for English).
    """
    override_map = config.get("rss", {}).get("language_specs", {})
    default_spec = DEFAULT_LANGUAGE_SPECS[language_code]
    override_spec = override_map.get(language_code, {})
    return {
        "locales": list(override_spec.get("locales", default_spec["locales"])),
    }
