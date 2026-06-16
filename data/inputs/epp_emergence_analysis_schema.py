from __future__ import annotations

from typing import Any


def response_schema(include_review: bool = False) -> dict[str, Any]:
    source = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "url": {"type": "string"},
            "title": {"type": "string"},
            "publisher": {"type": "string"},
            "used_for": {"type": "string"},
        },
        "required": ["url", "title", "publisher", "used_for"],
    }
    event = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "year": {"type": "string"},
            "continent": {"type": "string"},
            "sub_regions": {"type": "string"},
            "countries": {"type": "string"},
            "host": {"type": "string"},
            "first_global_emergence": {"type": "boolean"},
            "first_detection": {"type": "boolean"},
            "intra_continental_emergence": {"type": "boolean"},
            "inter_continental_emergence": {"type": "boolean"},
            "re_occurrence": {"type": "boolean"},
            "re_emergence": {"type": "boolean"},
            "confidence": {"type": "number"},
            "event_summary": {"type": "string"},
            "mechanism_or_driver": {"type": "string"},
            "source_urls": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "year", "continent", "sub_regions", "countries", "host",
            "first_global_emergence", "first_detection", "intra_continental_emergence",
            "inter_continental_emergence", "re_occurrence", "re_emergence",
            "confidence", "event_summary", "mechanism_or_driver", "source_urls",
        ],
    }
    if include_review:
        event["properties"]["review"] = {"type": "string"}
        event["required"].append("review")
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "scientific_name_standardize": {"type": "string"},
            "common_name": {"type": "string"},
            "host_system": {"type": "string"},
            "emergence_suitability": {"type": "string", "enum": ["Yes", "Maybe", "No"]},
            "first_emergence_time": {"type": "string"},
            "first_emergence_place": {"type": "string"},
            "first_emergence_summary": {"type": "string"},
            "major_subregions_2005_2025": {"type": "string"},
            "emergence_event_count_2005_2025": {"type": "integer"},
            "no_event_reason": {"type": "string"},
            "quality_notes": {"type": "string"},
            "events": {"type": "array", "items": event},
            "source_urls": {"type": "array", "items": source},
        },
        "required": [
            "scientific_name_standardize", "common_name", "host_system",
            "emergence_suitability", "first_emergence_time",
            "first_emergence_place", "first_emergence_summary",
            "major_subregions_2005_2025", "emergence_event_count_2005_2025",
            "no_event_reason", "quality_notes", "events", "source_urls",
        ],
    }
    if include_review:
        schema["properties"]["review_summary"] = {"type": "string"}
        schema["required"].append("review_summary")
    return schema
