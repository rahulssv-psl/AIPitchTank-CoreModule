from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from ddgs import DDGS
from ddgs.exceptions import DDGSException

from .models import WebDiscoveryInput, WebDiscoveryResult, WebReference
from .parsers import GenericParser, StackOverflowParser


def _clamp_confidence(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 3)


def _compact_text(value: str, limit: int = 500) -> str:
    normalized = re.sub(r"\s+", " ", value).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _keyword_matches(text: str, terms: list[str]) -> int:
    lowered = text.lower()
    return sum(1 for term in terms if term.lower() in lowered)


def _build_query(payload: WebDiscoveryInput) -> str:
    return (
        f"{payload.package_name} {payload.requested_package_version} "
        f'{" ".join(payload.architecture_targets)} {_compact_text(payload.error_message, 220)}'
    )


async def run_web_discovery(payload: WebDiscoveryInput) -> WebDiscoveryResult:
    query = _build_query(payload)
    references: list[WebReference] = []

    try:
        with DDGS() as search:
            results = list(search.text(query=query, max_results=payload.max_results))
    except DDGSException as ex:
        return WebDiscoveryResult(
            status="error",
            summary=f"Web search failed: {ex}",
            query=query,
            references=[],
            confidence=0.1,
        )

    for item in results:
        url = item.get("href", "")
        if not url:
            continue
        snippet = item.get("body", "") or item.get("title", "")
        domain = urlparse(url).netloc

        parsed_markdown = ""
        if re.match(r"https://stackoverflow.com/questions/\d+/", url):
            parser = StackOverflowParser()
            await parser.from_url(url, recurse=False)
            parsed_markdown = parser.get_markdown()
        elif domain:
            parser = GenericParser()
            await parser.from_url(url, recurse=False)
            parsed_markdown = parser.get_markdown()

        references.append(
            WebReference(
                url=url,
                source=domain,
                insight=_compact_text(parsed_markdown or snippet, 500),
                metadata={
                    "title": item.get("title", ""),
                },
            )
        )

    evidence_text = " ".join(reference.insight for reference in references)
    signal_terms = [
        payload.package_name,
        *payload.architecture_targets,
        "ppc64le",
        "s390x",
        "patch",
        "build",
        "failed",
    ]
    score = _keyword_matches(evidence_text, signal_terms)
    confidence = _clamp_confidence(min(0.95, 0.3 + 0.1 * score))
    summary = "Found similar error discussions online." if references else "No strong web references were found."

    return WebDiscoveryResult(
        status="success" if references else "partial",
        summary=summary,
        query=query,
        references=references,
        confidence=confidence if references else 0.25,
    )
