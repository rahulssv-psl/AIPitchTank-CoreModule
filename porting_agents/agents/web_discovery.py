from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from ddgs import DDGS
from ddgs.exceptions import DDGSException

from ..models import AgentName, AgentResult, BuildFailureInput
from ..utils import clamp_confidence, compact_text, keyword_matches
from .base import BasePortingAgent
from .web_parsers import GenericParser, StackOverflowParser


class WebDiscoveryAgent(BasePortingAgent):
    @property
    def name(self) -> str:
        return AgentName.WEB_DISCOVERY.value

    async def run(self, request_input: BuildFailureInput) -> AgentResult:
        search_query = (
            f'{request_input.package_name} {request_input.requested_package_version} '
            f'{" ".join(request_input.architecture_targets)} {compact_text(request_input.error_message, 220)}'
        )

        references: list[dict[str, Any]] = []
        try:
            with DDGS() as se:
                results = list(se.text(query=search_query, max_results=5))
        except DDGSException as ex:
            return AgentResult(
                agent_name=AgentName.WEB_DISCOVERY,
                status="error",
                summary=f"Web search failed: {ex}",
                data={"references": []},
                confidence=0.1,
            )

        for item in results:
            url = item.get("href", "")
            snippet = item.get("body", "") or item.get("title", "")
            if not url:
                continue

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
                {
                    "url": url,
                    "insight": compact_text(parsed_markdown or snippet, 500),
                    "source": domain,
                }
            )

        evidence_text = " ".join(item["insight"] for item in references)
        signal_terms = [
            request_input.package_name,
            *request_input.architecture_targets,
            "ppc64le",
            "s390x",
            "patch",
            "build",
            "failed",
        ]
        signal_score = keyword_matches(evidence_text, signal_terms)
        confidence = clamp_confidence(min(0.95, 0.3 + 0.1 * signal_score))

        summary = "Found similar error discussions online." if references else "No strong web references were found."
        return AgentResult(
            agent_name=AgentName.WEB_DISCOVERY,
            status="success" if references else "partial",
            summary=summary,
            data={"references": references, "query": search_query},
            confidence=confidence if references else 0.25,
        )
