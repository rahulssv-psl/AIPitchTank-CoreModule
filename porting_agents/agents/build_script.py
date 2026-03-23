from __future__ import annotations

from ddgs import DDGS
from ddgs.exceptions import DDGSException

from ..models import AgentName, AgentResult, BuildFailureInput
from ..utils import clamp_confidence, compact_text, keyword_matches
from .base import BasePortingAgent


class BuildScriptAgent(BasePortingAgent):
    @property
    def name(self) -> str:
        return AgentName.BUILD_SCRIPT.value

    async def run(self, request_input: BuildFailureInput) -> AgentResult:
        query = (
            "site:github.com/ppc64le/build-scripts "
            f"{request_input.package_name} {request_input.requested_package_version} "
            f'{" ".join(request_input.architecture_targets)}'
        )

        scripts: list[dict[str, str]] = []
        try:
            with DDGS() as se:
                results = list(se.text(query=query, max_results=8))
        except DDGSException as ex:
            return AgentResult(
                agent_name=AgentName.BUILD_SCRIPT,
                status="error",
                summary=f"Build script search failed: {ex}",
                data={"scripts": []},
                confidence=0.1,
            )

        for item in results:
            href = item.get("href", "")
            title = item.get("title", "")
            body = item.get("body", "")
            if "github.com/ppc64le/build-scripts" not in href:
                continue
            scripts.append(
                {
                    "url": href,
                    "title": compact_text(title, 140),
                    "hint": compact_text(body, 240),
                }
            )

        evidence = " ".join((entry["title"] + " " + entry["hint"]) for entry in scripts)
        terms = [
            request_input.package_name,
            request_input.requested_package_version,
            "patch",
            "ppc64le",
            "s390x",
            "build",
        ]
        confidence = clamp_confidence(min(0.95, 0.25 + 0.1 * keyword_matches(evidence, terms)))

        return AgentResult(
            agent_name=AgentName.BUILD_SCRIPT,
            status="success" if scripts else "partial",
            summary="Found relevant build-script references." if scripts else "No strongly matching build scripts found.",
            data={"scripts": scripts, "query": query},
            confidence=confidence if scripts else 0.25,
        )
