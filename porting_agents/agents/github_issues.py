from __future__ import annotations

from ddgs import DDGS
from ddgs.exceptions import DDGSException

from ..models import AgentName, AgentResult, BuildFailureInput
from ..utils import clamp_confidence, compact_text, keyword_matches
from .base import BasePortingAgent


class GitHubIssuesAgent(BasePortingAgent):
    @property
    def name(self) -> str:
        return AgentName.GITHUB_ISSUES.value

    async def run(self, request_input: BuildFailureInput) -> AgentResult:
        trimmed_error = compact_text(request_input.error_message, 180)
        query = (
            "site:github.com inurl:issues "
            f"{request_input.package_name} {request_input.requested_package_version} "
            f"{' '.join(request_input.architecture_targets)} {trimmed_error}"
        )
        issues: list[dict[str, str]] = []

        try:
            with DDGS() as se:
                results = list(se.text(query=query, max_results=6))
        except DDGSException as ex:
            return AgentResult(
                agent_name=AgentName.GITHUB_ISSUES,
                status="error",
                summary=f"GitHub issue search failed: {ex}",
                data={"issues": []},
                confidence=0.1,
            )

        for item in results:
            href = item.get("href", "")
            title = item.get("title", "")
            body = item.get("body", "")
            if "github.com" not in href or "/issues/" not in href:
                continue
            issues.append(
                {
                    "url": href,
                    "title": compact_text(title, 180),
                    "resolution": compact_text(body, 300),
                }
            )

        evidence = " ".join((x["title"] + " " + x["resolution"]) for x in issues)
        terms = [
            request_input.package_name,
            request_input.requested_package_version,
            "ppc64le",
            "s390x",
            "fix",
            "build",
            "patch",
        ]
        confidence = clamp_confidence(min(0.95, 0.25 + 0.1 * keyword_matches(evidence, terms)))

        return AgentResult(
            agent_name=AgentName.GITHUB_ISSUES,
            status="success" if issues else "partial",
            summary="Found related GitHub issues." if issues else "No strongly relevant GitHub issues found.",
            data={"issues": issues, "query": query},
            confidence=confidence if issues else 0.25,
        )
