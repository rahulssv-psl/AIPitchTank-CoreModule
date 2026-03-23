from __future__ import annotations

from webdiscoveryagent import WebDiscoveryInput, run_web_discovery

from ..models import AgentName, AgentResult, BuildFailureInput
from .base import BasePortingAgent


class WebDiscoveryAgent(BasePortingAgent):
    @property
    def name(self) -> str:
        return AgentName.WEB_DISCOVERY.value

    async def run(self, request_input: BuildFailureInput) -> AgentResult:
        result = await run_web_discovery(
            WebDiscoveryInput(
                package_name=request_input.package_name,
                requested_package_version=request_input.requested_package_version,
                architecture_targets=request_input.architecture_targets,
                error_message=request_input.error_message,
            )
        )
        return AgentResult(
            agent_name=AgentName.WEB_DISCOVERY,
            status=result.status,
            summary=result.summary,
            data={
                "references": [reference.model_dump() for reference in result.references],
                "query": result.query,
            },
            confidence=result.confidence,
        )
