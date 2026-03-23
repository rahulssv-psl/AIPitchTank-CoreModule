from __future__ import annotations

from .agents import BuildScriptAgent, GitHubIssuesAgent, PatchAgent, WebDiscoveryAgent
from .models import AgentName, AgentResult, CoreRequest
from .orchestrator import CorePortingOrchestrator


class AgentDispatcher:
    def __init__(self) -> None:
        self.core = CorePortingOrchestrator()
        self._single_agents = {
            AgentName.BUILD_SCRIPT: BuildScriptAgent(),
            AgentName.PATCH: PatchAgent(),
            AgentName.WEB_DISCOVERY: WebDiscoveryAgent(),
            AgentName.GITHUB_ISSUES: GitHubIssuesAgent(),
        }

    @staticmethod
    def _resolve_agent_name(name: AgentName | str) -> AgentName:
        if isinstance(name, AgentName):
            return name

        text = str(name).strip()
        if not text:
            return AgentName.CORE

        for candidate in AgentName:
            if text.lower() == candidate.value.lower():
                return candidate
        return AgentName.CORE

    async def run(self, request: CoreRequest):
        target = self._resolve_agent_name(request.agent_name)
        if target == AgentName.CORE:
            return await self.core.run(request)

        agent = self._single_agents[target]
        result: AgentResult = await agent.run(request.input)
        return result
