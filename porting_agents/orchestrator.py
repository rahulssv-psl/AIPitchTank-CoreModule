from __future__ import annotations

import asyncio
import operator
from typing import Annotated, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from .agents import BuildScriptAgent, GitHubIssuesAgent, PatchAgent, WebDiscoveryAgent
from .config import create_watsonx_chat_model
from .models import (
    AgentName,
    AgentResult,
    BuildFailureInput,
    CoreRequest,
    CoreResponse,
    FinalRecommendation,
)
from .utils import clamp_confidence, safe_json_load


class GraphState(TypedDict):
    request: CoreRequest
    selected_agents: list[AgentName]
    outcomes: Annotated[list[AgentResult], operator.add]
    recommendation: FinalRecommendation | None


class CorePortingOrchestrator:
    def __init__(self) -> None:
        self.web_agent = WebDiscoveryAgent()
        self.build_agent = BuildScriptAgent()
        self.patch_agent = PatchAgent()
        self.github_agent = GitHubIssuesAgent()
        self.planner_model = create_watsonx_chat_model()
        self.synth_model = create_watsonx_chat_model()
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(GraphState)
        workflow.add_node("select_agents", self._select_agents_node)
        workflow.add_node("execute_agents", self._execute_agents_node)
        workflow.add_node("synthesize", self._synthesize_node)
        workflow.set_entry_point("select_agents")
        workflow.add_edge("select_agents", "execute_agents")
        workflow.add_edge("execute_agents", "synthesize")
        workflow.add_edge("synthesize", END)
        return workflow.compile()

    @staticmethod
    def _heuristic_selection(request_input: BuildFailureInput) -> list[AgentName]:
        selected = [AgentName.BUILD_SCRIPT, AgentName.WEB_DISCOVERY]
        lowered = request_input.error_message.lower()

        patch_terms = ["patch", ".rej", "hunk", "apply", "diff", "context mismatch"]
        github_terms = ["issue", "regression", "upstream", "bug", "workaround", "module not found", "cmake"]
        torch_terms = ["torch", "triton", "cuda", "compile", "dynamo"]

        if any(term in lowered for term in patch_terms):
            selected.append(AgentName.PATCH)
        if any(term in lowered for term in github_terms):
            selected.append(AgentName.GITHUB_ISSUES)
        if any(term in lowered for term in torch_terms):
            selected.extend([AgentName.PATCH, AgentName.GITHUB_ISSUES])

        # Deduplicate while preserving order.
        deduped: list[AgentName] = []
        for item in selected:
            if item not in deduped:
                deduped.append(item)
        return deduped

    async def _plan_selection_with_llm(self, request_input: BuildFailureInput) -> list[AgentName]:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a routing planner for software-porting agents. "
                    "Return strict JSON with key selected_agents as array of names from: "
                    "BuildScriptAgent, PatchAgent, WebDiscoveryAgent, GitHubIssuesAgent.",
                ),
                (
                    "human",
                    "Package={package}\nVersion={version}\nTargets={targets}\nError={error}",
                ),
            ]
        )
        chain = prompt | self.planner_model
        response = await chain.ainvoke(
            {
                "package": request_input.package_name,
                "version": request_input.requested_package_version,
                "targets": ", ".join(request_input.architecture_targets),
                "error": request_input.error_message,
            }
        )
        payload = safe_json_load(response.content if hasattr(response, "content") else str(response))
        selected_agents_raw = payload.get("selected_agents", [])
        mapping = {
            AgentName.BUILD_SCRIPT.value: AgentName.BUILD_SCRIPT,
            AgentName.PATCH.value: AgentName.PATCH,
            AgentName.WEB_DISCOVERY.value: AgentName.WEB_DISCOVERY,
            AgentName.GITHUB_ISSUES.value: AgentName.GITHUB_ISSUES,
        }
        selected = [mapping[item] for item in selected_agents_raw if item in mapping]
        if not selected:
            selected = self._heuristic_selection(request_input)
        return selected

    async def _select_agents_node(self, state: GraphState):
        request = state["request"]
        base_selected = self._heuristic_selection(request.input)
        if request.options.llm_planning:
            llm_selected = await self._plan_selection_with_llm(request.input)
            merged = []
            for item in [*base_selected, *llm_selected]:
                if item not in merged:
                    merged.append(item)
            selected = merged[: request.options.max_agents]
        else:
            selected = base_selected[: request.options.max_agents]
        return {"selected_agents": selected}

    async def _run_selected_agent(self, agent_name: AgentName, request_input: BuildFailureInput) -> AgentResult:
        if agent_name == AgentName.BUILD_SCRIPT:
            return await self.build_agent.run(request_input)
        if agent_name == AgentName.PATCH:
            return await self.patch_agent.run(request_input)
        if agent_name == AgentName.GITHUB_ISSUES:
            return await self.github_agent.run(request_input)
        if agent_name == AgentName.WEB_DISCOVERY:
            return await self.web_agent.run(request_input)
        raise ValueError(f"Unsupported agent: {agent_name}")

    async def _execute_agents_node(self, state: GraphState):
        request_input = state["request"].input
        selected = state["selected_agents"]
        tasks = [self._run_selected_agent(name, request_input) for name in selected]
        outcomes = await asyncio.gather(*tasks)
        return {"outcomes": outcomes}

    async def _synthesize_node(self, state: GraphState):
        request_input = state["request"].input
        outcomes = state["outcomes"]
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a senior package-porting engineer for ppc64le and s390x migrations. "
                    "Return strict JSON with keys: diagnosis, best_actions (array), proposed_patch, references (array), confidence.",
                ),
                (
                    "human",
                    "Input:\n{request_input}\n\nAgent Results:\n{agent_results}",
                ),
            ]
        )
        chain = prompt | self.synth_model
        response = await chain.ainvoke(
            {
                "request_input": request_input.model_dump_json(indent=2),
                "agent_results": [item.model_dump() for item in outcomes],
            }
        )
        payload = safe_json_load(response.content if hasattr(response, "content") else str(response))

        diagnosis = payload.get(
            "diagnosis",
            "Likely architecture-specific build incompatibility requiring script and dependency adjustments.",
        )
        best_actions = payload.get(
            "best_actions",
            [
                "Check matching build scripts in ppc64le/build-scripts for package family/version.",
                "Apply architecture-guarded patch and rerun build on ppc64le and s390x.",
                "Validate against upstream issue workarounds before finalizing patch.",
            ],
        )
        proposed_patch = payload.get("proposed_patch")
        references = payload.get("references", [])
        confidence = clamp_confidence(float(payload.get("confidence", 0.75)))

        recommendation = FinalRecommendation(
            diagnosis=diagnosis,
            best_actions=best_actions if isinstance(best_actions, list) else [str(best_actions)],
            proposed_patch=proposed_patch,
            references=references if isinstance(references, list) else [str(references)],
            confidence=confidence,
        )
        return {"recommendation": recommendation}

    async def run(self, request: CoreRequest) -> CoreResponse:
        graph_result = await self.graph.ainvoke(
            {"request": request, "selected_agents": [], "outcomes": [], "recommendation": None}
        )

        outcomes: list[AgentResult] = graph_result["outcomes"]
        recommendation: FinalRecommendation = graph_result["recommendation"]
        selected_agents: list[AgentName] = graph_result["selected_agents"]

        if outcomes:
            confidence = clamp_confidence(
                (sum(item.confidence for item in outcomes) / len(outcomes) + recommendation.confidence) / 2
            )
        else:
            confidence = recommendation.confidence

        return CoreResponse(
            status="success",
            summary="Generated robust multi-agent recommendation for build-porting failure.",
            selected_agents=selected_agents,
            results=outcomes,
            recommendation=recommendation,
            confidence=confidence,
        )
