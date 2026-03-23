from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from ..config import create_watsonx_chat_model
from ..models import AgentName, AgentResult, BuildFailureInput
from ..utils import clamp_confidence, compact_text, safe_json_load
from .base import BasePortingAgent


class PatchAgent(BasePortingAgent):
    def __init__(self) -> None:
        self.model = create_watsonx_chat_model()

    @property
    def name(self) -> str:
        return AgentName.PATCH.value

    async def run(self, request_input: BuildFailureInput) -> AgentResult:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a Linux package porting expert. "
                    "Return JSON with keys: patch_diff, commit_message, steps, rationale, confidence. "
                    "Patch should target ppc64le/s390x build compatibility.",
                ),
                (
                    "human",
                    "Package: {package}\nVersion: {version}\nRepo: {repo}\nTargets: {targets}\nError:\n{error}\n",
                ),
            ]
        )
        chain = prompt | self.model
        response = await chain.ainvoke(
            {
                "package": request_input.package_name,
                "version": request_input.requested_package_version,
                "repo": str(request_input.github_repo_url),
                "targets": ", ".join(request_input.architecture_targets),
                "error": request_input.error_message,
            }
        )
        payload = safe_json_load(response.content if hasattr(response, "content") else str(response))

        patch_diff = payload.get(
            "patch_diff",
            (
                "--- a/build.sh\n+++ b/build.sh\n@@\n"
                "+# TODO: add architecture mapping for ppc64le/s390x\n"
            ),
        )
        commit_message = payload.get("commit_message", "Improve architecture compatibility for package build")
        steps = payload.get(
            "steps",
            [
                "Apply patch to build script or configure file",
                "Re-run package build on ppc64le and s390x",
                "Validate tests and packaging metadata",
            ],
        )
        rationale = payload.get("rationale", "Suggested patch based on architecture-specific build failure context.")
        confidence = clamp_confidence(float(payload.get("confidence", 0.65)))

        data: dict[str, Any] = {
            "patch_diff": compact_text(patch_diff, 2000),
            "commit_message": commit_message,
            "steps": steps if isinstance(steps, list) else [str(steps)],
            "rationale": rationale,
        }

        return AgentResult(
            agent_name=AgentName.PATCH,
            status="success",
            summary="Generated patch direction for resolving build failure.",
            data=data,
            confidence=confidence,
        )
