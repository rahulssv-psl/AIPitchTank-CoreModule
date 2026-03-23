from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl, field_validator


class AgentName(str, Enum):
    BUILD_SCRIPT = "BuildScriptAgent"
    PATCH = "PatchAgent"
    WEB_DISCOVERY = "WebDiscoveryAgent"
    GITHUB_ISSUES = "GitHubIssuesAgent"
    CORE = "CorePortingAgent"


class BuildFailureInput(BaseModel):
    package_name: str = Field(min_length=1)
    requested_package_version: str = Field(min_length=1)
    github_repo_url: HttpUrl
    error_message: str = Field(min_length=1)
    architecture_targets: list[str] = Field(default_factory=lambda: ["ppc64le", "s390x"])

    @field_validator("package_name", "requested_package_version", "error_message")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Value cannot be empty")
        return stripped

    @field_validator("architecture_targets")
    @classmethod
    def normalize_architectures(cls, value: list[str]) -> list[str]:
        normalized = sorted({item.strip().lower() for item in value if item.strip()})
        if not normalized:
            return ["ppc64le", "s390x"]
        return normalized


class RequestOptions(BaseModel):
    max_agents: int = Field(default=4, ge=1, le=4)
    prefer_local_knowledge: bool = True
    include_raw_evidence: bool = True
    llm_planning: bool = True


class CoreRequest(BaseModel):
    agent_name: AgentName | str = AgentName.CORE
    input: BuildFailureInput
    options: RequestOptions = Field(default_factory=RequestOptions)


class AgentResult(BaseModel):
    agent_name: AgentName
    status: str
    summary: str
    data: dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)


class FinalRecommendation(BaseModel):
    diagnosis: str
    best_actions: list[str]
    proposed_patch: str | None = None
    references: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class CoreResponse(BaseModel):
    agent_name: AgentName = AgentName.CORE
    status: str
    summary: str
    selected_agents: list[AgentName]
    results: list[AgentResult]
    recommendation: FinalRecommendation
    confidence: float = Field(ge=0.0, le=1.0)
