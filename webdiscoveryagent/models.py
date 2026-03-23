from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class WebDiscoveryInput(BaseModel):
    package_name: str = Field(min_length=1)
    requested_package_version: str = Field(min_length=1)
    architecture_targets: list[str] = Field(default_factory=lambda: ["ppc64le", "s390x"])
    error_message: str = Field(min_length=1)
    max_results: int = Field(default=5, ge=1, le=10)


class WebReference(BaseModel):
    url: str
    source: str
    insight: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class WebDiscoveryResult(BaseModel):
    status: str
    summary: str
    query: str
    references: list[WebReference] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
