from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import AgentResult, BuildFailureInput


class BasePortingAgent(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def run(self, request_input: BuildFailureInput) -> AgentResult:
        raise NotImplementedError
