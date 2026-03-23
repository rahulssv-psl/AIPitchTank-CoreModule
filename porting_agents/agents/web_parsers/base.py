from __future__ import annotations

from abc import ABC, abstractmethod
from logging import getLogger


class BaseParser(ABC):
    def __init__(self) -> None:
        self.content: list[str] = []
        self._bad_request = False
        self._markdown = ""
        self.headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
        self.logger = getLogger(self.__class__.__name__)

    @abstractmethod
    async def from_url(self, url: str, recurse: bool = False) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_markdown(self) -> str:
        raise NotImplementedError
