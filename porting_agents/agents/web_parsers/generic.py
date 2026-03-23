from __future__ import annotations

from aiohttp import ClientSession
from aiolimiter import AsyncLimiter
from trafilatura import extract

from .base import BaseParser


class GenericParser(BaseParser):
    def __init__(self) -> None:
        super().__init__()
        self.limiter = AsyncLimiter(2, 1)

    async def from_url(self, url: str, recurse: bool = False) -> None:
        async with ClientSession(headers=self.headers) as session:
            async with self.limiter:
                async with session.get(url) as response:
                    if response.status != 200:
                        self._bad_request = True
                        self._markdown = f"Failed to fetch {url}: {response.status}"
                        return
                    self.content = [await response.text()]

    def get_markdown(self) -> str:
        if self._markdown:
            return self._markdown
        if not self.content:
            return ""
        self._markdown = extract(self.content[0]) or ""
        return self._markdown
