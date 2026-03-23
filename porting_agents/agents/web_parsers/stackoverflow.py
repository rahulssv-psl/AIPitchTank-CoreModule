from __future__ import annotations

from aiohttp import ClientSession
from aiolimiter import AsyncLimiter
from bs4 import BeautifulSoup

from .base import BaseParser


class StackOverflowParser(BaseParser):
    def __init__(self) -> None:
        super().__init__()
        self.limiter = AsyncLimiter(2, 1)
        self._soups: list[BeautifulSoup] = []

    async def from_url(self, url: str, recurse: bool = False) -> None:
        async with ClientSession(headers=self.headers) as session:
            async with self.limiter:
                async with session.get(url) as response:
                    if response.status != 200:
                        self._bad_request = True
                        self._markdown = f"Failed to fetch {url}: {response.status}"
                        return
                    html = await response.text()
                    self.content = [html]
                    self._soups = [BeautifulSoup(html, "html.parser")]

    def get_markdown(self) -> str:
        if self._markdown:
            return self._markdown
        if self._bad_request or not self._soups:
            return self._markdown

        posts = self._soups[0].find_all("div", class_="js-post-body")
        text_blocks: list[str] = []
        for post in posts:
            text_blocks.append(post.get_text(separator="\n", strip=True))
        self._markdown = "\n\n".join(text_blocks)
        return self._markdown
