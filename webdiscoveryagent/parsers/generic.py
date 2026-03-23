from aiohttp import ClientSession
from aiolimiter import AsyncLimiter
from trafilatura import extract

from .base import BaseParser


class GenericParser(BaseParser):
    def __init__(self):
        super().__init__()
        # Rate limiter: 2 requests per second to stay respectful
        self.limiter = AsyncLimiter(2, 1)  # why do I even care about an instance level rate limiter, 1 instance = 1 URL (useless)

    async def from_url(self, url: str, recurse: bool = False):
        async with ClientSession(headers=self.headers) as session:
            async with self.limiter:
                async with session.get(url) as response:
                    if response.status != 200:
                        self.logger.error(f"Failed to fetch {url}: {response.status}")
                        print(f"Failed to fetch {url}: {response.status}")
                        self._bad_request = True
                        self._markdown = f"Failed to fetch {url}: {response.status}"

                    self.content = await response.text()
                    self.logger.info(f"[Generic] Fetched {url}")

    def from_file(self, file_path: str):
        raise NotImplementedError("Not implemented for GenericParser")

    def get_markdown(self) -> str:
        if not self._markdown:
            self._markdown = extract(self.content) or ""
        return self._markdown
