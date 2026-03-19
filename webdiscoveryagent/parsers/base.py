from abc import ABC, abstractmethod
from logging import getLogger


class BaseParser(ABC):
    def __init__(self):
        self.content = []
        self._bad_request = False
        self._markdown = ""
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        self.logger = getLogger(self.__class__.__name__)

    @abstractmethod
    async def from_url(self, url: str, recurse: bool = False):
        """Fetch content from a URL with rate limiting."""
        pass

    @abstractmethod
    def from_file(self, file_path: str):
        """Load content from a local HTML file."""
        pass

    @abstractmethod
    def get_markdown(self) -> str:
        """Convert stored content to Markdown."""
        pass