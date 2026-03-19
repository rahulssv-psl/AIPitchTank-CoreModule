"""
StackOverflow Question Parser
Example URL: https://stackoverflow.com/questions/78607102/how-to-load-a-quantized-fine-tuned-llama-3-8b-model-in-vllm-for-faster-inference
TODO: Haven't added the parsing of "Accepted Answers"
"""

from collections import deque

from aiohttp import ClientSession
from aiolimiter import AsyncLimiter
from bs4 import BeautifulSoup
from bs4.filter import SoupStrainer

from .base import BaseParser


def _get_markdown_from_structure(tree):
    txt = []
    for d in tree:
        if d["type"] == "text":
            txt.append(d["content"] + " ")
        elif d["type"] == "code_block":
            txt.append("\n```\n" + d["content"] + "```\n")
        elif d["type"] == "list":
            if txt and txt[-1] != "\n": txt.append("\n")
            bullet = d["list_type"] == "bullet"
            for i, j in enumerate(d["content"], 1):
                txt.append("* ") if bullet else txt.append(f"{i}. ")
                txt.append(j + "\n")
            txt.append("\n")
        elif d["type"] == "quote":
            # txt.append("\n".join(f"> {line}" for line in friendly_print(d["content"]).split("\n") if line.strip()))
            txt.append("\n".join(
                f"> {line}" for line in _get_markdown_from_structure(d["content"]).split("\n") if line.strip()))
    return "".join(txt).strip()


class StackOverflowParser(BaseParser):
    def __init__(self):
        super().__init__()
        self.content = []
        self._soups = []
        self._parsed_posts = []

        # Rate limiter: 2 requests per second to stay respectful
        self.limiter = AsyncLimiter(2, 1)

        self._pagination_strainer = SoupStrainer("div", class_="s-pagination")

    async def from_url(self, url: str, recurse: bool = False):
        url_queue = deque([url])

        async with ClientSession(headers=self.headers) as session:
            while url_queue:
                current_url = url_queue.popleft()

                async with self.limiter:
                    async with session.get(current_url) as response:
                        if response.status != 200:
                            # Example: https://stackoverflow.com/questions/79783549/podman-failed-to-obtain-configuration (HTTP: 404)
                            self.logger.error(f"Failed to fetch {current_url}: {response.status}")
                            print(f"Failed to fetch {current_url}: {response.status}")
                            self._bad_request = True
                            continue

                        content = await response.text()
                        self.content.append(content)
                        self.logger.info(f"[StackOverflow] Fetched {current_url}")
                        print(f"[StackOverflow] Fetched {current_url}")

                        if recurse:
                            # soup = BeautifulSoup(content, "html.parser", parse_only=self._pagination_strainer)
                            # Strainer Doesn't work for me :(
                            # Use html.parser, lxml wasn't working for stackoverflow
                            soup = BeautifulSoup(content, "html.parser")
                            self._soups.append(soup)  # why parse multiple times
                            next_button = soup.find("a", rel="next")

                            if next_button and next_button.get("href"):
                                next_page = next_button["href"]
                                if not next_page.startswith("http"):
                                    next_page = "https://stackoverflow.com" + next_page

                                url_queue.append(next_page)

    def from_file(self, file_path: str):
        with open(file_path, 'r') as f:
            content = f.read()
            soup = BeautifulSoup(content, "html.parser")
            self.content.append(content)
            self._soups.append(soup)

    def get_markdown(self) -> str:
        if self._markdown:
            return self._markdown
        self._parse()
        return "\n\n".join(post["markdown"] for post in self._parsed_posts)

    def _get_posts(self):
        if not self._soups:
            self._soups = [BeautifulSoup(content, "html.parser") for content in self.content]
        assert len(self._soups) >= 1
        posts = self._soups[0].find_all('div', class_='post-layout')
        for soup in self._soups[1:]:
            posts.extend(soup.find_all('div', class_='post-layout')[1:])  # first is always the question
        return posts

    def _parse(self):
        if self._bad_request:
            self._parsed_posts.append({"markdown": "BAD_REQUEST"})
            return
        question, *posts = self._get_posts() # what to do about the question?
        for post in posts:
            votes = int(post.find("div", class_="js-vote-count").get_text(strip=True))
            body = post.find("div", class_="js-post-body")
            structured = self._parse_stack_content(body)
            edited, created = None, None
            timestamps = post.find_all("div", class_="user-action-time")
            if len(timestamps) == 2:
                edited, created = map(lambda t: t.find("span").get_text(strip=True), timestamps)
            else:
                edited, created = None, timestamps[0].find("span").get_text(strip=True)
            self._parsed_posts.append({
                "votes": votes,
                "structured": structured,
                "markdown": _get_markdown_from_structure(structured),
                "edited": edited,
                "created": created
            })

    def _parse_stack_content(self, soup_obj):
        structured_data = []
        for element in soup_obj.find_all(recursive=False):
            if element.name == 'p':
                text_parts = []
                for child in element.children:
                    if child.name == 'code':
                        text_parts.append(f"`{child.get_text().strip()}`")
                    elif child.name == 'a':
                        text_parts.append(f"[{child.get_text().strip()}]({child.get('href')})")
                    elif child.name == 'strong':
                        text_parts.append(f"### {child.get_text().strip()}")
                    # elif child.name == 'blockquote':
                    #     text_parts.append(f"> {child.get_text().strip()}")
                    else:
                        text_parts.append(child.get_text())
                structured_data.append({
                    "type": "text",
                    "content": "\n" + "".join(text_parts).strip() + "\n"
                    ## <p> tags will create a new paragraph everytime
                })
            elif element.name == 'pre':
                code_tag = element.find('code')
                structured_data.append({
                    "type": "code_block",
                    "language": element.get('class', [''])[0].replace('lang-', '') if element.has_attr(
                        'class') else "none",
                    "content": code_tag.get_text() if code_tag else element.get_text()
                })
            elif element.name in ['ul', 'ol']:
                items = [li.get_text(separator=" ", strip=True) for li in element.find_all('li')]
                structured_data.append({
                    "type": "list",
                    "list_type": "bullet" if element.name == 'ul' else "numbered",
                    "content": items
                })
            elif element.name == "blockquote":
                structured_data.append({
                    "type": "quote",
                    "content": self._parse_stack_content(element)
                })
        return structured_data
