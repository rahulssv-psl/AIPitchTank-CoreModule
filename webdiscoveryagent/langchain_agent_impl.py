import asyncio
import os
import re
from sys import stderr

import trafilatura

from ddgs import DDGS
from ddgs.exceptions import DDGSException
from dotenv.main import load_dotenv
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_ibm import ChatWatsonx
from pydantic import SecretStr

from rich import print as rprint

from error import VLLM_ERROR, REACT_ERROR, PYTORCH_ERROR, JSOUP_ERROR, GRADLE_ERRORS, CARGO_ERROR, \
    PODMAN_ERROR  # same error logs that I want the model to search results for
from parsers import StackOverflowParser, GenericParser
from agent_prompts import *

load_dotenv()

model = ChatWatsonx(
    model_id="meta-llama/llama-3-3-70b-instruct",
    # model_id="ibm/granite-8b-code-instruct",
    url=SecretStr(os.getenv("WATSONX_URL")),
    project_id=os.getenv("PROJECT_ID"),
    api_key=SecretStr(os.getenv("WATSONX_API_KEY"))
)


@tool
async def web_search_tool(query: str):
    """
    Search solutions for a specific dev-style query.
    :param query: list of dev style search queries
    :return: list of Markdown of solutions
    """
    results = []
    with DDGS() as se:
        # search_results = list(map(lambda x: x['href'], chain(*[se.text(query=q, max_results=5) for q in queries])))
        try:
            search_results = list(map(lambda x: x['href'], se.text(query=f"site:stackoverflow.com {query}", max_results=5)))
        except DDGSException as ex:
            print(ex, file=stderr)
            results.append("Web Search Failed")
            return results


        for url in search_results:
            # print(url)
            # domain = urlparse(url).netloc  # stackoverflow.com
            if re.match(r"https://stackoverflow.com/questions/\d+/", url) :
                parser = StackOverflowParser()
                await parser.from_url(url)
                results.append(parser.get_markdown())
            else:
                parser = GenericParser()
                await parser.from_url(url)
                results.append(parser.get_markdown())
    return results


tools = [web_search_tool]
prompt = PromptTemplate.from_template(template=META_LLAMA_PROMPT)

async def main():
    agent = create_react_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,
                                   handle_parsing_errors=True,
                                   # fail if no conclusive results in 5 iterations
                                   max_iterations=5, # prevent agent from getting stuck in infinite THOUGHT->ACTION->OBSERVATION loop
                                   early_stopping_method="generate")
    response = await agent_executor.ainvoke({"input": f"How to fix the following error? {VLLM_ERROR}"})

    rprint("\n\n[bold magenta]AGENT's SOLUTION:[/bold magenta]\n")
    print(response["output"])

if __name__ == "__main__":
    asyncio.run(main())
