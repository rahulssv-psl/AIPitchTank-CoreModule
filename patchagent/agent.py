from langchain_community.tools import DuckDuckGoSearchRun
from state import llm
from utils import create_agent, create_agent_node

def patch_search_tool(query: str):
    "Searches ppc64le build-scripts GitHub repo for patches."
    search_query = f"site:github.com/ppc64le/build-scripts in:files language:patch {query}"
    return DuckDuckGoSearchRun().run(search_query)

patch_agent_prompt = "You are a patching specialist. Your goal is to find existing patches to resolve a build failure. Generate a concise search query (maximum 3 words) for patches in the ppc64le build-scripts GitHub repository. RESPOND ONLY WITH THE EXACT SEARCH QUERY, no quotes, no explanations."

patch_agent = create_agent(llm, [patch_search_tool], patch_agent_prompt)
patch_node = create_agent_node(patch_agent, patch_search_tool, "Patch_Agent")
