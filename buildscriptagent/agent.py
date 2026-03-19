from langchain_community.tools import DuckDuckGoSearchRun
from state import llm
from utils import create_agent, create_agent_node

def build_script_search_tool(query: str):
    "Searches ppc64le build-scripts GitHub repo for build scripts."
    search_query = f"site:github.com/ppc64le/build-scripts {query}"
    return DuckDuckGoSearchRun().run(search_query)

build_agent_prompt = "You are an expert build engineer. Your goal is to find similar build scripts to solve the current problem. Generate a concise search query (maximum 3 words) for the ppc64le build-scripts GitHub repository. RESPOND ONLY WITH THE EXACT SEARCH QUERY, no quotes, no explanations."

build_script_agent = create_agent(llm, [build_script_search_tool], build_agent_prompt)
build_script_node = create_agent_node(build_script_agent, build_script_search_tool, "Build-script_Agent")
