from langchain_community.tools import DuckDuckGoSearchRun
from state import llm
from utils import create_agent, create_agent_node

def github_issues_search_tool(query: str):
    "Searches GitHub issues for relevant solutions."
    search_query = f"site:github.com inurl:issues {query}"
    return DuckDuckGoSearchRun().run(search_query)

github_issues_agent_prompt = "You are a GitHub expert. Your goal is to find solutions in GitHub issues. Generate a concise search query (maximum 3 words) to find relevant GitHub issues. RESPOND ONLY WITH THE EXACT SEARCH QUERY, no quotes, no explanations."

github_issues_agent = create_agent(llm, [github_issues_search_tool], github_issues_agent_prompt)
github_issues_node = create_agent_node(github_issues_agent, github_issues_search_tool, "GitHub_issues_Agent")
