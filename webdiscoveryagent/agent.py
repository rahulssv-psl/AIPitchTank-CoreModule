from langchain_community.tools import DuckDuckGoSearchRun
from state import llm
from utils import create_agent, create_agent_node

web_search_tool = DuckDuckGoSearchRun()
web_agent_prompt = "You are a web research assistant. Your goal is to find solutions for software build errors. Generate a concise search query for sites like Stack Overflow. RESPOND ONLY WITH THE EXACT SEARCH QUERY, no quotes, no explanations."

web_discovery_agent = create_agent(llm, [web_search_tool], web_agent_prompt)
web_discovery_node = create_agent_node(web_discovery_agent, web_search_tool, "Web_Discovery_Agent")
