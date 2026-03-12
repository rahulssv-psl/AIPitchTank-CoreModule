import os
import operator
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage

from langchain_ibm import WatsonxLLM
from langgraph.graph import StateGraph, END
from langchain_community.tools import DuckDuckGoSearchRun

# Load environment variables
load_dotenv()

# --- 1. Define Tools ---
web_search_tool = DuckDuckGoSearchRun()

def build_script_search_tool(query: str):
    "Searches ppc64le build-scripts GitHub repo for build scripts."
    search_query = f"site:github.com/ppc64le/build-scripts {query}"
    return DuckDuckGoSearchRun().run(search_query)

def patch_search_tool(query: str):
    "Searches ppc64le build-scripts GitHub repo for patches."
    search_query = f"site:github.com/ppc64le/build-scripts in:files language:patch {query}"
    return DuckDuckGoSearchRun().run(search_query)

def github_issues_search_tool(query: str):
    "Searches GitHub issues for relevant solutions."
    search_query = f"site:github.com inurl:issues {query}"
    return DuckDuckGoSearchRun().run(search_query)

# --- 2. Define State ---
class AgentState(TypedDict):
    task: str
    agent_outcomes: Annotated[list[str], operator.add]
    next: str

# --- 3. Define Agents and Supervisor ---
llm = WatsonxLLM(
    model_id="meta-llama/llama-3-3-70b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    project_id=os.getenv("WATSONX_AI_PROJECT_ID"),
    params={"decoding_method": "greedy", "max_new_tokens": 1024, "stop_sequences": ["\\n\\n"]},
)

def create_agent(llm, tools, system_prompt):
    """Factory to create a new agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    agent = prompt | llm | StrOutputParser()
    return agent

def create_agent_node(agent, tool, agent_name):
    """Factory to create a node that executes an agent."""
    def agent_node(state: AgentState):
        print(f"---EXECUTING {agent_name}---")
        query = agent.invoke({"input": state["task"]})
        if hasattr(tool, "run"):
            tool_result = tool.run(query)
        else:
            tool_result = tool(query)
        outcome = (
            f"---<FROM {agent_name}>---\\n"
            f"Agent generated search query: {query}\\n"
            f"Tool Result:\\n{tool_result}\\n"
            f"---</FROM {agent_name}>---"
        )
        return {"agent_outcomes": [outcome]}
    return agent_node

# Agent Prompts
build_agent_prompt = "You are an expert build engineer. Your goal is to find similar build scripts to solve the current problem. Generate a search query for the ppc64le build-scripts GitHub repository. RESPOND ONLY WITH THE EXACT SEARCH QUERY, no quotes, no explanations."
patch_agent_prompt = "You are a patching specialist. Your goal is to find existing patches to resolve a build failure. Generate a search query for patches in the ppc64le build-scripts GitHub repository. RESPOND ONLY WITH THE EXACT SEARCH QUERY, no quotes, no explanations."
web_agent_prompt = "You are a web research assistant. Your goal is to find solutions for software build errors. Generate a concise search query for sites like Stack Overflow. RESPOND ONLY WITH THE EXACT SEARCH QUERY, no quotes, no explanations."
github_issues_agent_prompt = "You are a GitHub expert. Your goal is to find solutions in GitHub issues. Generate a search query to find relevant GitHub issues. RESPOND ONLY WITH THE EXACT SEARCH QUERY, no quotes, no explanations."

# Create Agents
build_script_agent = create_agent(llm, [build_script_search_tool], build_agent_prompt)
patch_agent = create_agent(llm, [patch_search_tool], patch_agent_prompt)
web_discovery_agent = create_agent(llm, [web_search_tool], web_agent_prompt)
github_issues_agent = create_agent(llm, [github_issues_search_tool], github_issues_agent_prompt)

# Create Agent Nodes
build_script_node = create_agent_node(build_script_agent, build_script_search_tool, "Build-script_Agent")
patch_node = create_agent_node(patch_agent, patch_search_tool, "Patch_Agent")
web_discovery_node = create_agent_node(web_discovery_agent, web_search_tool, "Web_Discovery_Agent")
github_issues_node = create_agent_node(github_issues_agent, github_issues_search_tool, "GitHub_issues_Agent")

# Supervisor (Core Agent)
members = ["Build-script_Agent", "Patch_Agent", "Web_Discovery_Agent", "GitHub_issues_Agent"]
supervisor_prompt_template = f"""You are a supervisor managing a team of agents: {", ".join(members)}.
Given the user's request and the conversation history, decide which agent should act next, or if the task is complete.
Once you have enough information or have called at least 3 agents, you MUST respond ONLY with "FINISH".

User Request: {{task}}
Conversation History:
{{agent_outcomes}}

Select the next agent from [{", ".join(members)}] or respond with FINISH. ONLY return the name or FINISH.
"""

supervisor_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", supervisor_prompt_template),
    ]
)
supervisor_chain = supervisor_prompt | llm | StrOutputParser()

def supervisor_node(state: AgentState):
    print("---SUPERVISOR---")
    # If it's the first turn, start with the web discovery agent
    if not state.get("agent_outcomes"):
         next_agent = "Web_Discovery_Agent"
    elif len(state.get("agent_outcomes", [])) >= 4:
         next_agent = "FINISH"
    else:
        next_agent_raw = supervisor_chain.invoke(state)
        # Clean up supervisor output
        next_agent = next_agent_raw.strip()
        for member in members + ["FINISH"]:
            if member in next_agent:
                next_agent = member
                break
        else: # Default to finish if supervisor output is unclear
            next_agent = "FINISH"

    print(f"Supervisor decision: {next_agent}")
    return {"next": next_agent}

# --- 4. Define Graph ---
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("Build-script_Agent", build_script_node)
workflow.add_node("Patch_Agent", patch_node)
workflow.add_node("Web_Discovery_Agent", web_discovery_node)
workflow.add_node("GitHub_issues_Agent", github_issues_node)

# Add Edges
workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {
        "Build-script_Agent": "Build-script_Agent",
        "Patch_Agent": "Patch_Agent",
        "Web_Discovery_Agent": "Web_Discovery_Agent",
        "GitHub_issues_Agent": "GitHub_issues_Agent",
        "FINISH": END,
    },
)

workflow.add_edge("Build-script_Agent", "supervisor")
workflow.add_edge("Patch_Agent", "supervisor")
workflow.add_edge("Web_Discovery_Agent", "supervisor")
workflow.add_edge("GitHub_issues_Agent", "supervisor")

# Compile
graph = workflow.compile()


# --- 5. Run the Graph ---
if __name__ == "__main__":
    build_error = "error: unrecognized command line option ‘-mrecord-mcount’ when building on ppc64le"

    print(f"Starting agent workflow for build error: '{build_error}'\\n")

    # The stream method returns an iterator of state updates
    final_state = None
    for event in graph.stream({"task": build_error}, {"recursion_limit": 25}):
        # 'event' is a dictionary with the name of the node that just ran and its output
        for key, value in event.items():
            print(f"--- Output from: {key} ---")
            print(value)
            print("\\n")
        final_state = event

    print("--- Agent Workflow Finished ---")
    if final_state:
        last_ran_agent = list(final_state.keys())[0]
        final_outcomes = final_state[last_ran_agent].get('agent_outcomes', [])
        print("--- Final Results ---")
        for outcome in final_outcomes:
            print(outcome)
