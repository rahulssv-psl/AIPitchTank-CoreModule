from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from state import AgentState, llm
from webdiscoveryagent.agent import web_discovery_node
from buildscriptagent.agent import build_script_node
from patchagent.agent import patch_node
from githubissuesagent.agent import github_issues_node

# --- 1. Supervisor Setup ---
members = ["Build-script_Agent", "Patch_Agent", "Web_Discovery_Agent", "GitHub_issues_Agent"]
supervisor_prompt_template = f"""You are a supervisor managing a team of agents: {", ".join(members)}.
Given the user's request and the conversation history, decide which agent should act next, or if the task is complete.
Once you have enough information or have called at least 3 agents, you MUST respond ONLY with "FINISH".

User Request: {{task}}
Conversation History:
{{agent_outcomes}}

Select the next agent from [{", ".join(members)}] or respond with FINISH. ONLY return the name or FINISH.
"""

supervisor_prompt = ChatPromptTemplate.from_messages([("system", supervisor_prompt_template)])
supervisor_chain = supervisor_prompt | llm | StrOutputParser()

def supervisor_node(state: AgentState):
    print("---SUPERVISOR---")
    if not state.get("agent_outcomes"):
         next_agent = "Web_Discovery_Agent"
    elif len(state.get("agent_outcomes", [])) >= 4:
         next_agent = "FINISH"
    else:
        next_agent_raw = supervisor_chain.invoke(state)
        next_agent = next_agent_raw.strip()
        for member in members + ["FINISH"]:
            if member in next_agent:
                next_agent = member
                break
        else:
            next_agent = "FINISH"

    print(f"Supervisor decision: {next_agent}")
    return {"next": next_agent}

# --- 2. Final Summarizer Node ---
def summarizer_node(state: AgentState):
    print("---GENERATING FINAL SUMMARY---")
    combined_outcomes = "\n\n".join(state["agent_outcomes"])
    
    summary_prompt = f"""You are an expert software engineer. Based on the following research results from different agents, 
    provide a concise, synthesized summary of the findings and a recommended solution for the user's build error.
    
    Research Findings:
    {combined_outcomes}
    
    Synthesized Summary and Recommendation:"""
    
    summary = llm.invoke(summary_prompt)
    print("\n--- FINAL SUMMARY ---")
    print(summary)
    return {"agent_outcomes": [f"---FINAL SYNTHESIZED SUMMARY---\n{summary}"]}

# --- 3. Graph Definition ---
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("summarizer", summarizer_node)
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
        "FINISH": "summarizer",
    },
)

workflow.add_edge("Build-script_Agent", "supervisor")
workflow.add_edge("Patch_Agent", "supervisor")
workflow.add_edge("Web_Discovery_Agent", "supervisor")
workflow.add_edge("GitHub_issues_Agent", "supervisor")
workflow.add_edge("summarizer", END)

graph = workflow.compile()

# --- 4. Execution ---
if __name__ == "__main__":
    build_error = "error: unrecognized command line option ‘-mrecord-mcount’ when building on ppc64le"
    print(f"Starting agent workflow for build error: '{build_error}'\n")

    final_state = None
    for event in graph.stream({"task": build_error}, {"recursion_limit": 25}):
        for key, value in event.items():
            if key != "summarizer":
                print(f"--- Output from: {key} ---")
                print(value)
                print("\n")
        final_state = event

    print("--- Agent Workflow Finished ---")
