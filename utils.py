from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from state import AgentState

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
            f"---<FROM {agent_name}>---\n"
            f"Agent generated search query: {query}\n"
            f"Tool Result:\n{tool_result}\n"
            f"---</FROM {agent_name}>---"
        )
        return {"agent_outcomes": [outcome]}
    return agent_node
