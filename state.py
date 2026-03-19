import os
import operator
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_ibm import WatsonxLLM

# Load environment variables
load_dotenv()

# Shared LLM Instance
llm = WatsonxLLM(
    model_id="meta-llama/llama-3-3-70b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    project_id=os.getenv("WATSONX_AI_PROJECT_ID"),
    params={
        "decoding_method": "greedy", 
        "max_new_tokens": 1024, 
        "stop_sequences": ["\n\n"]
    },
)

class AgentState(TypedDict):
    task: str
    agent_outcomes: Annotated[list[str], operator.add]
    next: str
