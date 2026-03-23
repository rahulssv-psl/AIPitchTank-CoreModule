from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_ibm import ChatWatsonx
from pydantic import SecretStr


def load_env() -> None:
    load_dotenv()


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def create_watsonx_chat_model() -> ChatWatsonx:
    load_env()
    api_key = _required_env("WATSONX_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID") or os.getenv("PROJECT_ID")
    if not project_id:
        raise RuntimeError("Missing required environment variable: WATSONX_PROJECT_ID (or PROJECT_ID)")
    url = _required_env("WATSONX_URL")

    return ChatWatsonx(
        model_id="meta-llama/llama-3-3-70b-instruct",
        url=SecretStr(url),
        project_id=project_id,
        api_key=SecretStr(api_key),
    )
