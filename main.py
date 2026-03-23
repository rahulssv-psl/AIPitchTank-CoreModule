from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from porting_agents.dispatcher import AgentDispatcher
from porting_agents.models import CoreRequest


def _load_input(input_arg: str) -> dict[str, Any]:
    path = Path(input_arg)
    if path.exists():
        return json.loads(path.read_text())
    return json.loads(input_arg)


async def _run(payload: dict[str, Any]) -> dict[str, Any]:
    request = CoreRequest.model_validate(payload)
    dispatcher = AgentDispatcher()
    response = await dispatcher.run(request)
    return response.model_dump()


def main() -> None:
    parser = argparse.ArgumentParser(description="Watsonx + LangGraph core porting agent")
    parser.add_argument(
        "--input",
        required=True,
        help="JSON payload or file path containing request payload",
    )
    args = parser.parse_args()

    try:
        payload = _load_input(args.input)
        result = asyncio.run(_run(payload))
        print(json.dumps(result, indent=2))
    except (json.JSONDecodeError, ValidationError) as ex:
        print(
            json.dumps(
                {
                    "agent_name": "CorePortingAgent",
                    "status": "error",
                    "summary": f"Invalid input payload: {ex}",
                },
                indent=2,
            )
        )
        raise SystemExit(1) from ex
    except Exception as ex:  # pragma: no cover
        print(
            json.dumps(
                {
                    "agent_name": "CorePortingAgent",
                    "status": "error",
                    "summary": f"Execution failed: {ex}",
                },
                indent=2,
            )
        )
        raise SystemExit(2) from ex


if __name__ == "__main__":
    main()
