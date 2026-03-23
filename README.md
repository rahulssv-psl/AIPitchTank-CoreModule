# PowerPortAI-Ecosystem

Watsonx + LangGraph core module for solving open-source package porting/build failures on `ppc64le` and `s390x`.

## What is implemented

- `CorePortingAgent` (LangGraph orchestrator) that selects and runs specialized agents.
- `BuildScriptAgent` for searching `ppc64le/build-scripts`.
- `PatchAgent` backed by the reusable `patchagent` module for patch discovery + adaptation.
- `WebDiscoveryAgent` backed by reusable `webdiscoveryagent` module.
- `GitHubIssuesAgent` for issue-history-based fixes.
- Single-agent or full-core execution through one JSON input.

## Input format

```json
{
  "agent_name": "CorePortingAgent",
  "input": {
    "package_name": "vllm",
    "requested_package_version": "0.8.5",
    "github_repo_url": "https://github.com/vllm-project/vllm",
    "error_message": "ModuleNotFoundError: No module named 'triton.compiler'; 'triton' is not a package",
    "architecture_targets": ["ppc64le", "s390x"]
  },
  "options": {
    "max_agents": 4,
    "prefer_local_knowledge": true,
    "include_raw_evidence": true,
    "llm_planning": true
  }
}
```

`agent_name` can also be: `BuildScriptAgent`, `PatchAgent`, `WebDiscoveryAgent`, `GitHubIssuesAgent`.

## Setup (required)

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -r requirements.txt
```

Ensure `.env` has:

```bash
WATSONX_API_KEY=...
WATSONX_PROJECT_ID=...
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

(`PROJECT_ID` is also accepted as fallback.)

## Run end-to-end

```bash
uv run python main.py --input examples/request.json
```

Or inline JSON:

```bash
uv run python main.py --input '{"agent_name":"CorePortingAgent","input":{"package_name":"vllm","requested_package_version":"0.8.5","github_repo_url":"https://github.com/vllm-project/vllm","error_message":"ModuleNotFoundError: No module named triton.compiler","architecture_targets":["ppc64le","s390x"]}}'
```

The output is machine-readable JSON with:

- per-agent results
- selected agent list
- final diagnosis/actions/patch direction
- confidence score

## Patch module (standalone)

You can run patch generation directly:

```bash
uv run python patchagent/patch_automation.py --input '{"package_name":"datadog-agent","requested_package_version":"7.60.1","github_repo_url":"https://github.com/DataDog/datadog-agent","error_message":"patch failed"}'
```

## Web discovery module (standalone API)

Use from Python:

```python
from webdiscoveryagent import WebDiscoveryInput, run_web_discovery
```
