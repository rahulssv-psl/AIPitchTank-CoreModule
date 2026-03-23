## Web Discovery Agent Module

`webdiscoveryagent` is now a reusable Python module consumed by the core `WebDiscoveryAgent`.

## Reusable API

```python
from webdiscoveryagent import WebDiscoveryInput, run_web_discovery
```

```python
result = await run_web_discovery(
    WebDiscoveryInput(
        package_name="vllm",
        requested_package_version="0.8.5",
        architecture_targets=["ppc64le", "s390x"],
        error_message="ModuleNotFoundError: No module named 'triton.compiler'; 'triton' is not a package",
    )
)
```

`result` contains:

- `status`
- `summary`
- `query`
- `references`
- `confidence`

## Legacy script

The existing `langchain_agent_impl.py` still works as before for standalone experimentation.
