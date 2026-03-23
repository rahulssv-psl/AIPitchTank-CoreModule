## Patch Agent Module

`patchagent` is now a reusable Python module used by the core `PatchAgent`.

It ports existing patch files from `ppc64le/build-scripts` to a requested package version by:

- locating related patch files in build-scripts
- selecting the best source patch for version context
- mapping chunks to target repo files
- using Watsonx to adapt patch hunks safely
- returning machine-readable patch output and evidence

## CLI usage

Run from repository root:

```bash
uv run python patchagent/patch_automation.py --input '{"package_name":"datadog-agent","requested_package_version":"7.60.1","github_repo_url":"https://github.com/DataDog/datadog-agent","error_message":"patch failed for ppc64le"}'
```

Or with JSON file:

```bash
uv run python patchagent/patch_automation.py --input examples/request.json
```

## Output

The script returns JSON containing:

- `status` (`success`/`partial`/`error`)
- `patch_diff`
- `commit_message`
- `steps`
- `evidence`
- `confidence`
