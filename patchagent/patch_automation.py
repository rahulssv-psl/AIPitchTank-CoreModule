from __future__ import annotations

import argparse
import base64
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ibm import ChatWatsonx
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, Field, HttpUrl, SecretStr, ValidationError, field_validator


BUILD_SCRIPTS_REPO = "ppc64le/build-scripts"
DEFAULT_TIMEOUT_SECONDS = 30


class PatchAutomationError(RuntimeError):
    """Raised when patch automation cannot complete required steps."""


class PatchAutomationInput(BaseModel):
    package_name: str = Field(min_length=1)
    requested_package_version: str = Field(min_length=1)
    github_repo_url: HttpUrl
    error_message: str = ""

    @field_validator("package_name", "requested_package_version", "error_message")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()


class PatchAutomationResult(BaseModel):
    status: str
    summary: str
    patch_diff: str
    commit_message: str
    steps: list[str]
    evidence: dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)


def _normalize_confidence(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    number = max(0.0, min(1.0, number))
    return round(number, 3)


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _parse_json_from_text(text: str) -> dict[str, Any]:
    candidate = _strip_code_fences(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", candidate)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _truncate_text(value: str, limit: int) -> str:
    text = value.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


class PatchAutomationEngine:
    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        model: ChatWatsonx | None = None,
        max_patch_files: int = 4,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        load_dotenv()
        self.max_patch_files = max_patch_files
        self.timeout_seconds = timeout_seconds
        self.session = session or self._create_session()
        self.model = model or self._create_model()

    @staticmethod
    def _create_session() -> requests.Session:
        session = requests.Session()
        session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "User-Agent": "AIPitchTank-CoreModule-PatchAgent",
            }
        )
        github_token = os.getenv("GITHUB_TOKEN", "").strip()
        if github_token:
            session.headers["Authorization"] = f"Bearer {github_token}"
        return session

    @staticmethod
    def _create_model() -> ChatWatsonx:
        url = os.getenv("WATSONX_URL")
        api_key = os.getenv("WATSONX_API_KEY")
        project_id = os.getenv("WATSONX_PROJECT_ID") or os.getenv("PROJECT_ID")

        if not url or not api_key or not project_id:
            raise PatchAutomationError(
                "Missing Watsonx configuration. Required: WATSONX_URL, WATSONX_API_KEY, "
                "WATSONX_PROJECT_ID (or PROJECT_ID)."
            )

        return ChatWatsonx(
            model_id="meta-llama/llama-3-3-70b-instruct",
            url=SecretStr(url),
            project_id=project_id,
            api_key=SecretStr(api_key),
            params={"temperature": 0, "max_new_tokens": 2048},
        )

    def _request_json(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self.session.get(url, params=params, timeout=self.timeout_seconds)
        if response.status_code >= 400:
            raise PatchAutomationError(f"GitHub API request failed ({response.status_code}) for URL: {url}")
        try:
            return response.json()
        except ValueError as exc:
            raise PatchAutomationError(f"GitHub API returned invalid JSON for URL: {url}") from exc

    def _fetch_tree(self) -> tuple[list[dict[str, Any]], str]:
        errors: list[str] = []
        for ref in ("master", "main"):
            api_url = f"https://api.github.com/repos/{BUILD_SCRIPTS_REPO}/git/trees/{ref}"
            try:
                payload = self._request_json(api_url, params={"recursive": 1})
            except PatchAutomationError as exc:
                errors.append(str(exc))
                continue

            tree = payload.get("tree")
            if isinstance(tree, list):
                return tree, ref
            errors.append(f"Unexpected tree payload for ref {ref}")

        raise PatchAutomationError("Failed to fetch build-scripts tree. " + " | ".join(errors))

    def _get_file_content(self, path: str, ref: str) -> str:
        if not path:
            raise PatchAutomationError("Path must be provided when fetching file content.")

        api_url = f"https://api.github.com/repos/{BUILD_SCRIPTS_REPO}/contents/{path}"
        payload = self._request_json(api_url, params={"ref": ref})
        encoded_content = payload.get("content")
        if not encoded_content:
            raise PatchAutomationError(f"No content available for {path}")

        encoding = payload.get("encoding", "")
        if encoding != "base64":
            raise PatchAutomationError(f"Unsupported encoding '{encoding}' for {path}")

        try:
            return base64.b64decode(encoded_content).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise PatchAutomationError(f"Failed to decode content for {path}") from exc

    @staticmethod
    def _extract_repo_owner_and_name(repo_url: str) -> tuple[str, str]:
        parsed = urlparse(repo_url)
        if parsed.netloc != "github.com":
            raise PatchAutomationError("Only GitHub repository URLs are supported.")
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            raise PatchAutomationError(f"Invalid GitHub repository URL: {repo_url}")
        owner = parts[0]
        repo = parts[1].removesuffix(".git")
        return owner, repo

    @staticmethod
    def _extract_version_token(path: str) -> str | None:
        matches = re.findall(r"\d+(?:\.\d+){1,3}", path)
        if not matches:
            return None
        return matches[-1]

    @staticmethod
    def _to_version(raw: str | None) -> Version | None:
        if not raw:
            return None
        candidate = raw.strip().lstrip("vV")
        try:
            return Version(candidate)
        except InvalidVersion:
            match = re.search(r"\d+(?:\.\d+)+", candidate)
            if not match:
                return None
            try:
                return Version(match.group(0))
            except InvalidVersion:
                return None

    def _select_patch_path(self, patch_paths: list[str], requested_version: str) -> str:
        if not patch_paths:
            raise PatchAutomationError("No patch files found for package in build-scripts repository.")

        for path in patch_paths:
            if requested_version in path:
                return path

        target_version = self._to_version(requested_version)
        if target_version:
            parsed: list[tuple[Version, str]] = []
            for path in patch_paths:
                parsed_version = self._to_version(self._extract_version_token(path))
                if parsed_version:
                    parsed.append((parsed_version, path))

            if parsed:
                older_or_equal = [item for item in parsed if item[0] <= target_version]
                if older_or_equal:
                    return max(older_or_equal, key=lambda item: item[0])[1]
                return min(parsed, key=lambda item: item[0])[1]

        return sorted(patch_paths)[0]

    @staticmethod
    def _split_patch_chunks(patch_content: str) -> list[str]:
        chunks = [chunk.strip() for chunk in re.split(r"(?m)(?=^diff --git )", patch_content) if chunk.strip()]
        return [chunk + "\n" for chunk in chunks]

    @staticmethod
    def _extract_file_path_from_chunk(chunk: str) -> str | None:
        match = re.search(r"^diff --git a/(.*?) b/.*$", chunk, re.MULTILINE)
        if not match:
            return None
        return match.group(1).strip()

    def _fetch_target_file_content(
        self,
        *,
        owner: str,
        repo: str,
        file_path: str,
        requested_version: str,
    ) -> tuple[str, str]:
        refs = list(
            dict.fromkeys(
                [
                    requested_version,
                    f"v{requested_version}",
                    "main",
                    "master",
                ]
            )
        )

        for ref in refs:
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{file_path}"
            response = self.session.get(raw_url, timeout=self.timeout_seconds)
            if response.status_code == 200:
                return response.text.replace("\r\n", "\n"), ref
            if response.status_code in (404, 400):
                continue
            raise PatchAutomationError(
                f"Failed to fetch target file {file_path} from ref {ref}: status={response.status_code}"
            )

        raise PatchAutomationError(
            f"Target file {file_path} not found in requested refs for {owner}/{repo} ({requested_version})."
        )

    def _adapt_patch_chunk(
        self,
        *,
        package_name: str,
        requested_version: str,
        file_path: str,
        old_patch_chunk: str,
        target_file_content: str,
        error_message: str,
    ) -> tuple[str, str, float]:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert patch-porting engineer.\n"
                    "Return strict JSON only with keys: patch_diff, rationale, confidence.\n"
                    "Rules:\n"
                    "- patch_diff must be a valid unified diff for the same file path.\n"
                    "- Preserve the intent of the old patch, only adapting where needed.\n"
                    "- Keep ppc64le/s390x compatibility in mind.\n"
                    "- Do not include markdown fences.",
                ),
                (
                    "human",
                    "Package: {package_name}\n"
                    "Requested version: {requested_version}\n"
                    "File path: {file_path}\n"
                    "Build error: {error_message}\n\n"
                    "Target file content (possibly truncated):\n{target_file_content}\n\n"
                    "Old patch chunk:\n{old_patch_chunk}\n",
                ),
            ]
        )

        chain = prompt | self.model
        response = chain.invoke(
            {
                "package_name": package_name,
                "requested_version": requested_version,
                "file_path": file_path,
                "error_message": error_message,
                "target_file_content": _truncate_text(target_file_content, 18000),
                "old_patch_chunk": _truncate_text(old_patch_chunk, 12000),
            }
        )
        raw_text = response.content if hasattr(response, "content") else str(response)
        payload = _parse_json_from_text(raw_text)

        patch_diff = str(payload.get("patch_diff", "")).strip()
        rationale = str(payload.get("rationale", "")).strip()
        confidence = _normalize_confidence(payload.get("confidence"), default=0.5)

        if not patch_diff:
            raw_diff = _strip_code_fences(raw_text)
            if "diff --git" in raw_diff and "@@" in raw_diff:
                patch_diff = raw_diff.strip()
                rationale = rationale or "Model returned raw diff text."
                confidence = max(0.4, confidence)
            else:
                patch_diff = old_patch_chunk.strip()
                rationale = (
                    rationale
                    or "Unable to confidently adapt patch chunk automatically. Returning closest known chunk."
                )
                confidence = min(confidence, 0.35)

        return patch_diff + "\n", rationale, confidence

    def generate(self, request_input: PatchAutomationInput) -> PatchAutomationResult:
        tree, ref = self._fetch_tree()
        package = request_input.package_name.lower()
        relevant_paths = [
            item["path"]
            for item in tree
            if item.get("type") == "blob"
            and isinstance(item.get("path"), str)
            and package in item["path"].lower()
            and (item["path"].endswith(".patch") or item["path"].endswith("build_info.json"))
        ]

        patch_paths = [path for path in relevant_paths if path.endswith(".patch")]
        selected_patch_path = self._select_patch_path(patch_paths, request_input.requested_package_version)
        patch_text = self._get_file_content(selected_patch_path, ref)
        patch_chunks = self._split_patch_chunks(patch_text)
        selected_chunks = patch_chunks[: self.max_patch_files]

        owner, repo = self._extract_repo_owner_and_name(str(request_input.github_repo_url))
        processed: list[str] = []
        rationales: list[str] = []
        chunk_confidences: list[float] = []
        skipped_files: list[dict[str, str]] = []
        processed_files: list[dict[str, str]] = []

        for chunk in selected_chunks:
            file_path = self._extract_file_path_from_chunk(chunk)
            if not file_path:
                skipped_files.append({"file_path": "unknown", "reason": "Missing diff file header"})
                continue

            try:
                target_content, source_ref = self._fetch_target_file_content(
                    owner=owner,
                    repo=repo,
                    file_path=file_path,
                    requested_version=request_input.requested_package_version,
                )
            except PatchAutomationError as exc:
                skipped_files.append({"file_path": file_path, "reason": str(exc)})
                continue

            patch_diff, rationale, confidence = self._adapt_patch_chunk(
                package_name=request_input.package_name,
                requested_version=request_input.requested_package_version,
                file_path=file_path,
                old_patch_chunk=chunk,
                target_file_content=target_content,
                error_message=request_input.error_message,
            )
            processed.append(patch_diff)
            rationales.append(f"{file_path}: {rationale}")
            chunk_confidences.append(confidence)
            processed_files.append({"file_path": file_path, "source_ref": source_ref})

        if not processed:
            raise PatchAutomationError(
                "Patch automation did not produce any patch chunks. "
                "Either source patch could not be mapped or target files were unavailable."
            )

        average_chunk_confidence = sum(chunk_confidences) / len(chunk_confidences)
        completion_ratio = len(processed) / max(1, len(selected_chunks))
        overall_confidence = _normalize_confidence(0.7 * average_chunk_confidence + 0.3 * completion_ratio, 0.5)
        status = "success" if not skipped_files else "partial"
        summary = (
            "Generated patch updates for target version."
            if status == "success"
            else "Generated partial patch updates; some chunks require manual follow-up."
        )

        commit_message = (
            f"Port {request_input.package_name} patches to {request_input.requested_package_version} "
            "for ppc64le/s390x compatibility"
        )
        steps = [
            "Save patch diff to a .patch file in your package/build workspace.",
            "Run `git apply --check <patch-file>` to verify patch applicability.",
            "Apply patch and rebuild on ppc64le and s390x.",
            "If skipped files exist, adjust those hunks manually using evidence paths.",
        ]

        return PatchAutomationResult(
            status=status,
            summary=summary,
            patch_diff="\n".join(processed).strip() + "\n",
            commit_message=commit_message,
            steps=steps,
            evidence={
                "build_scripts_patch_path": selected_patch_path,
                "processed_files": processed_files,
                "skipped_files": skipped_files,
                "rationales": rationales,
            },
            confidence=overall_confidence,
        )


def _load_input(input_value: str) -> PatchAutomationInput:
    input_path = Path(input_value)
    if input_path.exists():
        payload = json.loads(input_path.read_text())
    else:
        payload = json.loads(input_value)
    return PatchAutomationInput.model_validate(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch automation engine for build-script patch porting")
    parser.add_argument("--input", required=True, help="JSON payload or path to JSON file")
    args = parser.parse_args()

    try:
        request_input = _load_input(args.input)
        result = PatchAutomationEngine().generate(request_input)
        print(result.model_dump_json(indent=2))
    except (json.JSONDecodeError, ValidationError, PatchAutomationError) as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "summary": f"Patch automation failed: {exc}",
                },
                indent=2,
            )
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
