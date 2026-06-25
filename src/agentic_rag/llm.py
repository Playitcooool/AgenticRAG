"""Minimal LLM clients — OpenAI-compatible and Anthropic Messages API."""

from __future__ import annotations

import json
from http.client import IncompleteRead
from dataclasses import dataclass
from typing import Any, Protocol
from urllib import error, request


class LLMClientError(RuntimeError):
    """Raised when the LLM server cannot serve a request."""


class LLMClient(Protocol):
    """Duck-typed protocol both clients satisfy."""

    def chat(self, system: str, user: str) -> str: ...
    def chat_json(self, system: str, user: str) -> dict[str, Any]: ...


def _base_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _send_request(http_request: request.Request, timeout: float, label: str) -> dict[str, Any]:
    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (error.URLError, IncompleteRead) as exc:
        raise LLMClientError(f"{label}: {exc}") from exc


@dataclass
class OpenAICompatibleClient:
    """Small client for `/v1/chat/completions` and `/v1/models`."""

    base_url: str = "http://localhost:1234"
    model: str | None = None
    api_key: str = "no_need"
    timeout: float = 60.0
    temperature: float = 0.0

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")

    def resolve_model(self) -> str:
        if self.model:
            return self.model
        models = self.list_models()
        if not models:
            raise LLMClientError(f"No models returned by {self.base_url}/v1/models")
        self.model = models[0]
        return self.model

    def list_models(self) -> list[str]:
        payload = self._get("/v1/models")
        return [item["id"] for item in payload.get("data", []) if "id" in item]

    def chat_json(self, system: str, user: str) -> dict[str, Any]:
        content = self.chat(system=system, user=user)
        try:
            return json.loads(_extract_json_object(content))
        except json.JSONDecodeError as exc:
            raise LLMClientError(f"Model did not return valid JSON: {content}") from exc

    def chat(self, system: str, user: str, response_format: dict[str, str] | None = None) -> str:
        body: dict[str, Any] = {
            "model": self.resolve_model(),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
        }
        if response_format:
            body["response_format"] = response_format
        payload = self._post("/v1/chat/completions", body)
        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError(f"Unexpected chat response: {payload}") from exc

    def _get(self, path: str) -> dict[str, Any]:
        return _send_request(
            request.Request(f"{self.base_url}{path}", headers=_base_headers(self.api_key), method="GET"),
            self.timeout,
            f"Cannot reach OpenAI-compatible server at {self.base_url}",
        )

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8")
        return _send_request(
            request.Request(
                f"{self.base_url}{path}",
                data=data,
                headers={**_base_headers(self.api_key), "Content-Type": "application/json"},
                method="POST",
            ),
            self.timeout,
            f"Cannot reach OpenAI-compatible server at {self.base_url}",
        )


@dataclass
class AnthropicMessagesClient:
    """Client for the Anthropic Messages API (`POST /v1/messages`)."""

    base_url: str = "https://api.anthropic.com"
    model: str | None = None
    api_key: str = ""
    timeout: float = 60.0
    temperature: float = 0.0
    max_tokens: int = 4096

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")

    def resolve_model(self) -> str:
        if self.model:
            return self.model
        models = self.list_models()
        if not models:
            raise LLMClientError(f"No models returned by {self.base_url}/v1/models")
        self.model = models[0]
        return self.model

    def list_models(self) -> list[str]:
        payload = self._get("/v1/models")
        return [item["id"] for item in payload.get("data", []) if "id" in item]

    def chat_json(self, system: str, user: str) -> dict[str, Any]:
        content = self.chat(system=system, user=user)
        try:
            return json.loads(_extract_json_object(content))
        except json.JSONDecodeError as exc:
            raise LLMClientError(f"Model did not return valid JSON: {content}") from exc

    def chat(self, system: str, user: str) -> str:
        body: dict[str, Any] = {
            "model": self.resolve_model(),
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        payload = self._post("/v1/messages", body)
        try:
            for block in payload.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    return block["text"]
            raise LLMClientError(f"No text content block in response: {payload}")
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError(f"Unexpected Anthropic response: {payload}") from exc

    def _get(self, path: str) -> dict[str, Any]:
        return _send_request(
            request.Request(
                f"{self.base_url}{path}",
                headers={"x-api-key": self.api_key},
                method="GET",
            ),
            self.timeout,
            f"Cannot reach Anthropic API at {self.base_url}",
        )

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8")
        return _send_request(
            request.Request(
                f"{self.base_url}{path}",
                data=data,
                headers={
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
                method="POST",
            ),
            self.timeout,
            f"Cannot reach Anthropic API at {self.base_url}",
        )


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.removeprefix("json").strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        return stripped
    return stripped[start : end + 1]
