"""Thin client for any OpenAI-compatible ``/chat/completions`` endpoint.

Deliberately not the vendor SDK: one HTTP call is all we need, and this keeps
DeepSeek, Qwen, Moonshot, OpenAI, OpenRouter and a local vLLM interchangeable
behind a base URL.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Any, Iterable

import httpx

from .config import LLMConfig, get_settings

log = logging.getLogger(__name__)

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class LLMError(RuntimeError):
    pass


class LLMNotConfigured(LLMError):
    pass


class LLMClient:
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or get_settings().llm

    @property
    def is_configured(self) -> bool:
        return bool(self.config.api_key)

    def _require_key(self) -> str:
        key = self.config.api_key
        if not key:
            raise LLMNotConfigured(
                f"No API key found. Set {self.config.api_key_env} in your environment "
                "or .env file."
            )
        return key

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        tier: str = "default",
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_object: bool = False,
    ) -> str:
        """Run one chat completion and return the assistant text."""
        api_key = self._require_key()
        payload: dict[str, Any] = {
            "model": self.config.model_for(tier),
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if json_object:
            payload["response_format"] = {"type": "json_object"}

        url = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                with httpx.Client(timeout=self.config.timeout) as client:
                    response = client.post(url, json=payload, headers=headers)

                if response.status_code in (429, 500, 502, 503, 504, 529):
                    raise _Retryable(f"HTTP {response.status_code}: {response.text[:300]}")
                if response.status_code >= 400:
                    # `response_format` is not universally supported; retry without it
                    # rather than failing the whole run.
                    if json_object and "response_format" in response.text:
                        payload.pop("response_format", None)
                        json_object = False
                        continue
                    raise LLMError(f"HTTP {response.status_code}: {response.text[:500]}")

                data = response.json()
                choices = data.get("choices") or []
                if not choices:
                    raise _Retryable(f"no choices in response: {str(data)[:300]}")
                content = (choices[0].get("message") or {}).get("content")
                if not content:
                    # Some providers put reasoning in a sibling field and leave
                    # content empty when they hit the token cap.
                    raise _Retryable("empty completion content")
                return content.strip()

            except (_Retryable, httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = exc
                if attempt == self.config.max_retries - 1:
                    break
                delay = min(2**attempt + random.uniform(0, 1), 30)
                log.warning("LLM call failed (%s), retrying in %.1fs", exc, delay)
                time.sleep(delay)

        raise LLMError(f"LLM request failed after {self.config.max_retries} attempts: {last_error}")

    def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        tier: str = "default",
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Complete and parse a JSON object, tolerating fences and stray prose."""
        raw = self.complete(messages, tier=tier, max_tokens=max_tokens, json_object=True)
        parsed = parse_json_object(raw)
        if parsed is None:
            raise LLMError(f"model did not return JSON: {raw[:300]}")
        return parsed


class _Retryable(Exception):
    pass


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object from a model response."""
    text = text.strip()
    for candidate in _json_candidates(text):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _json_candidates(text: str) -> Iterable[str]:
    yield text
    fence = _JSON_FENCE.search(text)
    if fence:
        yield fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        yield text[start : end + 1]


def as_list(value: Any, limit: int = 24) -> list[str]:
    """Coerce a model-supplied field into a clean list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip(" -•\t") for part in value.split("\n") if part.strip()]
        return [part for part in parts if part][:limit]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                # Some models answer with [{"point": "..."}] instead of ["..."].
                for candidate in item.values():
                    if isinstance(candidate, str) and candidate.strip():
                        out.append(candidate.strip())
                        break
        return out[:limit]
    return [str(value)]


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(as_list(value))
    return str(value)
