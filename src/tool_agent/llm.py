"""LLM protocol: one interface, two implementations.

`ChatLLM` is the protocol the agent loop depends on. Implementations:

* `ScriptedLLM`     - replays a scripted sequence of JSON decisions.
  Makes the agent fully deterministic: tests, CI, and the offline demo
  run with zero credentials, zero cost, zero flakiness.
* `AzureOpenAILLM`   - thin adapter over the `openai` SDK configured for
  Azure OpenAI (env-provided credentials). Swapped in with one line at
  the composition root; the agent never knows the difference.

Both speak the same JSON contract with the agent loop:

    {"action": "tool", "tool": "...", "arguments": {...}}
    {"action": "final", "answer": "..."}
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from tool_agent.models import ToolCall


class LLMError(Exception):
    """Raised when an LLM implementation fails or violates the contract."""


class ChatLLM(Protocol):
    """Anything that can turn a conversation into a next agent decision."""

    def decide(self, conversation: list[dict[str, str]]) -> str: ...
    @property
    def name(self) -> str: ...


def parse_decision(raw: str) -> ToolCall | list[ToolCall] | tuple[str, str]:
    """Parse one JSON decision from the LLM.

    Returns:
        ToolCall          - when the LLM wants one tool executed.
        list[ToolCall]    - fan-out form: several calls in one turn.
        ("final", answer) - when the LLM has its final answer.
    Raises:
        LLMError on malformed JSON or unknown action.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = f"LLM output was not valid JSON: {raw[:200]!r}"
        raise LLMError(msg) from exc
    if not isinstance(payload, dict):
        msg = f"LLM decision must be a JSON object, got: {type(payload).__name__}"
        raise LLMError(msg)

    action = payload.get("action")
    if action == "tool":
        # fan-out form: one decision may request several tool calls
        if "tools" in payload:
            entries = payload["tools"]
            if not isinstance(entries, list) or not entries:
                msg = "tool action with 'tools' requires at least one call"
                raise LLMError(msg)
            calls: list[ToolCall] = []
            for entry in entries:
                if not isinstance(entry, dict):
                    msg = f"each entry of 'tools' must be an object, got: {type(entry).__name__}"
                    raise LLMError(msg)
                sub = parse_decision(json.dumps({**entry, "action": "tool"}))
                if not isinstance(sub, ToolCall):
                    msg = "entries of 'tools' must be tool calls"
                    raise LLMError(msg)
                calls.append(sub)
            return calls
        tool = payload.get("tool")
        if not tool or not isinstance(tool, str):
            msg = "tool action requires a non-empty 'tool' field"
            raise LLMError(msg)
        arguments = payload.get("arguments", {})
        if not isinstance(arguments, dict):
            msg = f"tool arguments must be an object for tool {tool!r}"
            raise LLMError(msg)
        return ToolCall(tool=tool, arguments=arguments)
    if action == "final":
        answer = payload.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            msg = "final action requires a non-empty 'answer' string"
            raise LLMError(msg)
        return ("final", answer)
    msg = f"unknown action {action!r} (expected 'tool' or 'final')"
    raise LLMError(msg)


@dataclass
class ScriptedLLM:
    """Deterministic LLM stand-in: replays scripted decisions in order.

    The core of the test/demo strategy: the agent loop is exercised
    end-to-end (tool dispatch, retries, tracing) without any network.
    """

    script: list[str]
    name: str = "scripted-llm"
    _cursor: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.script:
            msg = "script must contain at least one decision"
            raise ValueError(msg)

    def decide(self, conversation: list[dict[str, str]]) -> str:
        if self._cursor >= len(self.script):
            msg = (
                "script exhausted: the agent asked for more decisions than "
                f"were scripted ({len(self.script)})"
            )
            raise LLMError(msg)
        decision = self.script[self._cursor]
        self._cursor += 1
        return decision

    @property
    def remaining(self) -> int:
        return len(self.script) - self._cursor


@dataclass
class AzureOpenAILLM:
    """Adapter over the Azure OpenAI `openai` SDK (optional dependency).

    Credentials come from the environment (never code):

    * AZURE_OPENAI_ENDPOINT     e.g. https://<res>.openai.azure.com
    * AZURE_OPENAI_API_KEY
    * AZURE_OPENAI_DEPLOYMENT   the chat deployment name (e.g. gpt-4o)
    """

    deployment: str | None = None
    endpoint: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    name: str = "azure-openai"

    def __post_init__(self) -> None:
        try:
            import openai  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            msg = "openai is required; install with: pip install tool-agent[azure]"
            raise ImportError(msg) from exc
        self._deployment = self.deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self._endpoint = self.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self._api_key = self.api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if not (self._deployment and self._endpoint and self._api_key):
            msg = (
                "Azure OpenAI needs AZURE_OPENAI_ENDPOINT, "
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_DEPLOYMENT "
                "(env vars or constructor args)"
            )
            raise LLMError(msg)

    def decide(self, conversation: list[dict[str, str]]) -> str:  # pragma: no cover - live service
        from openai import AzureOpenAI as _AzureOpenAI  # type: ignore[attr-defined]

        client = _AzureOpenAI(  # type: ignore[attr-defined]
            api_version="2024-10-21",
            azure_endpoint=self._endpoint,
            api_key=self._api_key,
        )
        response = client.chat.completions.create(
            model=self._deployment,
            temperature=self.temperature,
            messages=conversation,  # type: ignore[arg-type]
        )
        content: Any = response.choices[0].message.content or ""
        return str(content)
