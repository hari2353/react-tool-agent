"""Domain models for the tool-calling agent.

Typed frozen dataclasses: messages, tool specs, tool calls, and the agent
trace. These are the stable contract between the LLM protocol, the tools,
and the agent loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Declarative description of a tool the agent may call."""

    name: str
    description: str
    parameters: dict[str, str]  # name -> human-readable type description

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            msg = "tool name must be non-empty"
            raise ValueError(msg)
        if not self.description or not self.description.strip():
            msg = f"tool {self.name}: description must be non-empty"
            raise ValueError(msg)
        if not self.name.replace("_", "").isalnum():
            msg = f"tool name {self.name!r} must be snake_case alphanumerics"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One call the LLM wants to make."""

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tool or not self.tool.strip():
            msg = "tool name must be non-empty"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class AgentStep:
    """One recorded step in the agent trace (observability contract)."""

    step: int
    kind: str                 # "llm_response" | "tool_call" | "tool_result" | "final"
    tool: str | None = None
    detail: str = ""
    tool_call: ToolCall | None = None
    result: Any = None


@dataclass(frozen=True, slots=True)
class AgentResult:
    """Final outcome of one agent run."""

    answer: str
    steps: tuple[AgentStep, ...]
    tool_calls_made: tuple[ToolCall, ...]
    success: bool
    error: str | None = None

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def tools_used(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(c.tool for c in self.tool_calls_made))
