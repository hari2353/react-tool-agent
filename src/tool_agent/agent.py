"""The ReAct-style agent loop.

One iteration = LLM decision -> tool dispatch -> observation appended.
The loop terminates on a "final" action, on an LLM contract violation
after a repair attempt, or on hitting `max_steps` (never loops forever).

Everything is injected (LLM protocol, tool registry) and every step is
recorded in an immutable trace - the observability story interviewers
ask about (tool latency, retries, token budget) is demonstrated by
structure here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from tool_agent.llm import ChatLLM, LLMError, parse_decision
from tool_agent.models import AgentResult, AgentStep, Role, ToolCall
from tool_agent.tools import ToolError, ToolRegistry

_SYSTEM_PREAMBLE = (
    "You are a precise assistant that answers questions using tools when "
    "helpful. Always respond with a single JSON object, no other text:\n"
)


@dataclass
class AgentConfig:
    """Loop controls."""

    max_steps: int = 8

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            msg = "max_steps must be >= 1"
            raise ValueError(msg)


class ToolAgent:
    """ReAct agent over an injected LLM and tool registry."""

    def __init__(
        self,
        llm: ChatLLM,
        tools: ToolRegistry,
        config: AgentConfig | None = None,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._config = config or AgentConfig()

    def run(self, question: str) -> AgentResult:
        """Answer one question; never raises for tool/LLM misbehavior -
        failures are captured in the result with success=False."""
        steps: list[AgentStep] = []
        calls: list[ToolCall] = []
        conversation: list[dict[str, str]] = [
            {"role": Role.SYSTEM.value, "content": _SYSTEM_PREAMBLE + self._tools.system_prompt()},
            {"role": Role.USER.value, "content": question},
        ]
        step_no = 0
        consecutive_errors = 0

        while step_no < self._config.max_steps:
            step_no += 1
            try:
                raw = self._llm.decide(conversation)
                decision: ToolCall | tuple[str, str] = parse_decision(raw)
            except LLMError as exc:
                steps.append(
                    AgentStep(
                        step=step_no,
                        kind="llm_response",
                        detail=f"LLM error: {exc}",
                    )
                )
                return _failure(steps, calls, str(exc))
            if isinstance(decision, tuple):  # ("final", answer)
                _, answer = decision
                steps.append(
                    AgentStep(step=step_no, kind="final", detail=answer)
                )
                return AgentResult(
                    answer=answer,
                    steps=tuple(steps),
                    tool_calls_made=tuple(calls),
                    success=True,
                )

            call: ToolCall = decision
            steps.append(
                AgentStep(
                    step=step_no,
                    kind="tool_call",
                    tool=call.tool,
                    tool_call=call,
                    detail=json.dumps(call.arguments, default=str),
                )
            )
            calls.append(call)
            try:
                result: Any = self._tools.dispatch(call)
                consecutive_errors = 0
                steps.append(
                    AgentStep(
                        step=step_no,
                        kind="tool_result",
                        tool=call.tool,
                        result=result,
                        detail=_summarize(result),
                    )
                )
                conversation.append(
                    {"role": Role.ASSISTANT.value, "content": raw}
                )
                conversation.append(
                    {
                        "role": Role.TOOL.value,
                        "content": f"tool {call.tool} returned: "
                        + _summarize(result),
                    }
                )
            except ToolError as exc:
                consecutive_errors += 1
                steps.append(
                    AgentStep(
                        step=step_no,
                        kind="tool_result",
                        tool=call.tool,
                        detail=f"error: {exc}",
                    )
                )
                # feed the error back so the LLM can repair its next decision
                conversation.append({"role": Role.ASSISTANT.value, "content": raw})
                conversation.append(
                    {
                        "role": Role.TOOL.value,
                        "content": f"tool {call.tool} failed with: {exc}. "
                        "Fix the arguments or try a different tool.",
                    }
                )
                if consecutive_errors >= 3:
                    return _failure(
                        steps,
                        calls,
                        f"three consecutive tool failures (last: {exc})",
                    )

        return _failure(
            steps,
            calls,
            f"reached max_steps={self._config.max_steps} without a final answer",
        )


def _summarize(result: Any, limit: int = 400) -> str:
    """Render a tool result compactly for the conversation and trace."""
    if isinstance(result, list | dict):
        text = json.dumps(result, default=str)
    else:
        text = str(result)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _failure(steps: list[AgentStep], calls: list[ToolCall], error: str) -> AgentResult:
    return AgentResult(
        answer=f"I could not complete the task: {error}",
        steps=tuple(steps),
        tool_calls_made=tuple(calls),
        success=False,
        error=error,
    )
