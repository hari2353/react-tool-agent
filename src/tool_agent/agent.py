"""The ReAct-style agent loop.

One iteration = LLM decision -> tool dispatch -> observation appended.
The loop terminates on a "final" action, on an LLM contract violation
after a repair attempt, or on hitting `max_steps` (never loops forever).

Everything is injected (LLM protocol, tool registry) and every step is
recorded in an immutable trace - the observability story interviewers
ask about (tool latency, retries, token budget) is demonstrated by
structure here.

Optional durability: `checkpoint_path=` persists a snapshot after every
completed step, `resume(path)` continues it, and an `approve` callback gates
each tool call before dispatch.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tool_agent.checkpoint import Checkpoint, JsonCheckpointStore
from tool_agent.llm import ChatLLM, LLMError, parse_decision
from tool_agent.models import AgentResult, AgentStep, Role, ToolCall
from tool_agent.tools import ToolError, ToolRegistry

_SYSTEM_PREAMBLE = (
    "You are a precise assistant that answers questions using tools when "
    "helpful. Always respond with a single JSON object, no other text:\n"
)


def _repair_message(error: str) -> str:
    """Format reminder fed back after a malformed model output."""
    sample_tool = '{"action": "tool", "tool": "name", "arguments": {}}'
    sample_final = '{"action": "final", "answer": "..."}'
    return (
        f"Your last output was invalid ({error}). Reply with EXACTLY one "
        f"JSON decision and nothing else: {sample_tool} or {sample_final}."
    )


@dataclass
class AgentConfig:
    """Loop controls."""

    max_steps: int = 8
    max_tools_per_decision: int = 4
    max_retries: int = 1
    max_output_chars: int = 40_000
    max_tool_retries: int = 1

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            msg = "max_steps must be >= 1"
            raise ValueError(msg)
        if self.max_tools_per_decision < 1:
            msg = "max_tools_per_decision must be >= 1"
            raise ValueError(msg)
        if self.max_retries < 0:
            msg = "max_retries must be >= 0"
            raise ValueError(msg)
        if self.max_output_chars < 1:
            msg = "max_output_chars must be >= 1"
            raise ValueError(msg)
        if self.max_tool_retries < 0:
            msg = "max_tool_retries must be >= 0"
            raise ValueError(msg)


@dataclass
class _State:
    """Everything the loop needs to continue after a crash."""

    question: str
    conversation: list[dict[str, str]]
    steps: list[AgentStep] = field(default_factory=list)
    calls: list[ToolCall] = field(default_factory=list)
    step_no: int = 0
    pending: list[ToolCall] = field(default_factory=list)
    used_chars: int = 0
    consecutive_errors: int = 0
    raw: str = ""
    assistant_echoed: bool = False


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

    def run(
        self,
        question: str,
        *,
        checkpoint_path: str | Path | None = None,
        approve: Callable[[ToolCall], bool] | None = None,
    ) -> AgentResult:
        """Answer one question; never raises for tool/LLM misbehavior -
        failures are captured in the result with success=False."""
        state = _State(
            question=question,
            conversation=[
                {
                    "role": Role.SYSTEM.value,
                    "content": _SYSTEM_PREAMBLE + self._tools.system_prompt(),
                },
                {"role": Role.USER.value, "content": question},
            ],
        )
        store = JsonCheckpointStore(checkpoint_path) if checkpoint_path else None
        return self._drive(state, approve, store)

    def resume(
        self,
        checkpoint_path: str | Path,
        *,
        approve: Callable[[ToolCall], bool] | None = None,
    ) -> AgentResult:
        """Continue a persisted run without repeating applied tool calls."""
        store = JsonCheckpointStore(checkpoint_path)
        checkpoint = store.load()
        state = _State(
            question=checkpoint.question,
            conversation=[dict(m) for m in checkpoint.conversation],
            steps=list(checkpoint.steps),
            calls=list(checkpoint.calls),
            step_no=checkpoint.step_no,
            pending=list(checkpoint.pending),
            used_chars=checkpoint.used_chars,
            consecutive_errors=checkpoint.consecutive_errors,
            raw=checkpoint.raw,
            assistant_echoed=checkpoint.assistant_echoed,
        )
        return self._drive(state, approve, store)

    def _drive(
        self,
        state: _State,
        approve: Callable[[ToolCall], bool] | None,
        store: JsonCheckpointStore | None,
    ) -> AgentResult:
        def save() -> None:
            if store is not None:
                store.save(
                    Checkpoint(
                        version=1,
                        question=state.question,
                        conversation=tuple(dict(m) for m in state.conversation),
                        steps=tuple(state.steps),
                        calls=tuple(state.calls),
                        step_no=state.step_no,
                        pending=tuple(state.pending),
                        used_chars=state.used_chars,
                        consecutive_errors=state.consecutive_errors,
                        raw=state.raw,
                        assistant_echoed=state.assistant_echoed,
                    )
                )

        def finish(result: AgentResult) -> AgentResult:
            if store is not None:
                store.clear()
            return result

        while state.step_no < self._config.max_steps:
            state.step_no += 1
            if state.pending:
                # remaining calls of a multi-tool decision need no new LLM turn
                call = state.pending.pop(0)
            else:
                state.assistant_echoed = False
                decision, failure = self._next_decision(state)
                if failure is not None:
                    return finish(failure)
                assert decision is not None
                if isinstance(decision, list):
                    if len(decision) > self._config.max_tools_per_decision:
                        detail = (
                            f"decision requested {len(decision)} tool calls "
                            f"> max_tools_per_decision={self._config.max_tools_per_decision}"
                        )
                        state.steps.append(
                            AgentStep(step=state.step_no, kind="llm_response", detail=detail)
                        )
                        return finish(_failure(state.steps, state.calls, detail))
                    state.pending = list(decision)
                    call = state.pending.pop(0)
                elif isinstance(decision, tuple):  # ("final", answer)
                    _, answer = decision
                    state.steps.append(AgentStep(step=state.step_no, kind="final", detail=answer))
                    return finish(
                        AgentResult(
                            answer=answer,
                            steps=tuple(state.steps),
                            tool_calls_made=tuple(state.calls),
                            success=True,
                        )
                    )
                else:
                    call = decision

            state.steps.append(
                AgentStep(
                    step=state.step_no,
                    kind="tool_call",
                    tool=call.tool,
                    tool_call=call,
                    detail=json.dumps(call.arguments, default=str),
                )
            )
            state.calls.append(call)
            # the model's raw decision is echoed once per decision turn,
            # before the first tool result (not re-echoed for queued calls)
            if not state.assistant_echoed:
                state.conversation.append({"role": Role.ASSISTANT.value, "content": state.raw})
                state.assistant_echoed = True

            if approve is not None and not approve(call):
                denied = "tool call denied by human approval"
                state.steps.append(
                    AgentStep(
                        step=state.step_no,
                        kind="approval_denied",
                        tool=call.tool,
                        tool_call=call,
                        detail=denied,
                    )
                )
                return finish(_failure(state.steps, state.calls, denied))

            failure = self._execute_tool(state, call)
            if failure is not None:
                return finish(failure)
            save()

        return finish(
            _failure(
                state.steps,
                state.calls,
                f"reached max_steps={self._config.max_steps} without a final answer",
            )
        )

    def _next_decision(
        self,
        state: _State,
    ) -> tuple[
        ToolCall | list[ToolCall] | tuple[str, str] | None,
        AgentResult | None,
    ]:
        """One LLM turn with budget accounting and bounded repair.

        Returns (decision, None) or (None, failure-result already
        recorded in the trace)."""
        error: LLMError | None = None
        for attempt in range(self._config.max_retries + 1):
            try:
                state.raw = self._llm.decide(state.conversation)
            except LLMError as exc:
                error = exc
            else:
                # every model output is charged, malformed or not
                state.used_chars += len(state.raw)
                if state.used_chars > self._config.max_output_chars:
                    detail = (
                        f"output budget exceeded: {state.used_chars} > "
                        f"{self._config.max_output_chars} chars of model output"
                    )
                    state.steps.append(
                        AgentStep(step=state.step_no, kind="llm_response", detail=detail)
                    )
                    return None, _failure(state.steps, state.calls, detail)
                try:
                    return parse_decision(state.raw), None
                except LLMError as exc:
                    error = exc
            if attempt < self._config.max_retries:
                # feed the parse error back so the model can fix its format
                state.steps.append(
                    AgentStep(
                        step=state.step_no,
                        kind="repair",
                        detail=f"repairing LLM error: {error}",
                    )
                )
                state.conversation.append(
                    {"role": Role.USER.value, "content": _repair_message(str(error))}
                )
        detail = str(error)
        state.steps.append(
            AgentStep(step=state.step_no, kind="llm_response", detail=f"LLM error: {detail}")
        )
        return None, _failure(state.steps, state.calls, detail)

    def _execute_tool(self, state: _State, call: ToolCall) -> AgentResult | None:
        """Dispatch with bounded retries; each attempt is in the trace."""
        for attempt in range(1, self._config.max_tool_retries + 2):
            try:
                result: Any = self._tools.dispatch(call)
            except ToolError as exc:
                state.consecutive_errors += 1
                state.steps.append(
                    AgentStep(
                        step=state.step_no,
                        kind="tool_result",
                        tool=call.tool,
                        detail=f"error: {exc}",
                    )
                )
                if state.consecutive_errors >= 3:
                    return _failure(
                        state.steps,
                        state.calls,
                        f"three consecutive tool failures (last: {exc})",
                    )
                if attempt <= self._config.max_tool_retries:
                    continue
                # feed the error back so the LLM can repair its next decision
                state.conversation.append(
                    {
                        "role": Role.TOOL.value,
                        "content": f"tool {call.tool} failed with: {exc}. "
                        "Fix the arguments or try a different tool.",
                    }
                )
                return None
            else:
                state.consecutive_errors = 0
                state.steps.append(
                    AgentStep(
                        step=state.step_no,
                        kind="tool_result",
                        tool=call.tool,
                        result=result,
                        detail=_summarize(result),
                    )
                )
                state.conversation.append(
                    {
                        "role": Role.TOOL.value,
                        "content": f"tool {call.tool} returned: " + _summarize(result),
                    }
                )
                return None
        return None


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
