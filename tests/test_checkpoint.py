"""Durable checkpoints, bounded tool retries, and the approval gate."""

from __future__ import annotations

import json

import pytest

from tool_agent.agent import AgentConfig, ToolAgent
from tool_agent.checkpoint import Checkpoint, JsonCheckpointStore
from tool_agent.llm import ScriptedLLM
from tool_agent.models import ToolCall, ToolSpec
from tool_agent.tools import ToolError, ToolRegistry


def _tool(name: str = "ping", **arguments: object) -> str:
    return json.dumps({"action": "tool", "tool": name, "arguments": arguments})


def _final(answer: str) -> str:
    return json.dumps({"action": "final", "answer": answer})


class FlakyRegistry(ToolRegistry):
    def __init__(self, failures: int = 0) -> None:
        super().__init__()
        self.calls = 0
        self.failures = failures
        self.register(ToolSpec("ping", "Ping.", {}), self._ping)

    def _ping(self) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise ToolError("temporary failure")
        return "pong"


class CrashLLM:
    name = "crash"

    def __init__(self) -> None:
        self.calls = 0

    def decide(self, conversation):
        self.calls += 1
        if self.calls == 1:
            return _tool()
        raise KeyboardInterrupt("simulated crash")


def test_checkpoint_roundtrip_and_atomic_overwrite(tmp_path) -> None:
    store = JsonCheckpointStore(tmp_path / "cp.json")
    first = Checkpoint(1, "q", ({"role": "user", "content": "q"},), (), (), 0, (), 0, 0, "", False)
    store.save(first)
    assert store.load() == first
    second = Checkpoint(1, "q2", (), (), (), 3, (ToolCall("ping", {}),), 9, 1, "raw", True)
    store.save(second)
    assert store.load() == second
    store.clear()
    with pytest.raises(ValueError):
        store.load()


def test_corrupt_checkpoint_raises(tmp_path) -> None:
    path = tmp_path / "cp.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        JsonCheckpointStore(path).load()
    path.write_text(json.dumps({"version": 99}), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        JsonCheckpointStore(path).load()


def test_resume_after_crash_does_not_repeat_the_tool(tmp_path) -> None:
    path = tmp_path / "cp.json"
    registry = FlakyRegistry()
    with pytest.raises(KeyboardInterrupt):
        ToolAgent(CrashLLM(), registry).run("q", checkpoint_path=path)
    assert registry.calls == 1
    assert path.exists()

    result = ToolAgent(ScriptedLLM([_final("done")]), registry).resume(path)
    assert result.success
    assert result.answer == "done"
    assert registry.calls == 1
    assert [s.kind for s in result.steps].count("tool_call") == 1
    assert not path.exists()


def test_approval_denial_does_not_execute_the_tool(tmp_path) -> None:
    registry = FlakyRegistry()
    agent = ToolAgent(ScriptedLLM([_tool(), _final("never")]), registry)
    result = agent.run("q", approve=lambda call: False)
    assert result.success is False
    assert result.error == "tool call denied by human approval"
    assert registry.calls == 0
    assert result.steps[-1].kind == "approval_denied"


def test_approval_accepts_and_runs() -> None:
    registry = FlakyRegistry()
    result = ToolAgent(ScriptedLLM([_tool(), _final("ok")]), registry).run(
        "q", approve=lambda call: call.tool == "ping"
    )
    assert result.success
    assert registry.calls == 1


def test_tool_retry_succeeds_on_second_attempt() -> None:
    registry = FlakyRegistry(failures=1)
    result = ToolAgent(
        ScriptedLLM([_tool(), _final("ok")]),
        registry,
        AgentConfig(max_tool_retries=1),
    ).run("q")
    assert result.success
    assert registry.calls == 2
    results = [s for s in result.steps if s.kind == "tool_result"]
    assert "error" in results[0].detail
    assert "pong" in results[1].detail


def test_tool_retry_limit_is_respected() -> None:
    registry = FlakyRegistry(failures=99)
    result = ToolAgent(
        ScriptedLLM([_tool(), _final("ok")]),
        registry,
        AgentConfig(max_tool_retries=0),
    ).run("q")
    assert registry.calls == 1
    assert result.success  # the error is fed back; the model may still finish


def test_invalid_tool_retry_config() -> None:
    with pytest.raises(ValueError, match="max_tool_retries"):
        AgentConfig(max_tool_retries=-1)
