"""Agent loop tests: multi-step scripted runs, repair, and failure modes."""

from __future__ import annotations

import json

import pytest

from tool_agent.agent import AgentConfig, ToolAgent
from tool_agent.llm import LLMError, ScriptedLLM
from tool_agent.retrieval import default_store
from tool_agent.tools import default_registry

pytestmark = pytest.mark.integration


def _tool(name: str, **arguments: object) -> str:
    return json.dumps({"action": "tool", "tool": name, "arguments": arguments})


def _final(answer: str) -> str:
    return json.dumps({"action": "final", "answer": answer})


def _agent(script: list[str], max_steps: int = 8) -> ToolAgent:
    registry = default_registry(default_store())
    return ToolAgent(
        llm=ScriptedLLM(script=script),
        tools=registry,
        config=AgentConfig(max_steps=max_steps),
    )


class TestHappyPaths:
    def test_immediate_final(self) -> None:
        result = _agent([_final("hello")]).run("anything")
        assert result.success
        assert result.answer == "hello"
        assert result.tool_calls_made == ()
        assert result.n_steps == 1

    def test_multi_step_tool_chain(self) -> None:
        script = [
            _tool("kb_search", query="zero tolerance findings", top_k=2),
            _tool("calculator", expression="7 * 6"),
            _final("zero tolerance suspends POs; 42"),
        ]
        result = _agent(script).run("explain zero tolerance policy and compute 7*6")
        assert result.success
        assert len(result.tool_calls_made) == 2
        assert result.tools_used == ("kb_search", "calculator")
        # 2 steps per tool (call + result) + 1 final step
        assert result.n_steps == 5
        # the retrieval result actually flowed back through the loop
        tool_results = [s for s in result.steps if s.kind == "tool_result"]
        assert any("zero-tolerance" in str(r.result) for r in tool_results)

    def test_tool_arguments_flow_correctly(self) -> None:
        result = _agent(
            [_tool("unit_convert", value=100, unit="mi", target="km"), _final("done")]
        ).run("convert 100 miles")
        call = result.tool_calls_made[0]
        assert call.arguments == {"value": 100, "unit": "mi", "target": "km"}


class TestRepairAndFailure:
    def test_agent_recovers_from_bad_tool_call(self) -> None:
        """LLM hallucinates an unknown tool, gets the error, repairs."""
        script = [
            _tool("web_search", query="supplier policy"),          # unknown tool
            _tool("kb_search", query="supplier policy", top_k=1),  # repair
            _final("found it"),
        ]
        result = _agent(script).run("find supplier policy")
        assert result.success
        assert result.answer == "found it"
        error_steps = [s for s in result.steps if s.kind == "tool_result" and "error" in s.detail]
        assert any("unknown tool" in s.detail for s in error_steps)

    def test_agent_recovers_from_missing_arguments(self) -> None:
        script = [
            _tool("calculator"),                       # missing 'expression'
            _tool("calculator", expression="2+2"),      # repair
            _final("4"),
        ]
        result = _agent(script).run("compute 2+2")
        assert result.success
        assert result.answer == "4"

    def test_three_consecutive_failures_abort(self) -> None:
        script = [
            _tool("nope1"), _tool("nope2"), _tool("nope3"), _final("never reached"),
        ]
        result = _agent(script).run("x")
        assert not result.success
        assert "three consecutive tool failures" in result.error
        assert result.answer.startswith("I could not complete")

    def test_max_steps_enforced(self) -> None:
        script = [_tool("calculator", expression=f"{i}+1") for i in range(10)]
        result = _agent(script, max_steps=3).run("loop forever")
        assert not result.success
        assert "max_steps=3" in result.error
        assert len(result.tool_calls_made) == 3

    def test_llm_contract_violation_repaired_then_recovered(self) -> None:
        # a malformed output gets one bounded repair re-prompt by default
        result = _agent(["not json at all", _final("recovered")]).run("x")
        assert result.success
        assert any(s.kind == "repair" for s in result.steps)

    def test_script_exhaustion_captured(self) -> None:
        # agent wants more decisions than scripted, within max_steps budget
        result = _agent([_tool("calculator", expression="1+1")], max_steps=8).run("x")
        assert not result.success
        assert "script exhausted" in result.error


class TestConfigValidation:
    def test_max_steps_bounds(self) -> None:
        with pytest.raises(ValueError, match="max_steps"):
            AgentConfig(max_steps=0)

    def test_trace_is_immutable_snapshot(self) -> None:
        result = _agent([_final("ok")]).run("x")
        with pytest.raises(AttributeError):
            result.steps = ()  # type: ignore[misc]


class TestDeterminism:
    def test_same_script_same_trace(self) -> None:
        question = "explain the capa deadlines and compute 30*7"
        script = [
            _tool("kb_search", query="corrective action deadlines", top_k=1),
            _tool("calculator", expression="30 * 7"),
            _final("critical=7d, major=30d, minor=60d; 210"),
        ]
        r1 = _agent(script).run(question)
        r2 = _agent(
            [
                _tool("kb_search", query="corrective action deadlines", top_k=1),
                _tool("calculator", expression="30 * 7"),
                _final("critical=7d, major=30d, minor=60d; 210"),
            ]
        ).run(question)
        assert r1.answer == r2.answer
        assert [s.kind for s in r1.steps] == [s.kind for s in r2.steps]
        assert [c.tool for c in r1.tool_calls_made] == [c.tool for c in r2.tool_calls_made]


class TestScriptedLLMDirectUse:
    def test_remaining_tracks_cursor(self) -> None:
        llm = ScriptedLLM(script=[_final("a"), _final("b")])
        assert llm.remaining == 2
        llm.decide([])
        assert llm.remaining == 1
        llm.decide([])
        with pytest.raises(LLMError):
            llm.decide([])
