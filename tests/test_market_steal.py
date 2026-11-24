"""Market-steal additions (2026-11-18): multi-tool decisions, run budget,
bounded repair re-prompt.

Compared products treat these as baseline: OpenAI/Anthropic tool-use and
LangGraph fan a single model turn out to several tool calls; production
loops bound total output (a cost proxy), not just step count; and a
malformed model output is repaired with a format reminder instead of
aborting the run on the first bad parse.
"""

from __future__ import annotations

import json

import pytest

from tool_agent.agent import AgentConfig, ToolAgent
from tool_agent.llm import LLMError, ScriptedLLM, parse_decision
from tool_agent.models import ToolCall
from tool_agent.retrieval import default_store
from tool_agent.tools import default_registry

pytestmark = pytest.mark.integration


def _tool(name: str, **arguments: object) -> str:
    return json.dumps({"action": "tool", "tool": name, "arguments": arguments})


def _tools(*calls: tuple[str, dict]) -> str:
    return json.dumps({"action": "tool", "tools": [{"tool": n, "arguments": a} for n, a in calls]})


def _final(answer: str) -> str:
    return json.dumps({"action": "final", "answer": answer})


def _agent(script: list[str], **config: object) -> ToolAgent:
    return ToolAgent(
        llm=ScriptedLLM(script=script),
        tools=default_registry(default_store()),
        config=AgentConfig(**config),  # type: ignore[arg-type]
    )


# ── multi-tool decisions ─────────────────────────────────────────────────────
class TestMultiToolDecisions:
    def test_parse_accepts_list_form(self) -> None:
        decision = parse_decision(_tools(("calculator", {"expression": "1+1"})))
        assert decision == [ToolCall(tool="calculator", arguments={"expression": "1+1"})]

    def test_list_form_rejects_empty(self) -> None:
        with pytest.raises(LLMError, match="at least one"):
            parse_decision(_tools())

    def test_list_form_rejects_non_dict_entries(self) -> None:
        with pytest.raises(LLMError, match="object"):
            parse_decision(json.dumps({"action": "tool", "tools": ["calculator"]}))

    def test_multi_tool_decision_executes_in_order(self) -> None:
        script = [
            _tools(
                ("kb_search", {"query": "audit cadence", "top_k": 1}),
                ("unit_convert", {"value": 100, "unit": "mi", "target": "km"}),
                ("calculator", {"expression": "6*7"}),
            ),
            _final("done"),
        ]
        result = _agent(script).run("multi")
        assert result.success
        assert [c.tool for c in result.tool_calls_made] == [
            "kb_search",
            "unit_convert",
            "calculator",
        ]
        # one recorded call+result pair per call, in order
        call_steps = [s for s in result.steps if s.kind == "tool_call"]
        result_steps = [s for s in result.steps if s.kind == "tool_result"]
        assert len(call_steps) == 3 and len(result_steps) == 3
        assert result.tools_used == ("kb_search", "unit_convert", "calculator")

    def test_mid_list_tool_failure_feeds_repair(self) -> None:
        script = [
            _tools(
                ("calculator", {"expression": "1+1"}),
                ("no_such_tool", {}),
            ),
            _final("gave my best"),
        ]
        result = _agent(script).run("multi")
        assert result.success, result.error  # the model recovers with a final
        errors = [s for s in result.steps if s.kind == "tool_result" and s.result is None]
        assert errors and "no_such_tool" in errors[0].detail

    def test_over_cap_multi_tool_decision_captured(self) -> None:
        calls = [("calculator", {"expression": f"{i}+1"}) for i in range(5)]
        result = _agent([_tools(*calls), _final("x")], max_tools_per_decision=4).run("x")
        assert not result.success
        assert "max_tools_per_decision" in result.error


# ── run budget ───────────────────────────────────────────────────────────────
class TestRunBudget:
    def test_budget_ends_run_cleanly(self) -> None:
        big = _tool("kb_search", query="supplier compliance audit cadence bands")
        script = [big, _final("tiny")]
        budget = len(big) + len(_final("tiny"))  # both decisions together
        result = _agent(script, max_output_chars=budget - 1).run("x")
        assert not result.success
        assert "output budget" in result.error

    def test_budget_within_limit_completes(self) -> None:
        big = _tool("calculator", expression="2+2")
        script = [big, _final("2+2 is 4")]
        budget = len(big) + len(_final("2+2 is 4"))
        result = _agent(script, max_output_chars=budget).run("x")
        assert result.success

    def test_budget_counts_malformed_decisions_too(self) -> None:
        result = _agent(["garbage " * 30, _final("x")], max_output_chars=100).run("x")
        assert not result.success
        assert "output budget" in result.error

    def test_budget_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_output_chars"):
            AgentConfig(max_output_chars=0)


# ── bounded repair re-prompt ─────────────────────────────────────────────────
class TestRepairReprompt:
    def test_malformed_output_repaired_within_default_retries(self) -> None:
        result = _agent(["not json at all", _final("recovered")]).run("x")
        assert result.success, result.error
        assert any(s.kind == "repair" for s in result.steps)

    def test_repair_exhausted_captures_parse_error(self) -> None:
        result = _agent(["bad one", "still bad"], max_retries=1).run("x")
        assert not result.success
        assert "not valid JSON" in result.error

    def test_script_exhaustion_after_repair_captured(self) -> None:
        result = _agent(["bad one"]).run("x")
        assert not result.success
        assert "script exhausted" in result.error

    def test_retries_must_be_non_negative(self) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            AgentConfig(max_retries=-1)
