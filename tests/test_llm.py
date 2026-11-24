"""Unit tests for the LLM protocol, decision parsing, and ScriptedLLM."""

from __future__ import annotations

import json

import pytest

from tool_agent.llm import LLMError, ScriptedLLM, parse_decision
from tool_agent.models import ToolCall


class TestParseDecision:
    def test_valid_tool_call(self) -> None:
        decision = parse_decision(
            json.dumps({"action": "tool", "tool": "calculator", "arguments": {"expression": "1+1"}})
        )
        assert decision == ToolCall(tool="calculator", arguments={"expression": "1+1"})

    def test_tool_call_defaults_empty_arguments(self) -> None:
        decision = parse_decision(json.dumps({"action": "tool", "tool": "kb_search"}))
        assert decision == ToolCall(tool="kb_search", arguments={})

    def test_valid_final(self) -> None:
        decision = parse_decision(json.dumps({"action": "final", "answer": "42"}))
        assert decision == ("final", "42")

    def test_invalid_json_rejected(self) -> None:
        with pytest.raises(LLMError, match="not valid JSON"):
            parse_decision("let me think about this...")

    def test_non_object_rejected(self) -> None:
        with pytest.raises(LLMError, match="JSON object"):
            parse_decision('["action", "tool"]')

    def test_unknown_action_rejected(self) -> None:
        with pytest.raises(LLMError, match="unknown action"):
            parse_decision(json.dumps({"action": "ponder"}))

    def test_tool_without_name_rejected(self) -> None:
        with pytest.raises(LLMError, match="non-empty 'tool'"):
            parse_decision(json.dumps({"action": "tool", "arguments": {}}))

    def test_final_without_answer_rejected(self) -> None:
        with pytest.raises(LLMError, match="non-empty 'answer'"):
            parse_decision(json.dumps({"action": "final", "answer": "  "}))

    def test_non_dict_arguments_rejected(self) -> None:
        with pytest.raises(LLMError, match="arguments"):
            parse_decision(json.dumps({"action": "tool", "tool": "calc", "arguments": [1, 2]}))


class TestScriptedLLM:
    def test_replays_in_order(self) -> None:
        llm = ScriptedLLM(script=['{"action": "final", "answer": "ok"}'])
        assert llm.decide([]) == '{"action": "final", "answer": "ok"}'
        assert llm.remaining == 0

    def test_exhausted_script_raises(self) -> None:
        llm = ScriptedLLM(script=['{"action": "final", "answer": "ok"}'])
        llm.decide([])
        with pytest.raises(LLMError, match="script exhausted"):
            llm.decide([])

    def test_empty_script_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one decision"):
            ScriptedLLM(script=[])


class TestAzureLLMGuardrails:
    def test_missing_credentials_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
        pytest.importorskip("openai")
        from tool_agent.llm import AzureOpenAILLM

        with pytest.raises(LLMError, match="AZURE_OPENAI_ENDPOINT"):
            AzureOpenAILLM()
