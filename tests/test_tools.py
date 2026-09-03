"""Unit tests for tools: calculator safety, conversions, registry validation."""

from __future__ import annotations

import pytest

from tool_agent.models import ToolCall, ToolSpec
from tool_agent.retrieval import default_store
from tool_agent.tools import ToolError, ToolRegistry, default_registry, safe_eval, unit_convert


class TestSafeEval:
    def test_basic_arithmetic(self) -> None:
        assert safe_eval("2 + 3 * 4") == 14.0
        assert safe_eval("10 / 4") == 2.5
        assert safe_eval("2 ** 10") == 1024.0
        assert safe_eval("7 // 2") == 3.0
        assert safe_eval("7 % 3") == 1.0

    def test_unary_and_functions(self) -> None:
        assert safe_eval("-5 + abs(3 - 8)") == 0.0
        assert safe_eval("round(2.567, 2)") == 2.57
        assert safe_eval("sqrt(16) + min(3, 9)") == 7.0
        assert safe_eval("max(1, 2, 3) * log(exp(1))") == pytest.approx(3.0)

    def test_names_rejected(self) -> None:
        with pytest.raises(ToolError, match="disallowed"):
            safe_eval("__import__('os').system('dir')")

    def test_attributes_rejected(self) -> None:
        with pytest.raises(ToolError, match="disallowed"):
            safe_eval("(1).__class__")

    def test_syntax_error_rejected(self) -> None:
        with pytest.raises(ToolError, match="invalid expression"):
            safe_eval("2 +* 3")

    def test_keywords_rejected(self) -> None:
        with pytest.raises(ToolError, match="disallowed"):
            safe_eval("[x for x in range(3)]")


class TestUnitConvert:
    def test_length(self) -> None:
        assert unit_convert(100, "mi", "km") == pytest.approx(160.9344)
        assert unit_convert(5, "km", "m") == pytest.approx(5000.0)

    def test_mass(self) -> None:
        assert unit_convert(1, "kg", "lb") == pytest.approx(2.20462, rel=1e-4)

    def test_temperature(self) -> None:
        assert unit_convert(100, "c", "f") == pytest.approx(212.0)
        assert unit_convert(32, "f", "c") == pytest.approx(0.0)

    def test_dimension_mismatch_rejected(self) -> None:
        with pytest.raises(ToolError, match="unsupported conversion"):
            unit_convert(10, "kg", "km")

    def test_unknown_unit_rejected(self) -> None:
        with pytest.raises(ToolError, match="unsupported conversion"):
            unit_convert(10, "stones", "kg")


class TestToolRegistry:
    def test_register_and_specs(self) -> None:
        registry = ToolRegistry()
        registry.register(
            ToolSpec(name="echo", description="echoes", parameters={"text": "string"}),
            lambda text: text,
        )
        (spec,) = registry.specs()
        assert spec.name == "echo"
        assert "echo" in registry.system_prompt()

    def test_duplicate_registration_rejected(self) -> None:
        registry = ToolRegistry()
        registry.register(ToolSpec(name="echo", description="echoes", parameters={}), lambda: 1)
        with pytest.raises(ToolError, match="already registered"):
            registry.register(ToolSpec(name="echo", description="again", parameters={}), lambda: 1)

    def test_dispatch_unknown_tool(self) -> None:
        registry = default_registry(default_store())
        with pytest.raises(ToolError, match="unknown tool"):
            registry.dispatch(ToolCall(tool="brain_surgery", arguments={}))

    def test_dispatch_missing_argument(self) -> None:
        registry = default_registry(default_store())
        with pytest.raises(ToolError, match="missing arguments"):
            registry.dispatch(ToolCall(tool="calculator", arguments={}))

    def test_dispatch_unexpected_argument(self) -> None:
        registry = default_registry(default_store())
        with pytest.raises(ToolError, match="unexpected arguments"):
            registry.dispatch(
                ToolCall(tool="calculator",
                         arguments={"expression": "1+1", "extra": "nope"})
            )

    def test_dispatch_success(self) -> None:
        registry = default_registry(default_store())
        result = registry.dispatch(
            ToolCall(tool="calculator", arguments={"expression": "6 * 7"})
        )
        assert result == 42.0

    def test_tool_spec_validation(self) -> None:
        with pytest.raises(ValueError, match="description"):
            ToolSpec(name="x", description="  ", parameters={})
        with pytest.raises(ValueError, match="snake_case"):
            ToolSpec(name="Bad Name!", description="d", parameters={})
