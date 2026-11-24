"""Safe tools the agent can call, plus the ToolRegistry.

Every tool is a pure function with an explicit `ToolSpec` (name,
description, typed parameters). The registry validates call arguments
against the spec before execution - the agent can never invoke a tool
that doesn't exist or with arguments of the wrong shape.

Included tools:
* calculator     - arithmetic expressions (eval'd on a whitelisted AST)
* unit_convert   - km<->mi, kg<->lb, c<->f
* kb_search      - retrieval over a TF-IDF knowledge base (retrieval.py)

The agent itself never imports tool implementations directly - it only
sees the registry (dependency inversion).
"""

from __future__ import annotations

import ast
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tool_agent.models import ToolCall, ToolSpec
from tool_agent.retrieval import TfIdfStore


class ToolError(Exception):
    """Raised for unknown tools or invalid arguments."""


def _add(a: float, b: float) -> float:
    return a + b


def _sub(a: float, b: float) -> float:
    return a - b


def _mul(a: float, b: float) -> float:
    return a * b


def _truediv(a: float, b: float) -> float:
    return a / b


def _pow(a: float, b: float) -> float:
    return a**b


def _mod(a: float, b: float) -> float:
    return a % b


def _floordiv(a: float, b: float) -> float:
    return a // b


_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: _add,
    ast.Sub: _sub,
    ast.Mult: _mul,
    ast.Div: _truediv,
    ast.Pow: _pow,
    ast.Mod: _mod,
    ast.FloorDiv: _floordiv,
}

_BINOPS: dict[str, Callable[..., float]] = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": lambda x: abs(x),
    "round": lambda x, ndigits=0: round(x, int(ndigits)),
}

_MULTIOPS: dict[str, Callable[..., float]] = {
    "min": min,
    "max": max,
}


def safe_eval(expression: str) -> float:
    """Evaluate arithmetic on a whitelisted AST - no names, no attributes,
    no subscripts; anything else raises ToolError."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        msg = f"invalid expression: {expression!r}"
        raise ToolError(msg) from exc

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.BinOp):
            op = _OPS.get(type(node.op))
            if op is not None:
                return op(visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
            value = visit(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            args = [visit(a) for a in node.args]
            if name in _BINOPS and args and node.keywords == []:
                return float(_BINOPS[name](*args))
            if name in _MULTIOPS and args and node.keywords == []:
                return float(_MULTIOPS[name](*args))
        msg = f"disallowed syntax in expression: {expression!r}"
        raise ToolError(msg)

    return visit(tree)


_UNIT_FACTORS_TO_KM = {"km": 1.0, "mi": 1.609344, "m": 0.001}
_MASS_TO_KG = {"kg": 1.0, "lb": 0.45359237}


def unit_convert(value: float, unit: str, target: str) -> float:
    """Convert within a dimension (length km/mi/m or mass kg/lb; temp C/F)."""
    unit, target = unit.lower(), target.lower()
    if unit in _UNIT_FACTORS_TO_KM and target in _UNIT_FACTORS_TO_KM:
        return value * _UNIT_FACTORS_TO_KM[unit] / _UNIT_FACTORS_TO_KM[target]
    if unit in _MASS_TO_KG and target in _MASS_TO_KG:
        return value * _MASS_TO_KG[unit] / _MASS_TO_KG[target]
    if {unit, target} == {"c", "f"}:
        return value * 9 / 5 + 32 if unit == "c" else (value - 32) * 5 / 9
    msg = f"unsupported conversion: {unit!r} -> {target!r}"
    raise ToolError(msg)


@dataclass
class ToolRegistry:
    """Validated tool dispatch: the only surface the agent sees."""

    _tools: dict[str, tuple[ToolSpec, Callable[..., Any]]]

    def __init__(self) -> None:
        self._tools = {}

    def register(self, spec: ToolSpec, fn: Callable[..., Any], *, replace: bool = False) -> None:
        if not replace and spec.name in self._tools:
            msg = f"tool {spec.name!r} already registered"
            raise ToolError(msg)
        self._tools[spec.name] = (spec, fn)

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(spec for spec, _ in self._tools.values())

    def system_prompt(self) -> str:
        """Render tool documentation for the system message."""
        lines = ["You can call these tools, one at a time:"]
        for spec, _ in self._tools.values():
            params = ", ".join(f"{k}: {v}" for k, v in spec.parameters.items()) or "none"
            lines.append(
                f"- {spec.name}({params}): {spec.description}\n"
                f'  respond with {{"action": "tool", "tool": "{spec.name}", '
                '"arguments": {...}}}'
            )
        lines.append(
            'When you have enough information, respond with {"action": "final", "answer": "..."}.'
        )
        return "\n".join(lines)

    def dispatch(self, call: ToolCall) -> Any:
        """Validate + execute one tool call. Never trusts the LLM blindly."""
        if call.tool not in self._tools:
            msg = f"unknown tool: {call.tool!r} (known: {sorted(self._tools)})"
            raise ToolError(msg)
        spec, fn = self._tools[call.tool]
        unexpected = set(call.arguments) - set(spec.parameters)
        if unexpected:
            msg = (
                f"tool {spec.name}: unexpected arguments "
                f"{sorted(unexpected)} (expected {sorted(spec.parameters)})"
            )
            raise ToolError(msg)
        missing = set(spec.parameters) - set(call.arguments)
        if missing:
            msg = f"tool {spec.name}: missing arguments {sorted(missing)}"
            raise ToolError(msg)
        return fn(**call.arguments)


def default_registry(store: TfIdfStore) -> ToolRegistry:
    """The standard tool set, wired to a knowledge base."""
    registry = ToolRegistry()
    calculator_description = (
        "Evaluate an arithmetic expression (numbers, + - * / % // **, "
        "sqrt, abs, round, min, max, log, exp)."
    )
    registry.register(
        ToolSpec(
            name="calculator",
            description=calculator_description,
            parameters={"expression": "string, arithmetic expression"},
        ),
        safe_eval,
    )
    registry.register(
        ToolSpec(
            name="unit_convert",
            description="Convert a value between units: km<->mi<->m, kg<->lb, c<->f.",
            parameters={"value": "number", "unit": "source unit", "target": "target unit"},
        ),
        unit_convert,
    )
    kb_description = (
        "Search the supplier-compliance knowledge base and get the top matching passages."
    )
    registry.register(
        ToolSpec(
            name="kb_search",
            description=kb_description,
            parameters={
                "query": "string, what to look up",
                "top_k": "int, how many passages (default 3)",
            },
        ),
        store.search,
    )
    return registry
