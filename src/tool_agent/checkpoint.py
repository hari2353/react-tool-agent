"""Durable agent checkpoints (LinkedIn portfolio project 5).

The loop persists a frozen snapshot after every completed step, so a crashed
or paused run resumes without repeating a tool call. JSON on disk, atomic
replace, explicit versioning.
"""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tool_agent.models import AgentStep, ToolCall

VERSION = 1


@dataclass(frozen=True)
class Checkpoint:
    version: int
    question: str
    conversation: tuple[dict[str, str], ...]
    steps: tuple[AgentStep, ...]
    calls: tuple[ToolCall, ...]
    step_no: int
    pending: tuple[ToolCall, ...]
    used_chars: int
    consecutive_errors: int
    raw: str
    assistant_echoed: bool


def _step_to_dict(step: AgentStep) -> dict[str, Any]:
    return {
        "step": step.step,
        "kind": step.kind,
        "tool": step.tool,
        "detail": step.detail,
        "tool_call": None
        if step.tool_call is None
        else {
            "tool": step.tool_call.tool,
            "arguments": step.tool_call.arguments,
        },
        "result": step.result,
    }


def _step_from_dict(payload: dict[str, Any]) -> AgentStep:
    call = payload.get("tool_call")
    return AgentStep(
        step=int(payload["step"]),
        kind=str(payload["kind"]),
        tool=payload.get("tool"),
        detail=str(payload.get("detail", "")),
        tool_call=None if call is None else ToolCall(call["tool"], call.get("arguments", {})),
        result=payload.get("result"),
    )


def _call_to_dict(call: ToolCall) -> dict[str, Any]:
    return {"tool": call.tool, "arguments": call.arguments}


def _call_from_dict(payload: dict[str, Any]) -> ToolCall:
    return ToolCall(payload["tool"], payload.get("arguments", {}))


def checkpoint_to_dict(checkpoint: Checkpoint) -> dict[str, Any]:
    return {
        "version": checkpoint.version,
        "question": checkpoint.question,
        "conversation": list(checkpoint.conversation),
        "steps": [_step_to_dict(s) for s in checkpoint.steps],
        "calls": [_call_to_dict(c) for c in checkpoint.calls],
        "step_no": checkpoint.step_no,
        "pending": [_call_to_dict(c) for c in checkpoint.pending],
        "used_chars": checkpoint.used_chars,
        "consecutive_errors": checkpoint.consecutive_errors,
        "raw": checkpoint.raw,
        "assistant_echoed": checkpoint.assistant_echoed,
    }


def checkpoint_from_dict(payload: dict[str, Any]) -> Checkpoint:
    if not isinstance(payload, dict):
        raise ValueError("corrupt checkpoint: expected an object")
    if payload.get("version") != VERSION:
        raise ValueError(
            f"unsupported checkpoint version: {payload.get('version')!r} (expected {VERSION})"
        )
    try:
        return Checkpoint(
            version=VERSION,
            question=str(payload["question"]),
            conversation=tuple(dict(m) for m in payload["conversation"]),
            steps=tuple(_step_from_dict(s) for s in payload["steps"]),
            calls=tuple(_call_from_dict(c) for c in payload["calls"]),
            step_no=int(payload["step_no"]),
            pending=tuple(_call_from_dict(c) for c in payload["pending"]),
            used_chars=int(payload["used_chars"]),
            consecutive_errors=int(payload["consecutive_errors"]),
            raw=str(payload["raw"]),
            assistant_echoed=bool(payload["assistant_echoed"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"corrupt checkpoint: {exc}") from exc


class JsonCheckpointStore:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    def save(self, checkpoint: Checkpoint) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(
            json.dumps(checkpoint_to_dict(checkpoint), default=str, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)

    def load(self) -> Checkpoint:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"corrupt checkpoint: {exc}") from exc
        return checkpoint_from_dict(payload)

    def clear(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()
