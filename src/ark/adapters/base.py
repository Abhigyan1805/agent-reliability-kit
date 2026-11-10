"""The contract between the kit and an agent."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from ..environment import MALFORMED_KEY, AgentTask


@runtime_checkable
class Session(Protocol):
    """One conversation. ``send`` handles a user turn, including any tool calls, and answers."""

    def send(self, message: str) -> str: ...


@runtime_checkable
class Agent(Protocol):
    def new_session(self, task: AgentTask) -> Session: ...


def parse_tool_args(raw: str | None) -> dict[str, Any]:
    """Decode model-written JSON arguments, keeping bad JSON visible to the trace."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {MALFORMED_KEY: raw}
    return parsed if isinstance(parsed, dict) else {MALFORMED_KEY: raw}


class _FunctionSession:
    def __init__(self, fn: Callable[..., str], task: AgentTask) -> None:
        self.fn = fn
        self.task = task
        self.history: list[dict[str, str]] = []

    def send(self, message: str) -> str:
        self.history.append({"role": "user", "content": message})
        answer = self.fn(self.task, self.history)
        self.history.append({"role": "assistant", "content": answer})
        return answer


class FunctionAgent:
    """Adapt a plain function ``fn(task, history) -> answer`` into an :class:`Agent`."""

    def __init__(self, fn: Callable[..., str]) -> None:
        self.fn = fn

    def new_session(self, task: AgentTask) -> Session:
        return _FunctionSession(self.fn, task)
