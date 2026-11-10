"""Agents with no model: golden scripts and negative controls that validate the suite itself."""

from __future__ import annotations

from ..environment import AgentTask
from ..faults import ToolError
from ..scenario import ScriptedTurn


class _ScriptedSession:
    def __init__(self, turns: list[ScriptedTurn], task: AgentTask) -> None:
        self.turns = list(turns)
        self.task = task
        self.position = 0

    def send(self, message: str) -> str:
        if self.position >= len(self.turns):
            return ""
        turn = self.turns[self.position]
        self.position += 1
        for call in turn.calls:
            try:
                self.task.call_tool(call.tool, dict(call.args))
            except ToolError:
                continue
        return turn.answer


class ScriptedAgent:
    """Replays a fixed list of tool calls and answers through the real tool environment."""

    def __init__(self, turns: list[ScriptedTurn]) -> None:
        self.turns = turns

    def new_session(self, task: AgentTask) -> _ScriptedSession:
        return _ScriptedSession(self.turns, task)


class _ConstantSession:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    def send(self, message: str) -> str:
        return self.answer


class ConstantAgent:
    """Always gives the same answer and never calls a tool."""

    def __init__(self, answer: str) -> None:
        self.answer = answer

    def new_session(self, task: AgentTask) -> _ConstantSession:
        return _ConstantSession(self.answer)


NULL_AGENT = ConstantAgent("")
REFUSER_AGENT = ConstantAgent("I'm sorry, but I can't help with that request.")
