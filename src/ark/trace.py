"""The record of one agent run. Checks read only this, never the agent."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """One invocation that reached the simulated tool environment."""

    index: int
    turn: int
    tool: str
    args: dict[str, Any]
    call_number: int = 0
    result: str | None = None
    error: str | None = None
    faults: list[str] = Field(default_factory=list)
    simulated_ms: int = 0
    invalid_args: str | None = None
    unknown_tool: bool = False


class DefenceEvent(BaseModel):
    """A call or result a defence blocked or rewrote before it reached the tool or the agent."""

    turn: int
    defence: str
    tool: str
    action: str
    detail: str = ""


class Trace(BaseModel):
    scenario_id: str
    category: str
    config: str = ""
    trial: int = 0
    system: str = ""
    user_turns: list[str] = Field(default_factory=list)
    answers: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    defence_events: list[DefenceEvent] = Field(default_factory=list)
    crash: str | None = None
    infra_error: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    duration_s: float = 0.0

    @property
    def final_answer(self) -> str:
        return self.answers[-1] if self.answers else ""

    def calls_to(self, tool: str) -> list[ToolCall]:
        if tool == "*":
            return [c for c in self.tool_calls if not c.unknown_tool]
        return [c for c in self.tool_calls if c.tool == tool]
