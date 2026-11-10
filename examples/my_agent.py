"""Minimal example of putting your own agent under test.

Replace ``naive_agent`` with a call into your real agent. It receives the task (system prompt and
tools) and the conversation so far, and returns its answer. Call tools with ``task.call_tool``, so
that the kit can inject faults and record what happened.

    ark validate --suite examples/my_scenario.yaml
    ark run --suite examples/my_scenario.yaml --config examples/my_agent.yaml --fresh
"""

from __future__ import annotations

from typing import Any

from ark.adapters import FunctionAgent
from ark.environment import AgentTask
from ark.faults import ToolError


def naive_agent(task: AgentTask, history: list[dict[str, str]]) -> str:
    """Looks a ticket up and repeats it. No model is involved, so this runs offline."""
    try:
        ticket = task.call_tool("get_ticket", {"ticket_id": "T-1"})
    except ToolError as exc:
        return f"The ticket system failed: {exc}"
    return f"Here is the ticket: {ticket}"


def build(config: Any) -> FunctionAgent:
    """The factory named by ``adapter:`` in the run configuration."""
    return FunctionAgent(naive_agent)
