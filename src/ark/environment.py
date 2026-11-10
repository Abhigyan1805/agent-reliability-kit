"""The simulated tool world an agent runs in. Every call is recorded, faults are applied here."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .faults import ToolError, ToolOutcome, apply_faults, stable_seed
from .matching import match_where, render
from .scenario import Scenario, ToolSpec
from .trace import DefenceEvent, ToolCall, Trace

MALFORMED_KEY = "__malformed__"


@dataclass
class ToolHandle:
    """What an agent sees of a tool: its contract and a callable returning a string."""

    name: str
    description: str
    parameters: dict[str, Any]
    risk: str
    fn: Callable[[dict[str, Any]], str]

    def __call__(self, **args: Any) -> str:
        return self.fn(args)

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class AgentTask:
    """Everything an agent is allowed to know about a scenario. It contains no checks."""

    scenario_id: str
    system: str
    tools: list[ToolHandle]
    env: ToolEnv
    trial: int = 0
    user_messages: list[str] = field(default_factory=list)

    def call_tool(self, name: str, args: dict[str, Any]) -> str:
        """Invoke a tool by name. A name the agent invented is recorded and answered with an error.

        Raises :class:`ark.faults.ToolError` when a fault makes the tool raise.
        """
        for tool in self.tools:
            if tool.name == name:
                return tool.fn(args)
        return self.env.record_unknown(name, args)


class ToolEnv:
    """Executes stubbed tools for one run and writes each call to the trace."""

    def __init__(
        self,
        scenario: Scenario,
        trace: Trace,
        max_real_delay_s: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.scenario = scenario
        self.trace = trace
        self.turn = 0
        self.max_real_delay_s = max_real_delay_s
        self._sleep = sleep
        self._counts: Counter[str] = Counter()
        self._specs = {t.name: t for t in scenario.tools}

    def handles(self) -> list[ToolHandle]:
        return [self._handle(spec) for spec in self.scenario.tools]

    def _handle(self, spec: ToolSpec) -> ToolHandle:
        return ToolHandle(
            name=spec.name,
            description=spec.description,
            parameters=spec.parameters,
            risk=spec.risk,
            fn=lambda args: self.call(spec.name, args),
        )

    def note(self, defence: str, tool: str, action: str, detail: str = "") -> None:
        self.trace.defence_events.append(
            DefenceEvent(turn=self.turn, defence=defence, tool=tool, action=action, detail=detail)
        )

    def _record(self, tool: str, args: dict[str, Any], **fields: Any) -> ToolCall:
        call = ToolCall(
            index=len(self.trace.tool_calls), turn=self.turn, tool=tool, args=args, **fields
        )
        self.trace.tool_calls.append(call)
        return call

    def record_unknown(self, name: str, args: dict[str, Any]) -> str:
        result = render({"error": f"Unknown tool: {name}"})
        self._record(name, args, result=result, unknown_tool=True)
        return result

    def _base_value(self, spec: ToolSpec, args: dict[str, Any]) -> Any:
        for response in spec.responses:
            if match_where(response.when, args):
                return response.returns
        return spec.default

    @staticmethod
    def _validate(spec: ToolSpec, args: dict[str, Any]) -> str | None:
        if MALFORMED_KEY in args:
            return f"arguments were not valid JSON: {str(args[MALFORMED_KEY])[:120]}"
        try:
            validator = Draft202012Validator(spec.parameters)
            error = next(validator.iter_errors(args), None)
        except SchemaError as exc:
            raise ValueError(f"tool {spec.name}: invalid parameter schema: {exc}") from exc
        return error.message if error else None

    def call(self, name: str, args: dict[str, Any]) -> str:
        spec = self._specs.get(name)
        if spec is None:
            return self.record_unknown(name, args)
        self._counts[name] += 1
        number = self._counts[name]
        invalid = self._validate(spec, args)
        if invalid:
            result = render({"error": f"Invalid arguments: {invalid}", "code": 400})
            self._record(name, args, call_number=number, result=result, invalid_args=invalid)
            return result

        outcome = apply_faults(
            self.scenario.faults,
            name,
            number,
            args,
            ToolOutcome(value=self._base_value(spec, args)),
            stable_seed(self.scenario.id, name, number),
        )
        if outcome.real_delay_s:
            self._sleep(min(outcome.real_delay_s, self.max_real_delay_s))
        if outcome.raises is not None:
            self._record(
                name,
                args,
                call_number=number,
                error=outcome.message,
                faults=outcome.applied,
                simulated_ms=outcome.simulated_ms,
            )
            raise outcome.raises(outcome.message or "tool failed")
        result = render(outcome.value)
        self._record(
            name,
            args,
            call_number=number,
            result=result,
            faults=outcome.applied,
            simulated_ms=outcome.simulated_ms,
        )
        return result


__all__ = ["AgentTask", "MALFORMED_KEY", "ToolEnv", "ToolError", "ToolHandle"]
