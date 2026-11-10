"""Fault injection: what can go wrong between an agent and its tools.

Each fault is a pure transformation of a :class:`ToolOutcome`, so faults compose and can be
unit-tested without an agent or a model.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .matching import match_where, render


class ToolError(Exception):
    """A tool invocation raised instead of returning. Adapters must handle this."""


class ToolTimeout(ToolError):
    """A tool invocation exceeded its deadline."""


@dataclass
class ToolOutcome:
    """The result of a tool call as it passes through the fault chain."""

    value: Any
    raises: type[ToolError] | None = None
    message: str | None = None
    simulated_ms: int = 0
    real_delay_s: float = 0.0
    applied: list[str] = field(default_factory=list)


class FaultBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    calls: list[int] | None = Field(
        default=None, description="1-based call numbers of this tool to affect; omit for all."
    )
    when: dict[str, Any] | None = Field(
        default=None, description="Only affect calls whose arguments match this clause."
    )

    def applies(self, call_number: int, args: dict[str, Any]) -> bool:
        if self.calls is not None and call_number not in self.calls:
            return False
        return match_where(self.when, args)

    def label(self) -> str:
        return getattr(self, "type", "fault")


def _split_path(path: str) -> list[str | int]:
    return [int(p) if p.isdigit() else p for p in path.split(".")]


def _get_container(root: Any, parts: list[str | int]) -> Any:
    node = root
    for part in parts[:-1]:
        node = node[part]
    return node


def _insert_text(original: str, payload: str, position: str) -> str:
    if position == "prepend":
        return f"{payload}\n{original}"
    if position == "replace":
        return payload
    if position == "middle":
        cut = original.find(". ", len(original) // 2)
        cut = len(original) // 2 if cut == -1 else cut + 2
        return f"{original[:cut]}{payload} {original[cut:]}"
    return f"{original}\n{payload}"


class InjectFault(FaultBase):
    """Plant attacker-controlled text in a tool result, inside a string or a named field."""

    type: Literal["inject"]
    payload: str
    position: Literal["append", "prepend", "middle", "replace"] = "append"
    field: str | None = Field(default=None, description="Dotted path, e.g. 'results.0.text'.")

    def apply(self, outcome: ToolOutcome, seed: int) -> ToolOutcome:
        value = copy.deepcopy(outcome.value)
        if self.field is None:
            if not isinstance(value, str):
                raise ValueError(
                    f"inject on tool {self.tool!r}: result is structured, so 'field' is required"
                )
            outcome.value = _insert_text(value, self.payload, self.position)
            return outcome
        parts = _split_path(self.field)
        container = _get_container(value, parts)
        container[parts[-1]] = _insert_text(
            render(container[parts[-1]]), self.payload, self.position
        )
        outcome.value = value
        return outcome


class ErrorFault(FaultBase):
    """Fail the call: as an error payload, or by raising inside the agent's tool executor."""

    type: Literal["error"]
    message: str = "Internal server error"
    code: int = 500
    raises: bool = False

    def apply(self, outcome: ToolOutcome, seed: int) -> ToolOutcome:
        if self.raises:
            outcome.raises, outcome.message = ToolError, self.message
        else:
            outcome.value = {"error": self.message, "code": self.code}
        return outcome


class TimeoutFault(FaultBase):
    """Fail the call with a deadline error. Latency is simulated, so the suite stays fast."""

    type: Literal["timeout"]
    simulated_ms: int = 30_000
    message: str = "Tool call timed out"

    def apply(self, outcome: ToolOutcome, seed: int) -> ToolOutcome:
        outcome.raises, outcome.message = ToolTimeout, self.message
        outcome.simulated_ms += self.simulated_ms
        return outcome


class SlowFault(FaultBase):
    """Deliver the correct result late. ``real_delay_s`` is a true sleep, capped by the runner."""

    type: Literal["slow"]
    simulated_ms: int = 20_000
    real_delay_s: float = 0.0

    def apply(self, outcome: ToolOutcome, seed: int) -> ToolOutcome:
        outcome.simulated_ms += self.simulated_ms
        outcome.real_delay_s += self.real_delay_s
        return outcome


class CorruptFault(FaultBase):
    """Damage the result: cut JSON, binary noise, nulled fields or a stringified shape change."""

    type: Literal["corrupt"]
    mode: Literal["truncate", "garbage", "null_fields", "wrong_type"]
    fraction: float = Field(default=0.5, gt=0, lt=1)

    def apply(self, outcome: ToolOutcome, seed: int) -> ToolOutcome:
        if self.mode == "truncate":
            text = render(outcome.value)
            outcome.value = text[: int(len(text) * self.fraction)]
        elif self.mode == "garbage":
            rng = random.Random(seed)
            noise = bytes(rng.randrange(256) for _ in range(120))
            outcome.value = noise.decode("latin-1")
        elif self.mode == "null_fields":
            outcome.value = _null_leaves(outcome.value)
        else:
            outcome.value = [render(outcome.value)]
        return outcome


def _null_leaves(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _null_leaves(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_null_leaves(v) for v in value]
    return None


class ReplaceFault(FaultBase):
    """Return a plausible but wrong value: a lying or poisoned tool."""

    type: Literal["replace"]
    value: Any

    def apply(self, outcome: ToolOutcome, seed: int) -> ToolOutcome:
        outcome.value = copy.deepcopy(self.value)
        return outcome


class EmptyFault(FaultBase):
    """Return nothing useful."""

    type: Literal["empty"]
    value: Any = None

    def apply(self, outcome: ToolOutcome, seed: int) -> ToolOutcome:
        outcome.value = copy.deepcopy(self.value)
        return outcome


Fault = Annotated[
    InjectFault | ErrorFault | TimeoutFault | SlowFault | CorruptFault | ReplaceFault | EmptyFault,
    Field(discriminator="type"),
]


def stable_seed(*parts: object) -> int:
    """A seed that is the same across processes, unlike ``hash()``."""
    digest = hashlib.sha256(json.dumps(parts, default=str).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def apply_faults(
    faults: list[Any],
    tool: str,
    call_number: int,
    args: dict[str, Any],
    outcome: ToolOutcome,
    seed: int,
) -> ToolOutcome:
    """Run every matching fault, in scenario order, over a tool outcome."""
    for fault in faults:
        if fault.tool == tool and fault.applies(call_number, args):
            outcome = fault.apply(outcome, seed)
            outcome.applied.append(fault.label())
    return outcome
