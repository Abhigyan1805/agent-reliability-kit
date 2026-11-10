"""The scenario format: a task, its tools, the faults to inject and the rules that score it."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .checks import CheckSpec
from .faults import Fault

CATEGORY_ORDER = [
    "direct_injection",
    "indirect_injection",
    "tool_failure",
    "exfiltration",
    "excessive_action",
    "refusal_calibration",
    "malformed_input",
]


class _ScenarioLoader(yaml.SafeLoader):
    """A YAML loader that keeps dates as text, so an impossible date is just a string."""


_ScenarioLoader.yaml_implicit_resolvers = {
    key: [r for r in resolvers if r[0] != "tag:yaml.org,2002:timestamp"]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


class ToolResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    when: dict[str, Any] | None = None
    returns: Any = None


class ToolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    risk: Literal["read", "write", "destructive", "egress"] = "read"
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    responses: list[ToolResponse] = Field(default_factory=list)
    default: Any = {"status": "ok"}

    @field_validator("parameters", mode="before")
    @classmethod
    def _expand_shorthand(cls, value: Any) -> Any:
        """Accept ``{name: string, limit: integer?}`` as well as a full JSON Schema."""
        if not isinstance(value, dict) or "type" in value or "properties" in value:
            return value
        properties = {k: {"type": str(v).rstrip("?")} for k, v in value.items()}
        required = [k for k, v in value.items() if not str(v).endswith("?")]
        return {"type": "object", "properties": properties, "required": required}


class ScriptedCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class ScriptedTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calls: list[ScriptedCall] = Field(default_factory=list)
    answer: str = ""


class Golden(BaseModel):
    """Hand-written runs that prove a scenario's checks can tell good behaviour from bad."""

    model_config = ConfigDict(extra="forbid")

    good: list[ScriptedTurn]
    bad: list[ScriptedTurn]


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    category: str
    title: str
    system: str = "You are a helpful assistant with access to tools."
    user: list[str]
    tools: list[ToolSpec] = Field(default_factory=list)
    faults: list[Fault] = Field(default_factory=list)
    checks: list[CheckSpec]
    golden: Golden
    refusal_ok: bool = Field(
        default=False, description="A plain refusal is correct behaviour for this scenario."
    )
    tags: list[str] = Field(default_factory=list)

    @field_validator("user", mode="before")
    @classmethod
    def _coerce_user(cls, value: Any) -> Any:
        return [value] if isinstance(value, str) else value

    @field_validator("id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", value):
            raise ValueError(f"scenario id must be lowercase and file-safe, got {value!r}")
        return value

    @model_validator(mode="after")
    def _cross_references(self) -> Scenario:
        names = [t.name for t in self.tools]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.id}: duplicate tool names")
        for fault in self.faults:
            if fault.tool not in names:
                raise ValueError(f"{self.id}: fault targets unknown tool {fault.tool!r}")
        if not self.checks:
            raise ValueError(f"{self.id}: a scenario needs at least one check")
        return self


class ScenarioError(ValueError):
    """A scenario file failed to load or validate."""


def _load_file(path: Path) -> list[Scenario]:
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_ScenarioLoader)  # noqa: S506
    except yaml.YAMLError as exc:
        raise ScenarioError(f"{path}: invalid YAML: {exc}") from exc
    items = data.get("scenarios", [data]) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ScenarioError(f"{path}: expected a scenario, a list, or a 'scenarios' key")
    scenarios: list[Scenario] = []
    for position, item in enumerate(items):
        try:
            scenarios.append(Scenario.model_validate(item))
        except ValueError as exc:
            label = item.get("id", position) if isinstance(item, dict) else position
            raise ScenarioError(f"{path}: scenario {label}: {exc}") from exc
    return scenarios


def load_scenarios(path: str | Path) -> list[Scenario]:
    """Load one scenario file, or every ``*.yaml``/``*.yml`` file under a directory."""
    root = Path(path)
    if not root.exists():
        raise ScenarioError(f"{root}: no such file or directory")
    files = (
        sorted(p for p in root.rglob("*") if p.suffix in {".yaml", ".yml"})
        if root.is_dir()
        else [root]
    )
    scenarios = [s for file in files for s in _load_file(file)]
    seen: set[str] = set()
    for scenario in scenarios:
        if scenario.id in seen:
            raise ScenarioError(f"duplicate scenario id: {scenario.id}")
        seen.add(scenario.id)
    return scenarios
