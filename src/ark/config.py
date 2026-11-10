"""Run configuration: which agent to build, on which model, with which defences."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    adapter: str = Field(description="'openai_loop', 'langgraph' or 'package.module:factory'.")
    model: str = ""
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_steps: int = 8
    defences: list[str] = Field(default_factory=list)
    cache_dir: str = ".ark_cache"
    min_interval_s: float = 0.0
    max_retries: int = 8
    request_timeout_s: float = 120.0
    options: dict[str, Any] = Field(default_factory=dict)

    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise RuntimeError(
                f"environment variable {self.api_key_env} is not set "
                f"(config {self.name!r}); export it or pass --env-file"
            )
        return key


def load_config(path: str | Path) -> RunConfig:
    return RunConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def load_env_file(path: str | Path) -> None:
    """Load KEY=VALUE lines into the environment without overriding what is already set."""
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip().removeprefix("export ").strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
