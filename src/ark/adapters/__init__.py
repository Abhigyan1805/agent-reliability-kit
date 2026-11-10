"""Agent adapters: how the kit drives a particular agent implementation."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .base import Agent, FunctionAgent, Session

if TYPE_CHECKING:
    from ..config import RunConfig

__all__ = ["Agent", "FunctionAgent", "Session", "build_agent"]


def build_agent(config: RunConfig) -> Agent:
    """Construct the agent a configuration names, importing optional dependencies lazily."""
    if config.adapter == "openai_loop":
        from .openai_loop import OpenAILoopAgent

        return OpenAILoopAgent(config)
    if config.adapter == "langgraph":
        from .langgraph_agent import LangGraphAgent

        return LangGraphAgent(config)
    if ":" in config.adapter:
        module_name, _, attribute = config.adapter.partition(":")
        return getattr(importlib.import_module(module_name), attribute)(config)
    raise ValueError(f"unknown adapter {config.adapter!r}")
