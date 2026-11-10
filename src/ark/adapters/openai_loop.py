"""A plain function-calling loop over any OpenAI-compatible chat completions endpoint."""

from __future__ import annotations

from collections import Counter
from typing import Any

import httpx
from openai import OpenAI

from ..config import RunConfig
from ..environment import AgentTask
from ..faults import ToolError
from ..llm.transport import CachingTransport
from .base import parse_tool_args


def build_transport(config: RunConfig) -> CachingTransport:
    return CachingTransport(
        cache_dir=config.cache_dir,
        min_interval_s=config.min_interval_s,
        max_retries=config.max_retries,
    )


class _LoopSession:
    def __init__(self, agent: OpenAILoopAgent, task: AgentTask) -> None:
        self.agent = agent
        self.task = task
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": task.system}]
        self.usage: Counter[str] = Counter()

    def _complete(self) -> Any:
        extra: dict[str, Any] = {}
        if self.task.tools:
            extra["tools"] = [t.openai_schema() for t in self.task.tools]
        response = self.agent.client.chat.completions.create(
            model=self.agent.config.model,
            messages=self.messages,
            temperature=self.agent.config.temperature,
            **extra,
        )
        if response.usage:
            self.usage["prompt_tokens"] += response.usage.prompt_tokens or 0
            self.usage["completion_tokens"] += response.usage.completion_tokens or 0
        self.usage["llm_calls"] += 1
        return response.choices[0].message

    def _execute(self, call: Any) -> str:
        args = parse_tool_args(call.function.arguments)
        try:
            return self.task.call_tool(call.function.name, args)
        except ToolError as exc:
            return f"Tool error: {exc}"

    def send(self, message: str) -> str:
        self.messages.append({"role": "user", "content": message})
        last_text = ""
        for _ in range(self.agent.config.max_steps):
            reply = self._complete()
            self.messages.append(reply.model_dump(exclude_none=True))
            last_text = reply.content or last_text
            if not reply.tool_calls:
                return reply.content or ""
            for call in reply.tool_calls:
                self.messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": self._execute(call)}
                )
        return last_text


class OpenAILoopAgent:
    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.transport = build_transport(config)
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key(),
            max_retries=0,
            timeout=config.request_timeout_s,
            http_client=httpx.Client(transport=self.transport),
        )

    def new_session(self, task: AgentTask) -> _LoopSession:
        self.transport.salt = f"trial-{task.trial}" if task.trial else ""
        return _LoopSession(self, task)
