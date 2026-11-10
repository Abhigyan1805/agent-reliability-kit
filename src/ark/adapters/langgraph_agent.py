"""Drive a LangGraph agent. Bring your own graph with ``options.graph_factory``."""

from __future__ import annotations

import importlib
from collections import Counter
from collections.abc import Callable
from typing import Any

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from ..config import RunConfig
from ..environment import AgentTask, ToolHandle
from .openai_loop import build_transport


def default_graph(model: ChatOpenAI, tools: list[BaseTool]) -> Any:
    """A standard model-then-tools loop, written out so it survives prebuilt API changes."""
    bound = model.bind_tools(tools) if tools else model

    def call_model(state: MessagesState) -> dict[str, list[Any]]:
        return {"messages": [bound.invoke(state["messages"])]}

    def route(state: MessagesState) -> str:
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route, ["tools", END])
    graph.add_edge("tools", "model")
    return graph.compile()


def _as_langchain_tool(handle: ToolHandle) -> StructuredTool:
    def invoke(**kwargs: Any) -> str:
        return handle.fn(kwargs)

    return StructuredTool.from_function(
        func=invoke,
        name=handle.name,
        description=handle.description,
        args_schema=handle.parameters,
    )


def _text(message: Any) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


class _GraphSession:
    def __init__(self, agent: LangGraphAgent, task: AgentTask) -> None:
        self.agent = agent
        self.graph = agent.graph_factory(agent.model, [_as_langchain_tool(t) for t in task.tools])
        self.messages: list[Any] = [SystemMessage(task.system)]
        self.usage: Counter[str] = Counter()

    def send(self, message: str) -> str:
        self.messages.append(HumanMessage(message))
        limit = self.agent.config.max_steps * 2 + 1
        try:
            state = self.graph.invoke(
                {"messages": self.messages}, config={"recursion_limit": limit}
            )
        except GraphRecursionError:
            return ""
        self.messages = state["messages"]
        ai_messages = [m for m in self.messages if isinstance(m, AIMessage)]
        for ai in ai_messages:
            meta = ai.usage_metadata or {}
            self.usage["prompt_tokens"] += meta.get("input_tokens", 0)
            self.usage["completion_tokens"] += meta.get("output_tokens", 0)
        return _text(ai_messages[-1]) if ai_messages else ""


class LangGraphAgent:
    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.transport = build_transport(config)
        self.model = ChatOpenAI(
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key(),
            temperature=config.temperature,
            max_retries=0,
            timeout=config.request_timeout_s,
            http_client=httpx.Client(transport=self.transport),
        )
        self.graph_factory: Callable[[Any, list[BaseTool]], Any] = default_graph
        reference = config.options.get("graph_factory")
        if reference:
            module_name, _, attribute = str(reference).partition(":")
            self.graph_factory = getattr(importlib.import_module(module_name), attribute)

    def new_session(self, task: AgentTask) -> _GraphSession:
        self.transport.salt = f"trial-{task.trial}" if task.trial else ""
        return _GraphSession(self, task)
