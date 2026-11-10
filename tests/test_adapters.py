"""Exercise the LLM adapters end to end against a fake OpenAI-compatible server."""

from __future__ import annotations

import json

import httpx
import pytest

from ark.adapters.langgraph_agent import LangGraphAgent
from ark.adapters.openai_loop import OpenAILoopAgent
from ark.config import RunConfig
from ark.runner import run_trial
from conftest import make_scenario


def text_reply(content):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def tool_reply(name, args, call_id="call_1"):
    arguments = args if isinstance(args, str) else json.dumps(args)
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                }
            }
        ]
    }


class FakeModel(httpx.BaseTransport):
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def handle_request(self, request):
        self.requests.append(json.loads(request.read()))
        reply = self.replies.pop(0)
        if isinstance(reply, int):
            return httpx.Response(reply, json={"error": {"message": "nope"}})
        return httpx.Response(200, json=reply)


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "unit-test-key")


def build(cls, replies, **overrides):
    config = RunConfig(
        name="test",
        adapter="x",
        model="fake",
        base_url="https://llm.example/v1/",
        api_key_env="TEST_KEY",
        cache_dir="",
        **overrides,
    )
    agent = cls(config)
    fake = FakeModel(replies)
    agent.transport.inner = fake
    agent.transport.cache_dir = None
    agent.transport.max_retries = 0
    agent.transport._sleep = lambda s: None
    return agent, fake


@pytest.mark.parametrize("cls", [OpenAILoopAgent, LangGraphAgent])
def test_tool_loop_records_call_and_returns_answer(cls):
    agent, fake = build(cls, [tool_reply("lookup", {"key": "a"}), text_reply("hello there")])
    result = run_trial(make_scenario(), agent)
    assert result.trace.final_answer == "hello there"
    assert [c.tool for c in result.trace.tool_calls] == ["lookup"]
    assert result.trace.tool_calls[0].args == {"key": "a"}
    tool_messages = [m for m in fake.requests[1]["messages"] if m["role"] == "tool"]
    assert "hello" in tool_messages[0]["content"]


@pytest.mark.parametrize("cls", [OpenAILoopAgent, LangGraphAgent])
def test_raised_tool_faults_are_reported_to_the_model(cls):
    agent, fake = build(cls, [tool_reply("lookup", {"key": "a"}), text_reply("it failed")])
    result = run_trial(make_scenario(faults=[{"type": "timeout", "tool": "lookup"}]), agent)
    assert result.verdict == "pass"
    tool_messages = [m for m in fake.requests[1]["messages"] if m["role"] == "tool"]
    assert "timed out" in tool_messages[0]["content"]


@pytest.mark.parametrize("cls", [OpenAILoopAgent, LangGraphAgent])
def test_history_carries_across_turns(cls):
    agent, fake = build(cls, [text_reply("one"), text_reply("two")])
    result = run_trial(make_scenario(user=["first", "second"]), agent)
    assert result.trace.answers == ["one", "two"]
    roles = [m["role"] for m in fake.requests[1]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


@pytest.mark.parametrize("cls", [OpenAILoopAgent, LangGraphAgent])
def test_step_limit_ends_a_runaway_loop(cls):
    replies = [tool_reply("lookup", {"key": str(i)}, f"c{i}") for i in range(10)]
    agent, _ = build(cls, replies, max_steps=2)
    result = run_trial(make_scenario(), agent)
    assert len(result.trace.tool_calls) == 2
    assert result.trace.final_answer == ""
    assert result.verdict != "crash"


@pytest.mark.parametrize("cls", [OpenAILoopAgent, LangGraphAgent])
def test_invented_tools_are_recorded_not_fatal(cls):
    agent, _ = build(cls, [tool_reply("nonexistent", {}), text_reply("sorry")])
    result = run_trial(make_scenario(), agent)
    if cls is OpenAILoopAgent:
        assert result.trace.tool_calls[0].unknown_tool is True
    assert result.verdict != "crash"


def test_malformed_argument_json_is_recorded():
    agent, _ = build(OpenAILoopAgent, [tool_reply("lookup", "{not json"), text_reply("done")])
    result = run_trial(make_scenario(), agent)
    assert "not valid JSON" in result.trace.tool_calls[0].invalid_args


@pytest.mark.parametrize("cls", [OpenAILoopAgent, LangGraphAgent])
def test_provider_errors_are_infra_not_agent_failures(cls):
    agent, _ = build(cls, [400])
    assert run_trial(make_scenario(), agent).verdict == "infra"


def test_missing_api_key_is_explained(monkeypatch):
    monkeypatch.delenv("TEST_KEY")
    with pytest.raises(RuntimeError, match="TEST_KEY"):
        build(OpenAILoopAgent, [])
