from __future__ import annotations

import json

import pytest
from pydantic import TypeAdapter

from ark.environment import ToolEnv
from ark.faults import (
    Fault,
    ToolError,
    ToolOutcome,
    ToolTimeout,
    apply_faults,
    stable_seed,
)
from ark.trace import Trace
from conftest import make_scenario


def fault(**spec):
    return TypeAdapter(Fault).validate_python(spec)


def env_for(*faults, **kwargs):
    scenario = make_scenario(faults=list(faults))
    trace = Trace(scenario_id=scenario.id, category=scenario.category)
    return ToolEnv(scenario, trace, **kwargs), trace


def run_fault(spec, value):
    return fault(**spec).apply(ToolOutcome(value=value), seed=1)


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        ("append", "doc\nX"),
        ("prepend", "X\ndoc"),
        ("replace", "X"),
    ],
)
def test_inject_positions_on_text(position, expected):
    spec = {"type": "inject", "tool": "t", "payload": "X", "position": position}
    assert run_fault(spec, "doc").value == expected


def test_inject_middle_splits_at_a_sentence_boundary():
    text = "First sentence here. Second sentence here. Third one."
    spec = {"type": "inject", "tool": "t", "payload": "X", "position": "middle"}
    out = run_fault(spec, text).value
    assert out.startswith("First sentence here. Second sentence here. X")
    assert out.endswith("Third one.")


def test_inject_into_nested_field_does_not_mutate_the_stub():
    original = {"results": [{"text": "a"}, {"text": "b"}]}
    spec = {"type": "inject", "tool": "t", "payload": "X", "field": "results.1.text"}
    out = run_fault(spec, original).value
    assert out["results"][1]["text"] == "b\nX"
    assert original["results"][1]["text"] == "b"


def test_inject_on_structured_result_requires_a_field():
    with pytest.raises(ValueError, match="field"):
        run_fault({"type": "inject", "tool": "t", "payload": "X"}, {"a": 1})


def test_error_fault_payload_and_raise_modes():
    assert run_fault({"type": "error", "tool": "t", "code": 503}, "ok").value["code"] == 503
    raised = run_fault({"type": "error", "tool": "t", "raises": True, "message": "boom"}, "ok")
    assert raised.raises is ToolError
    assert raised.message == "boom"


def test_timeout_raises_timeout_and_counts_simulated_latency():
    out = run_fault({"type": "timeout", "tool": "t", "simulated_ms": 1234}, "ok")
    assert out.raises is ToolTimeout
    assert out.simulated_ms == 1234


def test_corrupt_truncate_produces_invalid_json():
    value = {"items": [1, 2, 3, 4, 5, 6, 7, 8]}
    out = run_fault({"type": "corrupt", "tool": "t", "mode": "truncate", "fraction": 0.5}, value)
    assert len(out.value) == int(len(json.dumps(value)) * 0.5)
    with pytest.raises(json.JSONDecodeError):
        json.loads(out.value)


def test_corrupt_null_fields_keeps_shape():
    out = run_fault(
        {"type": "corrupt", "tool": "t", "mode": "null_fields"}, {"a": 1, "b": [{"c": "x"}]}
    )
    assert out.value == {"a": None, "b": [{"c": None}]}


def test_corrupt_wrong_type_and_garbage_are_deterministic():
    wrong = run_fault({"type": "corrupt", "tool": "t", "mode": "wrong_type"}, {"a": 1})
    assert wrong.value == ['{"a": 1}']
    one = run_fault({"type": "corrupt", "tool": "t", "mode": "garbage"}, "x").value
    two = run_fault({"type": "corrupt", "tool": "t", "mode": "garbage"}, "x").value
    assert one == two and len(one) == 120


def test_replace_and_empty():
    assert run_fault({"type": "replace", "tool": "t", "value": {"x": 2}}, {"x": 1}).value == {
        "x": 2
    }
    assert run_fault({"type": "empty", "tool": "t"}, {"x": 1}).value is None


def test_fault_targets_only_named_calls_and_matching_args():
    spec = fault(type="replace", tool="t", value="F", calls=[2], when={"k": "yes"})
    assert not spec.applies(1, {"k": "yes"})
    assert not spec.applies(2, {"k": "no"})
    assert spec.applies(2, {"k": "yes"})


def test_faults_compose_in_order():
    faults = [
        fault(type="replace", tool="t", value="base"),
        fault(type="inject", tool="t", payload="!", position="append"),
        fault(type="slow", tool="t", simulated_ms=5),
    ]
    out = apply_faults(faults, "t", 1, {}, ToolOutcome(value="x"), seed=1)
    assert out.value == "base\n!"
    assert out.simulated_ms == 5
    assert out.applied == ["replace", "inject", "slow"]


def test_faults_for_other_tools_are_ignored():
    faults = [fault(type="replace", tool="other", value="F")]
    assert apply_faults(faults, "t", 1, {}, ToolOutcome(value="x"), seed=1).applied == []


def test_stable_seed_is_stable_and_input_sensitive():
    assert stable_seed("a", 1) == stable_seed("a", 1)
    assert stable_seed("a", 1) != stable_seed("a", 2)


def test_env_applies_call_numbers_per_tool():
    env, trace = env_for({"type": "error", "tool": "lookup", "calls": [1], "message": "first"})
    assert "first" in env.call("lookup", {"key": "a"})
    assert "hello" in env.call("lookup", {"key": "a"})
    assert [c.faults for c in trace.tool_calls] == [["error"], []]


def test_env_records_raised_errors_with_latency():
    env, trace = env_for({"type": "timeout", "tool": "lookup", "simulated_ms": 9})
    with pytest.raises(ToolTimeout):
        env.call("lookup", {"key": "a"})
    assert trace.tool_calls[0].error == "Tool call timed out"
    assert trace.tool_calls[0].simulated_ms == 9


def test_env_caps_real_delay():
    slept: list[float] = []
    env, _ = env_for(
        {"type": "slow", "tool": "lookup", "real_delay_s": 30},
        max_real_delay_s=0.5,
        sleep=slept.append,
    )
    env.call("lookup", {"key": "a"})
    assert slept == [0.5]


def test_env_rejects_invalid_args_without_running_the_tool_or_faults():
    env, trace = env_for({"type": "replace", "tool": "lookup", "value": "F"})
    assert "Invalid arguments" in env.call("lookup", {})
    assert trace.tool_calls[0].invalid_args
    assert trace.tool_calls[0].faults == []


def test_env_records_invented_tools():
    env, trace = env_for()
    assert "Unknown tool" in env.call("nope", {"a": 1})
    assert trace.tool_calls[0].unknown_tool is True
