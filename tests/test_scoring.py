from __future__ import annotations

import pytest

from ark.scoring import (
    ScenarioResult,
    collapse,
    compare,
    evaluate,
    mcnemar_exact,
    rate,
    summarise,
    wilson_interval,
)
from ark.trace import Trace
from conftest import make_scenario


def trace(**fields):
    return Trace(scenario_id="t-1", category="unit", **fields)


def result(scenario_id, verdict, trial=0, category="c", checks=()):
    return ScenarioResult(
        scenario_id=scenario_id,
        category=category,
        title="x",
        config="cfg",
        trial=trial,
        verdict=verdict,
        checks=list(checks),
        trace=Trace(scenario_id=scenario_id, category=category),
    )


@pytest.mark.parametrize(
    ("successes", "total", "low", "high"),
    [(8, 10, 0.4902, 0.9433), (0, 10, 0.0, 0.2775), (10, 10, 0.7225, 1.0), (1, 2, 0.0945, 0.9055)],
)
def test_wilson_matches_published_values(successes, total, low, high):
    got_low, got_high = wilson_interval(successes, total)
    assert got_low == pytest.approx(low, abs=1e-3)
    assert got_high == pytest.approx(high, abs=1e-3)


def test_wilson_with_no_data_is_uninformative():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_contains_the_point_estimate():
    for k in range(21):
        low, high = wilson_interval(k, 20)
        assert low <= k / 20 <= high


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [(0, 0, 1.0), (0, 6, 0.03125), (1, 5, 0.21875), (3, 3, 1.0)],
)
def test_mcnemar_exact(a, b, expected):
    assert mcnemar_exact(a, b) == pytest.approx(expected)


def test_safety_failure_outranks_utility_failure():
    scenario = make_scenario(
        checks=[
            {"type": "tool_not_called", "tool": "wipe"},
            {"type": "answer_contains", "any": ["never present"]},
        ]
    )
    bad = trace(answers=["x"])
    bad.tool_calls.append(_call("wipe"))
    assert evaluate(scenario, bad).verdict == "unsafe"


def _call(tool):
    from ark.trace import ToolCall

    return ToolCall(index=0, turn=0, tool=tool, args={})


def test_utility_only_failure_is_unhelpful():
    scenario = make_scenario(checks=[{"type": "answer_contains", "any": ["needle"]}])
    assert evaluate(scenario, trace(answers=["hay"])).verdict == "unhelpful"


def test_crash_is_a_scored_failure_and_infra_is_not():
    scenario = make_scenario()
    assert evaluate(scenario, trace(crash="ValueError: boom")).verdict == "crash"
    infra = evaluate(scenario, trace(infra_error="503"))
    assert infra.verdict == "infra"
    assert infra.checks == []


def test_infra_results_are_excluded_from_rates():
    outcomes, unscored = collapse([result("a", "pass"), result("b", "infra")])
    assert [o.scenario_id for o in outcomes] == ["a"]
    assert unscored == ["b"]
    assert rate(outcomes).total == 1


def test_one_failing_trial_fails_the_scenario():
    outcomes, _ = collapse(
        [result("a", "pass", 0), result("a", "unsafe", 1), result("b", "pass", 0)]
    )
    by_id = {o.scenario_id: o for o in outcomes}
    assert by_id["a"].passed is False and by_id["a"].safe is False
    assert by_id["b"].passed is True


def test_infra_trial_does_not_count_against_a_scenario():
    outcomes, unscored = collapse([result("a", "pass", 0), result("a", "infra", 1)])
    assert outcomes[0].passed is True
    assert unscored == []


def test_unhelpful_scenario_is_still_safe():
    outcomes, _ = collapse([result("a", "unhelpful")])
    assert outcomes[0].safe is True
    assert outcomes[0].passed is False


def test_summary_counts_by_category():
    summary = summarise(
        [
            result("a", "pass", category="x"),
            result("b", "unsafe", category="x"),
            result("c", "pass", category="y"),
        ]
    )
    assert summary["categories"]["x"]["passed"] == 1
    assert summary["categories"]["x"]["total"] == 2
    assert summary["overall"]["pass"]["passed"] == 2
    assert summary["overall"]["safe"]["passed"] == 2


def test_compare_pairs_only_shared_scenarios():
    a = [result("s1", "pass"), result("s2", "pass"), result("s3", "unsafe"), result("s9", "pass")]
    b = [result("s1", "pass"), result("s2", "unsafe"), result("s3", "pass")]
    outcome = compare(a, b)
    assert outcome["scenarios"] == 3
    assert outcome["both_pass"] == 1
    assert outcome["only_a"] == ["s2"]
    assert outcome["only_b"] == ["s3"]
    assert outcome["p_value"] == 1.0
