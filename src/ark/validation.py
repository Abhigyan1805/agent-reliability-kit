"""Check the checks: prove each scenario separates good behaviour from bad before trusting it."""

from __future__ import annotations

from .adapters.scripted import NULL_AGENT, REFUSER_AGENT, ScriptedAgent
from .runner import run_trial
from .scenario import Scenario


def validate_scenario(scenario: Scenario) -> list[str]:
    """Return every way a scenario's rules fail their controls; empty means sound.

    The golden good run must pass, the golden bad run must not, and an agent that does nothing or
    refuses everything must not pass, or a safety score could be earned by being useless.
    """
    problems: list[str] = []
    good = run_trial(scenario, ScriptedAgent(scenario.golden.good), max_real_delay_s=0)
    if good.verdict != "pass":
        failed = [f"{c.type}: {c.detail}" for c in good.checks if not c.passed]
        problems.append(f"golden good run scored {good.verdict}: {failed or good.trace.crash}")
    bad = run_trial(scenario, ScriptedAgent(scenario.golden.bad), max_real_delay_s=0)
    if bad.verdict == "pass":
        problems.append("golden bad run passed, so the checks do not catch the failure")
    if run_trial(scenario, NULL_AGENT, max_real_delay_s=0).verdict == "pass":
        problems.append("an agent that does nothing passes this scenario")
    if (
        not scenario.refusal_ok
        and run_trial(scenario, REFUSER_AGENT, max_real_delay_s=0).verdict == "pass"
    ):
        problems.append("an agent that refuses everything passes this scenario")
    return problems


def validate_suite(scenarios: list[Scenario]) -> dict[str, list[str]]:
    results = {s.id: validate_scenario(s) for s in scenarios}
    return {scenario_id: found for scenario_id, found in results.items() if found}
