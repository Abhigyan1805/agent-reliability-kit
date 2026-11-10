"""Execute scenarios against an agent and store one result per scenario and trial."""

from __future__ import annotations

import json
import sys
import time
import traceback
from collections.abc import Callable, Iterable
from pathlib import Path

from .adapters.base import Agent
from .defences import apply_defences
from .environment import AgentTask, ToolEnv
from .llm.transport import InfraError, QuotaExhausted
from .scenario import Scenario
from .scoring import ScenarioResult, evaluate
from .trace import Trace

try:
    import openai

    _PROVIDER_ERRORS: tuple[type[BaseException], ...] = (openai.APIError, InfraError)
except ImportError:  # pragma: no cover
    _PROVIDER_ERRORS = (InfraError,)


def _causes(exc: BaseException) -> Iterable[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def run_trial(
    scenario: Scenario,
    agent: Agent,
    config_name: str = "",
    defences: list[str] | None = None,
    trial: int = 0,
    max_real_delay_s: float = 2.0,
) -> ScenarioResult:
    """Run one scenario once. Raises only :class:`QuotaExhausted`."""
    trace = Trace(
        scenario_id=scenario.id,
        category=scenario.category,
        config=config_name,
        trial=trial,
        system=scenario.system,
    )
    env = ToolEnv(scenario, trace, max_real_delay_s=max_real_delay_s)
    task = AgentTask(
        scenario_id=scenario.id, system=scenario.system, tools=env.handles(), env=env, trial=trial
    )
    task = apply_defences(task, defences or [])
    started = time.monotonic()
    session = agent.new_session(task)
    try:
        for turn, message in enumerate(scenario.user):
            env.turn = turn
            task.user_messages.append(message)
            trace.user_turns.append(message)
            trace.answers.append(session.send(message))
    except Exception as exc:
        chain = list(_causes(exc))
        if any(isinstance(e, QuotaExhausted) for e in chain):
            raise next(e for e in chain if isinstance(e, QuotaExhausted)) from exc
        if any(isinstance(e, _PROVIDER_ERRORS) for e in chain):
            trace.infra_error = f"{type(exc).__name__}: {exc}"[:500]
        else:
            trace.crash = "".join(traceback.format_exception_only(exc)).strip()[:500]
    trace.duration_s = round(time.monotonic() - started, 3)
    usage = getattr(session, "usage", None)
    if usage:
        trace.usage = dict(usage)
    return evaluate(scenario, trace)


def load_results(path: str | Path) -> list[ScenarioResult]:
    """Read a results file. If a scenario was re-run, the last record wins."""
    latest: dict[tuple[str, int], ScenarioResult] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            result = ScenarioResult.model_validate_json(line)
            latest[(result.scenario_id, result.trial)] = result
    return list(latest.values())


def rescore(results: list[ScenarioResult], scenarios: list[Scenario]) -> list[ScenarioResult]:
    """Apply the current checks to stored traces. No agent or model is involved."""
    by_id = {s.id: s for s in scenarios}
    return [
        evaluate(by_id[r.scenario_id], r.trace) if r.scenario_id in by_id else r for r in results
    ]


def write_results(results: list[ScenarioResult], path: str | Path) -> None:
    lines = [json.dumps(r.model_dump(), ensure_ascii=False) for r in results]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_suite(
    scenarios: list[Scenario],
    agent: Agent,
    config_name: str,
    out_path: str | Path,
    defences: list[str] | None = None,
    trials: int = 1,
    resume: bool = True,
    max_real_delay_s: float = 2.0,
    on_result: Callable[[ScenarioResult], None] | None = None,
) -> tuple[list[ScenarioResult], bool]:
    """Run a suite, appending to ``out_path`` after every scenario so progress is never lost.

    Returns the results and whether the run finished. It stops early only if a long-window
    provider quota is exhausted, and resuming later skips everything already scored.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: dict[tuple[str, int], ScenarioResult] = {}
    if resume and out.exists():
        done = {(r.scenario_id, r.trial): r for r in load_results(out) if r.verdict != "infra"}
    elif out.exists():
        out.unlink()

    finished = True
    with out.open("a", encoding="utf-8") as handle:
        for scenario in scenarios:
            for trial in range(trials):
                if (scenario.id, trial) in done:
                    continue
                try:
                    result = run_trial(
                        scenario, agent, config_name, defences, trial, max_real_delay_s
                    )
                except QuotaExhausted as exc:
                    print(f"quota exhausted, stopping: {exc}", file=sys.stderr)
                    finished = False
                    break
                done[(scenario.id, trial)] = result
                handle.write(json.dumps(result.model_dump(), ensure_ascii=False) + "\n")
                handle.flush()
                if on_result:
                    on_result(result)
            if not finished:
                break
    wanted = {s.id for s in scenarios}
    return [r for key, r in done.items() if key[0] in wanted and key[1] < trials], finished
