"""Render results as Markdown tables and readable traces."""

from __future__ import annotations

from collections import Counter

from .scoring import ScenarioResult, compare, ordered_categories, summarise


def _cell(entry: dict) -> str:
    if not entry["total"]:
        return "n/a"
    low, high = entry["ci95"]
    return f"{entry['rate'] * 100:.0f}% ({entry['passed']}/{entry['total']}) [{low * 100:.0f}-{high * 100:.0f}]"


def render_report(runs: dict[str, list[ScenarioResult]]) -> str:
    """Build the comparison report for one or more named runs."""
    summaries = {name: summarise(results) for name, results in runs.items()}
    names = list(runs)
    lines = [
        "Each cell shows the pass rate, (passed/total) and a 95% Wilson interval. "
        "A scenario passes only if every safety and utility check holds.",
        "",
        "| Category | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    categories = ordered_categories({c for s in summaries.values() for c in s["categories"]})
    for category in categories:
        cells = [_cell(s["categories"].get(category, {"total": 0})) for s in summaries.values()]
        lines.append(f"| {category} | " + " | ".join(cells) + " |")
    for key, label in [
        ("pass", "**Overall pass**"),
        ("safe", "Safe (safety checks only)"),
        ("useful", "Useful (utility checks only)"),
    ]:
        lines.append(
            f"| {label} | "
            + " | ".join(_cell(s["overall"][key]) for s in summaries.values())
            + " |"
        )

    lines += ["", "| Failure type | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    tallies = {name: Counter(r.verdict for r in results) for name, results in runs.items()}
    for verdict in ["unsafe", "unhelpful", "crash", "infra"]:
        lines.append(f"| {verdict} | " + " | ".join(str(tallies[n][verdict]) for n in names) + " |")

    if len(names) > 1:
        lines += [
            "",
            "Paired comparisons use the exact McNemar test on scenarios scored in both runs.",
            "",
            "| A | B | shared | both pass | only A passes | only B passes | p |",
            "|---|---|---|---|---|---|---|",
        ]
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                result = compare(runs[a], runs[b])
                lines.append(
                    f"| {a} | {b} | {result['scenarios']} | {result['both_pass']} "
                    f"| {len(result['only_a'])} | {len(result['only_b'])} "
                    f"| {result['p_value']:.3f} |"
                )
    return "\n".join(lines) + "\n"


def render_trace(result: ScenarioResult, max_chars: int = 700) -> str:
    """A human-readable account of what an agent did in one scenario."""

    def clip(text: str) -> str:
        return text if len(text) <= max_chars else text[:max_chars] + " ...[clipped]"

    trace = result.trace
    lines = [
        f"### {result.scenario_id}: {result.title}",
        f"Config `{result.config}`, verdict **{result.verdict}**.",
        "",
        "```text",
    ]
    if trace.infra_error:
        lines.append(f"INFRA ERROR: {clip(trace.infra_error)}")
    if trace.crash:
        lines.append(f"AGENT CRASH: {clip(trace.crash)}")
    for turn, message in enumerate(trace.user_turns):
        lines.append(f"USER[{turn}]: {clip(message)}")
        for call in (c for c in trace.tool_calls if c.turn == turn):
            lines.append(f"  CALL {call.tool}({clip(str(call.args))})")
            tag = f" <fault: {', '.join(call.faults)}>" if call.faults else ""
            outcome = f"RAISED {call.error}" if call.error else clip(call.result or "")
            lines.append(f"  RESULT{tag}: {outcome}")
        for event in (e for e in trace.defence_events if e.turn == turn):
            lines.append(f"  DEFENCE {event.defence} {event.action} {event.tool}: {event.detail}")
        if turn < len(trace.answers):
            lines.append(f"AGENT[{turn}]: {clip(trace.answers[turn])}")
    lines += ["```", ""]
    for check in result.checks:
        mark = "PASS" if check.passed else "FAIL"
        lines.append(f"- {mark} ({check.kind}) `{check.type}` {check.detail[:200]}")
    return "\n".join(lines) + "\n"
