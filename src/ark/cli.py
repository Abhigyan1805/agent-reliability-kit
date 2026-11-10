"""Command line interface: ``ark list | validate | run | rescore | report | show``."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .adapters import build_agent
from .config import load_config, load_env_file
from .report import render_report, render_trace
from .runner import load_results, rescore, run_suite, write_results
from .scenario import ScenarioError, load_scenarios
from .scoring import ScenarioResult, summarise
from .validation import validate_suite

EXIT_QUOTA = 3


def _selected(args: argparse.Namespace):
    scenarios = load_scenarios(args.suite)
    if args.category:
        scenarios = [s for s in scenarios if s.category in args.category]
    if args.only:
        scenarios = [s for s in scenarios if s.id in args.only]
    if args.limit:
        scenarios = scenarios[: args.limit]
    return scenarios


def cmd_list(args: argparse.Namespace) -> int:
    scenarios = _selected(args)
    for scenario in scenarios:
        print(f"{scenario.id:32} {scenario.category:22} {scenario.title}")
    counts = Counter(s.category for s in scenarios)
    print(f"\n{len(scenarios)} scenarios: {dict(counts)}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    scenarios = _selected(args)
    problems = validate_suite(scenarios)
    for scenario_id, found in problems.items():
        for problem in found:
            print(f"{scenario_id}: {problem}")
    print(f"{len(scenarios) - len(problems)}/{len(scenarios)} scenarios are sound")
    return 1 if problems else 0


def _progress(result: ScenarioResult) -> None:
    print(f"{result.verdict:10} {result.scenario_id}", flush=True)


def cmd_run(args: argparse.Namespace) -> int:
    if args.env_file:
        load_env_file(args.env_file)
    config = load_config(args.config)
    out = Path(args.out or f"runs/{config.name}/results.jsonl")
    results, finished = run_suite(
        _selected(args),
        build_agent(config),
        config.name,
        out,
        defences=config.defences,
        trials=args.trials,
        resume=not args.fresh,
        on_result=_progress,
    )
    print(json.dumps(summarise(results), indent=2))
    if not finished:
        print("run incomplete: re-run the same command after the quota resets", file=sys.stderr)
        return EXIT_QUOTA
    return 0


def cmd_rescore(args: argparse.Namespace) -> int:
    scenarios = load_scenarios(args.suite)
    for path in args.results:
        before = load_results(path)
        after = rescore(before, scenarios)
        write_results(after, path)
        changed = [
            f"{old.scenario_id}: {old.verdict} -> {new.verdict}"
            for old, new in zip(before, after, strict=True)
            if old.verdict != new.verdict
        ]
        print(f"{path}: {len(changed)} verdict(s) changed")
        for line in changed:
            print(f"  {line}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    runs: dict[str, list[ScenarioResult]] = {}
    for path in args.results:
        results = load_results(path)
        name = results[0].config if results else Path(path).parent.name
        runs[name] = results
    text = render_report(runs)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    for result in load_results(args.results):
        if args.scenario and result.scenario_id not in args.scenario:
            continue
        if args.failures and result.verdict == "pass":
            continue
        print(render_trace(result, max_chars=args.max_chars))
    return 0


def _add_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--suite", default="scenarios", help="scenario file or directory")
    parser.add_argument("--category", action="append", help="repeatable category filter")
    parser.add_argument("--only", action="append", help="repeatable scenario id filter")
    parser.add_argument("--limit", type=int, help="run only the first N selected scenarios")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ark", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    listing = commands.add_parser("list", help="list scenarios")
    _add_selection(listing)
    listing.set_defaults(handler=cmd_list)

    validate = commands.add_parser("validate", help="prove each scenario's checks discriminate")
    _add_selection(validate)
    validate.set_defaults(handler=cmd_validate)

    run = commands.add_parser("run", help="run scenarios against a configured agent")
    _add_selection(run)
    run.add_argument("--config", required=True, help="run configuration YAML")
    run.add_argument("--out", help="results JSONL path (default runs/<config>/results.jsonl)")
    run.add_argument("--trials", type=int, default=1, help="repeats per scenario")
    run.add_argument("--fresh", action="store_true", help="ignore existing results")
    run.add_argument("--env-file", help="KEY=VALUE file with API keys")
    run.set_defaults(handler=cmd_run)

    rescoring = commands.add_parser("rescore", help="re-apply current checks to stored traces")
    rescoring.add_argument("results", nargs="+", help="results JSONL files, updated in place")
    rescoring.add_argument("--suite", default="scenarios", help="scenario file or directory")
    rescoring.set_defaults(handler=cmd_rescore)

    report = commands.add_parser("report", help="compare one or more result files")
    report.add_argument("results", nargs="+", help="results JSONL files")
    report.add_argument("--out", help="write the Markdown report here too")
    report.set_defaults(handler=cmd_report)

    show = commands.add_parser("show", help="print readable traces")
    show.add_argument("results", help="results JSONL file")
    show.add_argument("--scenario", action="append", help="repeatable scenario id filter")
    show.add_argument("--failures", action="store_true", help="only non-passing runs")
    show.add_argument("--max-chars", type=int, default=700)
    show.set_defaults(handler=cmd_show)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (ScenarioError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
