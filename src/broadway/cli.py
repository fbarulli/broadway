"""CLI entry point — parses args, loads config, dispatches to pipeline."""

from __future__ import annotations

import argparse
import logging

from broadway.config.loader import DEFAULT_ENVIRONMENT, STEP_MODELS, load_config, resolve_full_steps
from broadway.lineage.sample import load_sample

STEPS = list(STEP_MODELS.keys())


def _add_step_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--analysis", type=str, default=None)
    parser.add_argument("--environment", type=str, default=DEFAULT_ENVIRONMENT)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ds-pipeline")
    sub = parser.add_subparsers(dest="step", required=True)

    discover = sub.add_parser("discover")
    discover.add_argument("--csv", type=str, required=True)
    discover.add_argument("--target", type=str, required=True)
    discover.add_argument("--task", type=str, required=True)
    discover.add_argument("--datetime-column", type=str, default=None)
    discover.add_argument("--ignore-columns", nargs="*", default=[])

    lineage = sub.add_parser("lineage")
    lineage.add_argument("--analysis", type=str, required=True)
    lineage.add_argument("--dataset", type=str, required=True)

    report = sub.add_parser("report")
    report.add_argument("--analysis", required=True)
    report.add_argument("--dataset", required=True)

    profile = sub.add_parser("profile")
    profile.add_argument("--dataset", type=str, required=True)

    audit = sub.add_parser("audit")
    audit.add_argument("--dataset", type=str, required=True)
    audit.add_argument("--analysis", type=str, default=None)
    audit.add_argument("--environment", type=str, default=DEFAULT_ENVIRONMENT)

    sub.add_parser("ingest")

    init = sub.add_parser("init")
    init.add_argument("csv")
    init.add_argument("--name", required=True)
    init.add_argument("--target")
    init.add_argument("--task")
    init.add_argument("--datetime-columns", nargs="*")
    init.add_argument("--ignore-columns", nargs="*")
    init.add_argument("--split-column")
    init.add_argument("--mode")
    init.add_argument("--goal")
    init.add_argument("--row-definition")
    init.add_argument("--decision-moment")
    init.add_argument("--available-info", nargs="*")
    init.add_argument("--leakage-notes", nargs="*")
    init.add_argument("--success-criterion")

    for step in STEPS:
        if step in ("discover", "stats"):
            continue
        _add_step_args(sub.add_parser(step))

    stats = sub.add_parser("stats")
    stats_sub = stats.add_subparsers(dest="stats_subcommand", required=True)
    for sc in ("run", "describe"):
        p = stats_sub.add_parser(sc)
        _add_step_args(p)
        p.add_argument("--sample", required=True)

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    args = _build_parser().parse_args()

    if args.step == "discover":
        from broadway.discover.module import run

        run(args.csv, args.target, args.task, args.datetime_column, args.ignore_columns)
    elif args.step == "lineage":
        from broadway.lineage.module import run

        run(args.analysis, args.dataset)
    elif args.step == "report":
        from pathlib import Path

        from broadway.reports import index
        from broadway.reports.paths import REPORTS_DIR

        cfg = load_config("stats", dataset=args.dataset, analysis=args.analysis)
        question = cfg.analysis.goal if cfg.analysis else "no analysis contract"
        stats_dir = Path(cfg.stats.output_dir) if cfg.stats else Path("artifacts/stats")
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "index.md").write_text(index.render_index(question, stats_dir), encoding="utf-8")
        print("wrote reports/index.md")
    elif args.step == "profile":
        from broadway.discover.module import profile

        profile(args.dataset)
    elif args.step == "audit":
        from broadway.reports.audit import run as audit_run

        audit_run(args.dataset, args.analysis, args.environment)
    elif args.step == "ingest":
        from broadway.etl.process import process_data

        process_data()
    elif args.step == "init":
        from broadway.onboard.module import init

        init(args.csv, args.name, args.target, args.task, args.datetime_columns, args.ignore_columns, args.split_column, args.mode, args.goal, args.row_definition, args.decision_moment, args.available_info, args.leakage_notes, args.success_criterion)
    elif args.step == "stats":
        cfg = load_config(
            step="stats", dataset=args.dataset, experiment=args.experiment,
            analysis=args.analysis, environment=args.environment,
        )
        if args.stats_subcommand == "describe":
            from broadway.stats.describe import run as describe_run

            sample = load_sample(args.sample)
            describe_run(cfg, sample)
        else:  # "run"
            from broadway.stats.module import run as module_run

            sample = load_sample(args.sample)
            module_run(cfg, sample)
    else:
        from broadway.pipeline import run

        cfg = load_config(
            step=args.step,
            dataset=args.dataset,
            experiment=args.experiment,
            analysis=args.analysis,
            environment=args.environment,
        )
        steps = resolve_full_steps(cfg) if cfg.full else [args.step]
        run(cfg, steps)
