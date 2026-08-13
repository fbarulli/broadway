"""CLI entry point — parses args, loads config, dispatches to pipeline."""

from __future__ import annotations

import argparse

from broadway.config.loader import DEFAULT_ENVIRONMENT, STEP_MODELS, load_config, resolve_full_steps

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

    for step in STEPS:
        if step == "discover":
            continue
        _add_step_args(sub.add_parser(step))

    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.step == "discover":
        from broadway.discover.module import run

        run(args.csv, args.target, args.task, args.datetime_column, args.ignore_columns)
    elif args.step == "lineage":
        from broadway.lineage.module import run

        run(args.analysis, args.dataset)
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
