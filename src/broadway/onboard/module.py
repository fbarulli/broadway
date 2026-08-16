"""Interactive scaffolder: read CSV → infer hints → ask questions → write contracts."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import yaml

from broadway.analysis.contracts import AnalysisContract, AnalysisMode
from broadway.config.loader import CONFIGS_DIR
from broadway.config.schema import (
    ColumnRole,
    ColumnSchema,
    DatasetContract,
    DerivedFeature,
    EncodingConfig,
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    SplitConfig,
    TaskType,
)
from broadway.discover.profile import build_profile
from broadway.lineage.ids import node_id
from broadway.lineage.records import write_record
from broadway.onboard.infer import infer
from broadway.onboard.models import InferenceReport

ARTIFACTS_DIR = Path(os.getenv("BROADWAY_ARTIFACTS_DIR", "artifacts"))


def _read(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path) if csv_path.endswith(".csv") else pd.read_parquet(csv_path)


def _prompt(question: str, default: str | None = None) -> str | None:
    prompt = question if default is None else f"{question} [{default}]"
    answer = input(prompt)
    return answer if answer else default


def _prompt_required(question: str, default: str | None = None) -> str:
    value = _prompt(question, default)
    while not value:
        value = _prompt(question)
    return value


def _split_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_dataset_contract(
    name: str,
    path: str,
    target: str,
    task: str,
    datetime_columns: list[str],
    ignore_columns: list[str],
    split_column: str | None,
    report: InferenceReport,
    row_count: int,
) -> DatasetContract:
    columns: dict[str, ColumnSchema] = {}
    for col, hint in report.columns.items():
        if col == target:
            role = ColumnRole.TARGET
        elif col in ignore_columns:
            role = ColumnRole.IGNORE
        elif col in datetime_columns:
            role = ColumnRole.DATETIME
        else:
            role = ColumnRole.FEATURE
        columns[col] = ColumnSchema(
            dtype="datetime64" if role == ColumnRole.DATETIME else hint.dtype,
            null_count=round(hint.null_rate * row_count),
            role=role,
        )
    return DatasetContract(
        name=name,
        path=path,
        target=target,
        task=TaskType(task),
        datetime_column=split_column,
        columns=columns,
        lookup_tables={},
    )


def build_analysis_contract(
    name: str,
    mode: AnalysisMode,
    goal: str,
    row_definition: str,
    decision_moment: str,
    available_info: list[str],
    leakage_notes: list[str],
    success_criterion: str,
) -> AnalysisContract:
    return AnalysisContract(
        name=name,
        mode=mode,
        goal=goal,
        row_definition=row_definition,
        decision_moment=decision_moment,
        available_info=available_info,
        leakage_notes=leakage_notes,
        success_criterion=success_criterion,
    )


def build_experiment_config(
    report: InferenceReport,
    target: str,
    datetime_columns: list[str],
    ignore_columns: list[str],
    split_column: str | None,
) -> ExperimentConfig:
    feature_cols = [
        col
        for col, hint in report.columns.items()
        if hint.suggested_role == "feature"
        and col != target
        and col not in ignore_columns
        and col not in datetime_columns
    ]
    include = [c for c in feature_cols if report.columns[c].datetime_candidate is False]
    derived = [
        DerivedFeature(name=f"{c}_{suffix}", func=func, source=c)
        for c in datetime_columns
        for suffix, func in (
            ("hour", "datetime_hour"),
            ("dayofweek", "datetime_dayofweek"),
            ("month", "datetime_month"),
        )
    ]
    categorical_cols = [c for c in feature_cols if report.columns[c].categorical is True]
    encodings = (
        [
            EncodingConfig(type="target", columns=categorical_cols, smoothing=20),
            EncodingConfig(type="frequency", columns=categorical_cols, smoothing=None),
        ]
        if categorical_cols
        else []
    )
    return ExperimentConfig(
        features=FeatureConfig(include=include, exclude=[], derived=derived, encodings=encodings),
        model=ModelConfig(type="linear", params={}),
        split=SplitConfig(type="time" if split_column else "random", validation_size=0.2),
        random_state=42,
        target_metric="rmse",
        hpo=None,
    )


def _print_summary(report: InferenceReport) -> None:
    print(f"inferred {len(report.columns)} columns from {report.row_count} rows:")
    for col, hint in report.columns.items():
        print(
            f"  {col}: dtype={hint.dtype}, cardinality={hint.cardinality}, "
            f"role={hint.suggested_role}, datetime_candidate={hint.datetime_candidate}"
        )


def _write_configs(
    name: str,
    dataset_contract: DatasetContract,
    analysis_contract: AnalysisContract,
    experiment_config: ExperimentConfig,
) -> list[Path]:
    contracts = [
        (CONFIGS_DIR / "dataset" / f"{name}.yaml", dataset_contract),
        (CONFIGS_DIR / "analysis" / f"{name}.yaml", analysis_contract),
        (CONFIGS_DIR / "experiment" / f"{name}.yaml", experiment_config),
    ]
    written: list[Path] = []
    for path, contract in contracts:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(contract.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
        written.append(path)
    return written


def _write_profile(name: str, csv: str, df: pd.DataFrame) -> Path:
    profile = build_profile(name, csv, df)
    profile_dir = ARTIFACTS_DIR / "discover"
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_path = profile_dir / "profile.json"
    profile_path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    write_record(
        node_id("profile", name),
        "profile",
        str(profile_path),
        [node_id("dataset", name)],
    )
    return profile_path


def init(
    csv: str,
    name: str,
    target: str | None,
    task: str | None,
    datetime_columns: list[str] | None = None,
    ignore_columns: list[str] | None = None,
    split_column: str | None = None,
    mode: str | None = None,
    goal: str | None = None,
    row_definition: str | None = None,
    decision_moment: str | None = None,
    available_info: list[str] | None = None,
    leakage_notes: list[str] | None = None,
    success_criterion: str | None = None,
) -> None:
    df = _read(csv)
    report = infer(name, df)
    _print_summary(report)

    required = {
        "target": target,
        "task": task,
        "mode": mode,
        "goal": goal,
        "row_definition": row_definition,
        "decision_moment": decision_moment,
        "success_criterion": success_criterion,
    }
    missing = [flag for flag, value in required.items() if value is None]
    tty = sys.stdin.isatty()
    if missing and not tty:
        raise ValueError(f"missing required init flags: {', '.join(missing)}")

    dt_candidates = [c for c, hint in report.columns.items() if hint.datetime_candidate]
    ignore_candidates = [c for c, hint in report.columns.items() if hint.suggested_role == "ignore"]
    feature_suggestion = next(
        (c for c, hint in report.columns.items() if hint.suggested_role == "feature"), None
    )

    if target is None:
        target = _prompt_required("target column", feature_suggestion)
    if task is None:
        task = _prompt_required("task (regression/classification)", "regression")
    if datetime_columns is None:
        datetime_columns = (
            _split_list(_prompt("datetime columns (comma-separated, or empty)", ",".join(dt_candidates)))
            if tty
            else dt_candidates
        )
    if ignore_columns is None:
        ignore_columns = (
            _split_list(_prompt("columns to ignore (comma-separated, or empty)", ",".join(ignore_candidates)))
            if tty
            else ignore_candidates
        )
    if split_column is None and tty:
        split_column = _prompt("column governing train/val split (or empty for random)", "") or None
    if mode is None:
        mode = _prompt_required("analysis mode (prediction/hypothesis/causal)", "prediction")
    if goal is None:
        goal = _prompt_required("goal")
    if row_definition is None:
        row_definition = _prompt_required("row definition")
    if decision_moment is None:
        decision_moment = _prompt_required("decision moment")
    if available_info is None:
        available_info = (
            _split_list(_prompt("available info at decision time (comma-separated)")) if tty else []
        )
    if leakage_notes is None:
        leakage_notes = (
            _split_list(_prompt("leakage notes (comma-separated)", "")) if tty else []
        )
    if success_criterion is None:
        success_criterion = _prompt_required("success criterion")

    dataset_contract = build_dataset_contract(
        name, csv, target, task, datetime_columns, ignore_columns, split_column, report, report.row_count
    )
    analysis_contract = build_analysis_contract(
        name, AnalysisMode(mode), goal, row_definition, decision_moment, available_info, leakage_notes, success_criterion
    )
    experiment_config = build_experiment_config(
        report, target, datetime_columns, ignore_columns, split_column
    )

    for path in _write_configs(name, dataset_contract, analysis_contract, experiment_config):
        print(f"wrote {path}")
    profile_path = _write_profile(name, csv, df)
    print(f"wrote {profile_path}")
    print(f"next steps: ds-pipeline full --dataset {name} --analysis {name}")
