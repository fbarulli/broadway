#!/usr/bin/env python3
"""Live verification recorder — one evidence record per real command run.

Single owner of the evidence record in ``LIVE_VERIFICATION.md`` §3: run the real
command, measure wall time, capture exit code and output tails, and write
``<output_dir>/<run_id>.json`` for the evidence pack.

PROVENANCE ONLY (§4): the row count is read from parquet footer metadata and
recorded; nothing here validates, hashes, sizes, or gates data. A failed live
run is recorded with its exit code and the process exits with it — never masked.

Usage::

    python scripts/verify_live.py --step etl \
        --command "ds-pipeline etl --dataset amazon" \
        --dataset data/processed/amazon/amazon_canonical.parquet [--log-path logs/etl.log]

The script prints the written JSON path on stdout and exits with the recorded
command's exit code (1 when the command produced no exit code, e.g. timeout).
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from pyarrow import ArrowInvalid
from pydantic import BaseModel, ConfigDict, Field

from broadway.config.loader import config_path
from broadway.observability.context import RUN_ID_ENV, new_run_id
from broadway.paths import repo_root

VERIFICATION_CONFIG = "verification.yaml"
PARQUET_SUFFIX = ".parquet"


class VerificationConfig(BaseModel):
    """Boundary-validated shape of ``configs/verification.yaml``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    output_dir: Path
    default_timeout_seconds: int = Field(gt=0)
    output_tail_lines: int = Field(gt=0)

    @classmethod
    def from_config(cls, path: Path) -> VerificationConfig:
        """Load and validate the config file; a missing/invalid file fails loud."""
        with open(path, encoding="utf-8") as handle:
            return cls.model_validate(yaml.safe_load(handle))


class VerificationRequest(BaseModel):
    """One recording request, validated from the CLI boundary."""

    model_config = ConfigDict(frozen=True)

    step: str = Field(min_length=1)
    command: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    log_path: str | None = None
    output_dir: Path | None = None
    run_id: str = Field(min_length=1)


class Evidence(BaseModel):
    """The persisted evidence record shape (``LIVE_VERIFICATION.md`` §3)."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    commit: str
    step: str
    command: str
    dataset: str
    rows: int | None
    rows_error: str | None = None
    duration_seconds: float
    exit_code: int | None
    stdout_tail: str
    stderr_tail: str
    log_path: str | None
    created_at: datetime


@dataclass(frozen=True)
class RowProvenance:
    """Parquet row count plus the recorded reason when it could not be read."""

    rows: int | None
    error: str | None


@dataclass(frozen=True)
class CommandOutcome:
    """What one live command produced; ``exit_code`` is None only on timeout."""

    exit_code: int | None
    stdout: str | bytes | None
    stderr: str | bytes | None
    duration_seconds: float


@dataclass(frozen=True)
class RecordingResult:
    """Path of the written record plus the exit code the caller must propagate."""

    path: Path
    exit_code: int | None


class CommandRunner:
    """Runs the live command once and measures its wall time.

    ``time.perf_counter`` is the PROFILING.md seam only in ``src/``; this script
    is the evidence recorder and has no pipeline hot path (waiver: tiers 1–3 do
    not apply, one invocation per record).
    """

    def __init__(self, timeout_seconds: int) -> None:
        self._timeout_seconds = timeout_seconds

    def run(self, command: str, run_id: str) -> CommandOutcome:
        """Execute ``command``; a timeout is recorded as a missing exit code."""
        started = time.perf_counter()
        exit_code: int | None
        stdout: str | bytes | None
        stderr: str | bytes | None
        environment = {**os.environ, RUN_ID_ENV: run_id}
        try:
            completed = subprocess.run(
                shlex.split(command),
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                check=False,
                env=environment,
            )
            exit_code, stdout, stderr = completed.returncode, completed.stdout, completed.stderr
        except subprocess.TimeoutExpired as expired:
            exit_code, stdout, stderr = None, expired.stdout, expired.stderr
        return CommandOutcome(exit_code, stdout, stderr, time.perf_counter() - started)


class CommitResolver:
    """Resolves the commit the evidence belongs to; git is the authority."""

    def short_head(self) -> str:
        """Return ``git rev-parse --short HEAD`` or fail loud."""
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()


class ParquetRowCounter:
    """Row-count provenance from parquet footer metadata only (no data load)."""

    def count(self, dataset: str) -> RowProvenance:
        """Return the parquet row count, or the reason it could not be read."""
        if not dataset.lower().endswith(PARQUET_SUFFIX):
            return RowProvenance(None, None)
        try:
            metadata = pq.ParquetFile(dataset).metadata
        except (OSError, ArrowInvalid) as error:
            traceback.print_exc()
            return RowProvenance(None, f"{type(error).__name__}: {error}")
        return RowProvenance(None if metadata is None else int(metadata.num_rows), None)


class EvidenceRecorder:
    """Single owner of evidence recording: build one record and persist it."""

    def __init__(
        self,
        config: VerificationConfig,
        runner: CommandRunner,
        rows: ParquetRowCounter,
        commits: CommitResolver,
    ) -> None:
        self._config = config
        self._runner = runner
        self._rows = rows
        self._commits = commits

    @classmethod
    def from_config(cls, config: VerificationConfig) -> EvidenceRecorder:
        """Wire the recorder from config (construction lives in the factory)."""
        return cls(
            config,
            CommandRunner(config.default_timeout_seconds),
            ParquetRowCounter(),
            CommitResolver(),
        )

    def record(self, request: VerificationRequest) -> RecordingResult:
        """Run the command, write its evidence record, return the result."""
        commit = self._commits.short_head()
        outcome = self._runner.run(request.command, request.run_id)
        evidence = self._build(commit, request, outcome)
        return RecordingResult(self._write(evidence, request), evidence.exit_code)

    def _build(
        self,
        commit: str,
        request: VerificationRequest,
        outcome: CommandOutcome,
    ) -> Evidence:
        lines = self._config.output_tail_lines
        provenance = self._rows.count(request.dataset)
        return Evidence(
            run_id=request.run_id,
            commit=commit,
            step=request.step,
            command=request.command,
            dataset=request.dataset,
            rows=provenance.rows,
            rows_error=provenance.error,
            duration_seconds=outcome.duration_seconds,
            exit_code=outcome.exit_code,
            stdout_tail=self._tail(outcome.stdout, lines),
            stderr_tail=self._tail(outcome.stderr, lines),
            log_path=request.log_path,
            created_at=datetime.now(UTC),
        )

    def _write(self, evidence: Evidence, request: VerificationRequest) -> Path:
        target = self._output_dir(request)
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"{evidence.run_id}.json"
        path.write_text(evidence.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path

    def _output_dir(self, request: VerificationRequest) -> Path:
        """Configured dir is repo-relative (SSOT); a CLI override is caller-relative."""
        if request.output_dir is not None:
            return Path(request.output_dir).expanduser()
        return repo_root() / self._config.output_dir

    @staticmethod
    def _tail(output: str | bytes | None, lines: int) -> str:
        """Last ``lines`` lines of captured output (empty when there is none)."""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        if not output:
            return ""
        return "\n".join(output.splitlines()[-lines:])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--step", required=True, help="step name recorded in the evidence")
    parser.add_argument(
        "--command", required=True, help='exact command to run, e.g. "ds-pipeline etl ..."'
    )
    parser.add_argument("--dataset", required=True, help="dataset the run used (provenance)")
    parser.add_argument("--log-path", default=None, help="log file of the run (LOGGING.md)")
    parser.add_argument("--output-dir", default=None, help="override the configured output dir")
    parser.add_argument(
        "--run-id",
        default=new_run_id(),
        help="run id shared with the command's logs (default: a fresh uuid4 hex)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Record one live run and exit with its exit code."""
    try:
        request = VerificationRequest.model_validate(vars(_build_parser().parse_args(argv)))
        config = VerificationConfig.from_config(config_path(VERIFICATION_CONFIG))
        result = EvidenceRecorder.from_config(config).record(request)
    except Exception:
        traceback.print_exc()
        return 1
    print(result.path)
    if result.exit_code is None:
        print("verify_live: command produced no exit code (timed out)", file=sys.stderr)
        return 1
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
