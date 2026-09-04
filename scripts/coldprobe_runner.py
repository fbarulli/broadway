#!/usr/bin/env python3
"""COLDPROBE-1 runner planner — PLAN-ONLY machinery for D23 (DECISIONS.md :283-286).

This script NEVER mutates the tree, the index, or the remote. Applying the
one-byte change, committing, and pushing belong to the ORCHESTRATOR (ship
ops); the runner neither chooses nor applies the byte location. Given an
explicit ``--file``/``--byte-offset`` target plus ``--note``, it emits the
five-step firing plan and exposes the metric-collection helpers.

Recorded metrics per D23: ``{push sha8, cold_bb_s, cache_hit=false, queue_s}``

  queue_s   = first Actions run visible for the new sha MINUS git-push return
  cold_bb_s = run updated_at MINUS run created_at (Actions-API wall clock)
  cache_hit = asserted False by scanning the job-logs endpoint for the
              restore-keys MISS marker; a HIT marker aborts the recording.

Negative handling: run-not-found timeout raises RunNotFoundTimeout; a
terminal conclusion other than the expected one raises NonSuccessConclusion;
both suppress the telemetry row (nothing is pasted on a red probe).

Usage (plan only; ``--dry-run`` adds the exact command sequence + schema)::

    python3 scripts/coldprobe_runner.py --file src/broadway/module.py \
        --byte-offset 1234 --note "inert whitespace byte" --dry-run

Stdlib only; pure helpers are importable and unit-testable without network.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

# --- Law-pinned constants (D23 + STATE.md ## TELEMETRY; not free choices) ---
# D23: the probe rides the CURRENT checkout's branch — never a hard-coded
# name (say-it-once law 2026-09-02). CI supplies GITHUB_REF_NAME; a local
# run falls back to the checked-out HEAD's symbolic ref.
DEFAULT_BRANCH_ENV_VARS = ("GITHUB_REF_NAME", "GIT_BRANCH", "BRANCH_NAME")
PAGE_LIMIT = "30"  # STATE.md hazard: LIST endpoints default to page size 30
EXPECTED_CONCLUSION = "success"  # a green tip is the recording precondition
POLL_INTERVAL_S = 15.0
RUN_APPEAR_TIMEOUT_S = 900.0  # push-return -> first run visible (queue window)
CONCLUDE_TIMEOUT_S = 7200.0  # terminal-conclusion wait ceiling

# Job-log wording of actions/cache@v4 (ci.yml 'Restore base image'); the
# setup-uv enable-cache surface logs different lines — markers are constants
# so a live-run owner can re-pin them WITHOUT editing logic.
CACHE_MISS_MARKERS = ("Cache not found",)
CACHE_HIT_MARKERS = ("Cache restored from key", "Received cache from")

_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")
_TELEMETRY_ROW = (
    "| {date} | {push_sha8} | {conclusion} "
    "| n/a (post-bar COLDPROBE-1: cold_bb_s={cold_bb_s:.1f}s "
    "queue_s={queue_s:.1f}s cache_hit=false) |"
)


class ColdprobeError(RuntimeError):
    """Base class for recoverable probe-recording failures."""


class RunNotFoundTimeout(ColdprobeError):
    """No Actions run for the pushed sha appeared before the deadline."""


class NonSuccessConclusion(ColdprobeError):
    """The run concluded, but not with the expected green conclusion."""


class CacheHitAssertFailed(ColdprobeError):
    """Job logs failed the cache_hit=false assertion (hit seen, or no miss)."""


# --- Pure, unit-testable metric helpers (no I/O, no globals) ---
def parse_utc(timestamp: str) -> datetime:
    """Parse an ISO-8601 GitHub-API timestamp into an aware UTC datetime."""
    text = timestamp.strip()
    if text.endswith(("Z", "z")):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:  # defensive: treat bare stamps as UTC, never naive
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def seconds_between(start: datetime, end: datetime) -> float:
    """Return end-minus-start seconds; rejects naive datetimes (DTZ-safe)."""
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("naive datetime rejected: probe timing must be TZ-aware")
    return (end - start).total_seconds()


def cold_bb_seconds(created_at: str, updated_at: str) -> float:
    """cold_bb_s = run updated_at - created_at, from raw API stamp strings."""
    return seconds_between(parse_utc(created_at), parse_utc(updated_at))


def queue_seconds(push_returned_at: datetime, run_created_at: str) -> float:
    """queue_s = first-run created_at minus the instant ``git push`` returned."""
    return seconds_between(push_returned_at, parse_utc(run_created_at))


def select_run_for_sha(
    runs: list[dict[str, Any]], sha8: str, after: datetime
) -> dict[str, Any] | None:
    """First page entry whose headSha starts with sha8 and was created after.

    Caller supplies the page in GitHub order (newest first); "first" is that
    order, i.e. the earliest listing match, not the oldest run overall.
    """
    for run in runs:
        head_sha = str(run.get("headSha", ""))
        created = run.get("createdAt")
        if head_sha.startswith(sha8) and created and parse_utc(str(created)) > after:
            return run
    return None


def assert_conclusion(run: dict[str, Any]) -> str:
    """Return the terminal conclusion or raise NonSuccessConclusion."""
    conclusion = run.get("conclusion")
    status = str(run.get("status", ""))
    if status != "completed":
        raise ValueError(f"assert_conclusion needs a completed run, got {status!r}")
    if conclusion != EXPECTED_CONCLUSION:
        raise NonSuccessConclusion(
            f"conclusion={conclusion!r}, expected {EXPECTED_CONCLUSION!r}"
        )
    return str(conclusion)


def _poll(
    probe: Callable[[], Any],
    timeout_s: float,
    interval_s: float,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> Any:
    """Generic poll loop; injectable clock/sleep keep it network-free to test."""
    started = monotonic()
    while True:
        found = probe()
        if found is not None:
            return found
        if monotonic() - started >= timeout_s:
            raise RunNotFoundTimeout(
                f"condition unmet after {timeout_s:.0f}s (poll {interval_s:.0f}s)"
            )
        sleep(interval_s)


def await_new_run(
    fetch_page: Callable[[], list[dict[str, Any]]],
    *,
    sha8: str,
    after: datetime,
    timeout_s: float,
    interval_s: float = POLL_INTERVAL_S,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Poll the Actions page until the pushed sha's first run is visible."""
    def probe() -> dict[str, Any] | None:
        return select_run_for_sha(fetch_page(), sha8, after)

    return _poll(probe, timeout_s, interval_s, monotonic, sleep)


def await_terminal(
    refetch_run: Callable[[], dict[str, Any]],
    *,
    timeout_s: float,
    interval_s: float = POLL_INTERVAL_S,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Poll one run until ``status == completed``; deadline => RunNotFoundTimeout."""
    def probe() -> dict[str, Any] | None:
        run = refetch_run()
        return run if str(run.get("status", "")) == "completed" else None

    return _poll(probe, timeout_s, interval_s, monotonic, sleep)


def assert_cache_miss(log_text: str) -> bool:
    """Assert cache_hit=False: a miss marker present, no hit marker anywhere."""
    if any(marker in log_text for marker in CACHE_HIT_MARKERS):
        raise CacheHitAssertFailed("cache HIT marker found in job logs")
    if not any(marker in log_text for marker in CACHE_MISS_MARKERS):
        raise CacheHitAssertFailed("no restore-keys MISS marker found in job logs")
    return False  # the recorded value of cache_hit


def validate_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate the D23 record shape; returns a plain copy or raises ValueError."""
    required = {
        "date": str,
        "push_sha8": str,
        "conclusion": str,
        "cold_bb_s": (int, float),
        "queue_s": (int, float),
        "cache_hit": bool,
        "run_url": str,
    }
    missing = sorted(set(required) - set(record))
    if missing:
        raise ValueError(f"record missing keys: {missing}")
    for key, kinds in required.items():
        if not isinstance(record[key], kinds):
            raise ValueError(f"{key} must be {kinds}, got {type(record[key])}")
    if not isinstance(record["cache_hit"], bool) or record["cache_hit"] is not False:
        raise ValueError("cache_hit must be the literal False (D23)")
    if not _SHA_RE.match(record["push_sha8"]):
        raise ValueError(f"push_sha8 not 7-40 lowercase hex: {record['push_sha8']!r}")
    for key in ("cold_bb_s", "queue_s"):
        if record[key] < 0:
            raise ValueError(f"{key} must be >= 0, got {record[key]}")
    return dict(record)


def format_telemetry_row(record: dict[str, Any]) -> str:
    """Render the validated record as one TELEMETRY-format pipe row."""
    clean = validate_record(record)
    return _TELEMETRY_ROW.format(**clean)


# --- Pure command builders (printed verbatim; orchestrator executes) ---
def gh_run_list_command(branch: str) -> list[str]:
    """Baseline/new-sha Actions query class (STATE.md re-derive command family)."""
    fields = "databaseId,headSha,createdAt,updatedAt,status,conclusion,url"
    return ["gh", "run", "list", "--branch", branch, "--limit", PAGE_LIMIT,
            "--json", fields]


def gh_run_watch_command(run_id: str) -> list[str]:
    """Terminal-conclusion wait for one run id (placeholder-friendly)."""
    return ["gh", "run", "watch", run_id, "--exit-status"]


def gh_jobs_command(run_id: str) -> list[str]:
    """List job ids for the run (job-logs endpoint discovery step)."""
    return ["gh", "api", f"repos/{{owner}}/{{repo}}/actions/runs/{run_id}/jobs",
            "--jq", ".jobs[] | .id"]


def gh_job_logs_command(job_id: str) -> list[str]:
    """Fetch one job's log text for the cache-miss assertion scan."""
    return ["gh", "api", f"repos/{{owner}}/{{repo}}/actions/jobs/{job_id}/logs"]


# --- Plan emission (stage-boundary prints only; no per-loop logging) ---
def _target_line(file_path: str, byte_offset: int, note: str) -> str:
    return (
        f"target: {file_path} @ byte-offset {byte_offset}\n"
        f"note:   {note}\n"
        "(location chosen by the ORCHESTRATOR; the runner records, never picks)"
    )


def build_plan(
    file_path: str, byte_offset: int, note: str, branch: str,
    appear_timeout_s: float, conclude_timeout_s: float,
) -> list[str]:
    """The five-step COLDPROBE-1 firing plan, commands included verbatim."""
    runs_cmd = shlex.join(gh_run_list_command(branch))
    return [
        "STEP 1 baseline:",
        f"  $ {runs_cmd}",
        "  -> freeze newest createdAt as the pre-push watermark W (ignore runs <= W).",
        f"STEP 2 one-byte change (ORCHESTRATOR ONLY):\n{_target_line(file_path, byte_offset, note)}",
        "  -> apply, commit, and push from the orchestrator lane; runner stays read-only.",
        "STEP 3 push wall clock:",
        "  $ time git push origin HEAD:<branch>",
        "  -> t0 = UTC instant push returns; poll STEP-1 query class until a run with",
        "     headSha startswith <push_sha8> and createdAt > W appears.",
        f"     caps: appear {appear_timeout_s:.0f}s, poll {POLL_INTERVAL_S:.0f}s.",
        "     queue_s = that run's createdAt - t0. Deadline breach => RunNotFoundTimeout.",
        "STEP 4 conclusion + cold build:",
        f"  $ {shlex.join(gh_run_watch_command('{run_id}'))}",
        f"     (or poll the STEP-1 query; terminal-wait cap {conclude_timeout_s:.0f}s)",
        "  -> cold_bb_s = run.updated_at - run.created_at;",
        f"     conclusion must equal {EXPECTED_CONCLUSION!r} else NonSuccessConclusion.",
        f"  $ {shlex.join(gh_jobs_command('{run_id}'))}",
        f"  $ {shlex.join(gh_job_logs_command('{job_id}'))}   # per job id",
        "  -> scan every job log: miss marker REQUIRED, hit marker FORBIDDEN",
        f"     (miss={CACHE_MISS_MARKERS!r}; hit={CACHE_HIT_MARKERS!r})",
        "     else CacheHitAssertFailed; passing scan fixes cache_hit=false.",
        "STEP 5 record:",
        "  -> stdout emits the D23 JSON record + one TELEMETRY-format row",
        "     for the orchestrator to paste into STATE.md ## TELEMETRY + D23 addendum.",
    ]


def metrics_schema_text() -> str:
    """Expected JSON schema of the recorded metrics block (dry-run display)."""
    schema = {
        "date": "str YYYY-MM-DD (UTC day of the run)",
        "push_sha8": "str, 7-40 lowercase hex (ledger rows currently use 7)",
        "cold_bb_s": "float >= 0, run.updated_at - created_at seconds",
        "cache_hit": "literal false (asserted against job logs)",
        "queue_s": "float >= 0, first-run createdAt - push-return instant",
        "conclusion": f"str == {EXPECTED_CONCLUSION!r} or recording aborts",
        "run_url": "str, Actions html_url for provenance",
    }
    return json.dumps(schema, indent=2, sort_keys=True)


def sample_telemetry_row() -> str:
    """Placeholder row showing the paste shape (tokens, never fake data)."""
    placeholder = {
        "date": "<YYYY-MM-DD>", "push_sha8": "<push_sha8>",
        "conclusion": EXPECTED_CONCLUSION, "cold_bb_s": 0.0, "queue_s": 0.0,
        "cache_hit": False,
    }
    return _TELEMETRY_ROW.format(**placeholder)


def print_plan_lines(lines: list[str]) -> None:
    """Print plan stages with loud boundaries (strategic logging only)."""
    print("== COLDPROBE-1 FIRING PLAN (plan only — zero side effects)")
    for line in lines:
        print(line)


# --- CLI wiring ---
def _current_branch() -> str:
    """Current checkout's branch name; 'main' fallback for detached HEAD.

    A hard failure here would make plan-only invocations impossible on a
    detached HEAD, so the fallback keeps the probe usable while the
    orchestrator supplies the explicit --branch it knows.
    """
    result = subprocess.run(
        ["git", "symbolic-ref", "--short", "HEAD"],
        capture_output=True, text=True, timeout=10, check=False,
    )
    branch = result.stdout.strip()
    return branch or "main"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse and sanity-check CLI arguments; usage errors exit 2."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", required=True, help="src file holding the byte")
    parser.add_argument("--byte-offset", type=int, required=True,
                        help="0-based byte offset of the one-byte tweak")
    parser.add_argument("--note", required=True,
                        help="verbatim description of the inert byte change")
    branch_default = next(
        (os.environ[var] for var in DEFAULT_BRANCH_ENV_VARS if os.environ.get(var)),
        None,
    ) or _current_branch()
    parser.add_argument("--branch", default=branch_default,
                        help=f"target branch (default: current checkout's branch: {branch_default})")
    parser.add_argument("--appear-timeout-s", type=float,
                        default=RUN_APPEAR_TIMEOUT_S)
    parser.add_argument("--conclude-timeout-s", type=float,
                        default=CONCLUDE_TIMEOUT_S)
    parser.add_argument("--dry-run", action="store_true",
                        help="also print exact commands + metrics JSON schema")
    args = parser.parse_args(argv)
    if args.byte_offset < 0:
        parser.error("--byte-offset must be >= 0")
    if min(args.appear_timeout_s, args.conclude_timeout_s) <= 0:
        parser.error("timeouts must be > 0")
    return args


def main(argv: list[str] | None = None) -> int:
    """Emit the plan (and, under --dry-run, commands/schema/sample row)."""
    try:
        args = parse_args(argv)
        plan = build_plan(args.file, args.byte_offset, args.note, args.branch,
                          args.appear_timeout_s, args.conclude_timeout_s)
        print_plan_lines(plan)
        if args.dry_run:
            print("== EXACT COMMAND SEQUENCE (as embedded above, in order)")
            print(metrics_schema_text())
            print("== SAMPLE TELEMETRY ROW (placeholders)")
            print(sample_telemetry_row())
        print("== PLAN COMPLETE — orchestrator owns execution; runner wrote nothing")
        return 0
    except ColdprobeError as error:
        print(f"FAIL coldprobe: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
