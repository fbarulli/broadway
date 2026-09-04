#!/usr/bin/env python3
"""COLDPROBE-1 runner planner — PLAN-ONLY machinery for D23 (DECISIONS.md :283-286).

This script NEVER mutates the tree, the index, or the remote. Applying the
one-byte change, committing, and pushing belong to the ORCHESTRATOR (ship
ops); the runner neither chooses nor applies the byte location. Given an
explicit ``--file``/``--byte-offset`` target plus ``--note``, it emits the
five-step firing plan and exposes the metric-collection helpers.

Recorded metrics per D23:
    {push sha8, cold_bb_s, cache_hit=false, queue_s}

queue_s   = first Actions run visible for the new sha MINUS git-push return
cold_bb_s = run updated_at MINUS run created_at (Actions-API wall clock)
cache_hit = asserted False by scanning the job-logs endpoint for the
            restore-keys MISS marker; a HIT marker aborts the recording.

Negative handling: run-not-found timeout raises RunNotFoundTimeout; a terminal
conclusion other than the expected one raises NonSuccessConclusion; both
suppress the telemetry row (nothing is pasted on a red probe).

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
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

# --- Law-pinned constants (D23 + STATE.md ## TELEMETRY; not free choices) ---

# D23: the probe rides the CURRENT checkout's branch — never a hard-coded
# name. CI supplies GITHUB_REF_NAME; a local run falls back to the checked-out
# HEAD's symbolic ref.
DEFAULT_BRANCH_ENV_VARS = ("GITHUB_REF_NAME", "GIT_BRANCH", "BRANCH_NAME")

PAGE_LIMIT = "30"  # STATE.md hazard: LIST endpoints default to page size 30
EXPECTED_CONCLUSION = "success"

POLL_INTERVAL_S = 15.0
RUN_APPEAR_TIMEOUT_S = 900.0
CONCLUDE_TIMEOUT_S = 7200.0

CACHE_MISS_MARKERS = ("Cache not found",)
CACHE_HIT_MARKERS = (
    "Cache restored from key",
    "Received cache from",
)

_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")
_BRANCH_RE = re.compile(r"^[A-Za-z0-9._/-]+$")

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
    """Job logs failed the cache_hit=false assertion."""


# --- Pure, unit-testable metric helpers -------------------------------------


def parse_utc(timestamp: str) -> datetime:
    """Parse an ISO-8601 GitHub-API timestamp into an aware UTC datetime."""
    text = timestamp.strip()

    if not text:
        raise ValueError("empty timestamp")

    if text.endswith(("Z", "z")):
        text = f"{text[:-1]}+00:00"

    parsed = datetime.fromisoformat(text)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)

    return parsed.astimezone(UTC)


def seconds_between(start: datetime, end: datetime) -> float:
    """Return end-minus-start seconds; rejects naive datetimes."""
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("naive datetime rejected: probe timing must be TZ-aware")

    seconds = (end - start).total_seconds()

    if seconds < 0:
        raise ValueError(
            f"negative duration rejected: start={start!r}, end={end!r}"
        )

    return seconds


def cold_bb_seconds(created_at: str, updated_at: str) -> float:
    """cold_bb_s = run.updated_at - created_at."""
    return seconds_between(parse_utc(created_at), parse_utc(updated_at))


def queue_seconds(push_returned_at: datetime, run_created_at: str) -> float:
    """queue_s = first-run created_at - git-push return instant."""
    return seconds_between(push_returned_at, parse_utc(run_created_at))


def _valid_sha8(value: str) -> bool:
    return bool(_SHA_RE.fullmatch(value))


def select_run_for_sha(
    runs: Sequence[dict[str, Any]],
    sha8: str,
    after: datetime,
) -> dict[str, Any] | None:
    """Return the first run whose FULL head SHA matches the supplied prefix.

    GitHub's API returns ``headSha`` as the full commit SHA. The supplied
    ``sha8`` is intentionally a prefix for the D23 telemetry identifier, but
    matching is anchored in the full SHA after validating both sides as hex.
    """
    if not _valid_sha8(sha8):
        raise ValueError(f"invalid sha8: {sha8!r}")

    if after.tzinfo is None:
        raise ValueError("after must be timezone-aware")

    for run in runs:
        head_sha = str(run.get("headSha", "")).lower()

        if not re.fullmatch(r"[0-9a-f]{40}", head_sha):
            continue

        if not head_sha.startswith(sha8.lower()):
            continue

        created = run.get("createdAt")

        if not created:
            continue

        if parse_utc(str(created)) > after.astimezone(UTC):
            return run

    return None


def assert_conclusion(run: dict[str, Any]) -> str:
    """Return the terminal conclusion or raise NonSuccessConclusion."""
    status = str(run.get("status", ""))

    if status != "completed":
        raise ValueError(
            f"assert_conclusion needs a completed run, got {status!r}"
        )

    conclusion = run.get("conclusion")

    if conclusion != EXPECTED_CONCLUSION:
        raise NonSuccessConclusion(
            f"conclusion={conclusion!r}, expected={EXPECTED_CONCLUSION!r}"
        )

    return str(conclusion)


def _poll(
    probe: Callable[[], Any],
    timeout_s: float,
    interval_s: float,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> Any:
    """Generic injectable poll loop."""
    if timeout_s <= 0:
        raise ValueError("timeout_s must be > 0")

    if interval_s <= 0:
        raise ValueError("interval_s must be > 0")

    started = monotonic()

    while True:
        found = probe()

        if found is not None:
            return found

        elapsed = monotonic() - started

        if elapsed >= timeout_s:
            raise RunNotFoundTimeout(
                f"condition unmet after {timeout_s:.0f}s "
                f"(poll {interval_s:.0f}s)"
            )

        sleep(min(interval_s, max(0.0, timeout_s - elapsed)))


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
    """Poll Actions until a post-watermark run for the pushed SHA appears."""

    def probe() -> dict[str, Any] | None:
        return select_run_for_sha(fetch_page(), sha8, after)

    return _poll(
        probe,
        timeout_s,
        interval_s,
        monotonic,
        sleep,
    )


def await_terminal(
    refetch_run: Callable[[], dict[str, Any]],
    *,
    timeout_s: float,
    interval_s: float = POLL_INTERVAL_S,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Poll one run until status == completed."""
    def probe() -> dict[str, Any] | None:
        run = refetch_run()

        if str(run.get("status", "")) == "completed":
            return run

        return None

    return _poll(
        probe,
        timeout_s,
        interval_s,
        monotonic,
        sleep,
    )


def assert_cache_miss(log_text: str) -> bool:
    """Assert cache_hit=False: miss marker present, no hit marker anywhere."""
    if any(marker in log_text for marker in CACHE_HIT_MARKERS):
        raise CacheHitAssertFailed(
            "cache HIT marker found in job logs"
        )

    if not any(marker in log_text for marker in CACHE_MISS_MARKERS):
        raise CacheHitAssertFailed(
            "no restore-keys MISS marker found in job logs"
        )

    return False


def validate_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate the D23 record shape and return a plain copy."""
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
            raise ValueError(
                f"{key} must be {kinds}, got {type(record[key])}"
            )

    if record["cache_hit"] is not False:
        raise ValueError("cache_hit must be the literal False (D23)")

    if not _SHA_RE.fullmatch(record["push_sha8"]):
        raise ValueError(
            f"push_sha8 not 7-40 lowercase hex: {record['push_sha8']!r}"
        )

    if record["conclusion"] != EXPECTED_CONCLUSION:
        raise ValueError(
            f"conclusion must be {EXPECTED_CONCLUSION!r}"
        )

    for key in ("cold_bb_s", "queue_s"):
        value = record[key]

        if value < 0:
            raise ValueError(f"{key} must be >= 0, got {value}")

    if not record["run_url"].strip():
        raise ValueError("run_url must not be empty")

    return dict(record)


def format_telemetry_row(record: dict[str, Any]) -> str:
    """Render the validated record as one TELEMETRY-format pipe row."""
    clean = validate_record(record)
    return _TELEMETRY_ROW.format(**clean)


# --- Pure command builders --------------------------------------------------


def _validate_branch(branch: str) -> str:
    """Validate a branch name before embedding it in a displayed command."""
    branch = branch.strip()

    if (
        not branch
        or branch.startswith(("-", "/"))
        or ".." in branch
        or branch.endswith("/")
        or "//" in branch
        or not _BRANCH_RE.fullmatch(branch)
    ):
        raise ValueError(f"invalid branch name: {branch!r}")

    return branch


def gh_run_list_command(branch: str) -> list[str]:
    """Actions LIST command used for the baseline and new-run lookup."""
    branch = _validate_branch(branch)
    fields = "databaseId,headSha,createdAt,updatedAt,status,conclusion,url"

    return [
        "gh",
        "run",
        "list",
        "--branch",
        branch,
        "--limit",
        PAGE_LIMIT,
        "--json",
        fields,
    ]


def gh_run_watch_command(run_id: str) -> list[str]:
    """Terminal-conclusion wait for one run id."""
    if not re.fullmatch(r"[0-9]+", str(run_id)):
        raise ValueError(f"invalid Actions run id: {run_id!r}")

    return [
        "gh",
        "run",
        "watch",
        str(run_id),
        "--exit-status",
    ]


def gh_jobs_command(run_id: str) -> list[str]:
    """List job ids for the run."""
    if not re.fullmatch(r"[0-9]+", str(run_id)):
        raise ValueError(f"invalid Actions run id: {run_id!r}")

    return [
        "gh",
        "api",
        f"repos/{{owner}}/{{repo}}/actions/runs/{run_id}/jobs",
        "--paginate",
        "--jq",
        ".jobs[] | .id",
    ]


def gh_job_logs_command(job_id: str) -> list[str]:
    """Fetch one job's log text for the cache assertion scan."""
    if not re.fullmatch(r"[0-9]+", str(job_id)):
        raise ValueError(f"invalid Actions job id: {job_id!r}")

    return [
        "gh",
        "api",
        f"repos/{{owner}}/{{repo}}/actions/jobs/{job_id}/logs",
    ]


# --- Plan emission ----------------------------------------------------------


def _target_line(
    file_path: str,
    byte_offset: int,
    note: str,
) -> str:
    return (
        f"target: {file_path} @ byte-offset {byte_offset}\n"
        f"note:   {note}\n"
        "(location chosen by the ORCHESTRATOR; the runner records, never picks)"
    )


def build_plan(
    file_path: str,
    byte_offset: int,
    note: str,
    branch: str,
    appear_timeout_s: float,
    conclude_timeout_s: float,
) -> list[str]:
    """Build the five-step COLDPROBE-1 firing plan."""
    if not file_path.strip():
        raise ValueError("file path must not be empty")

    if byte_offset < 0:
        raise ValueError("byte offset must be >= 0")

    branch = _validate_branch(branch)

    if appear_timeout_s <= 0 or conclude_timeout_s <= 0:
        raise ValueError("timeouts must be > 0")

    runs_cmd = shlex.join(gh_run_list_command(branch))
    push_ref = f"HEAD:refs/heads/{branch}"

    return [
        "STEP 1 baseline:",
        f"  $ {runs_cmd}",
        "  -> freeze the newest createdAt as the pre-push watermark W.",
        "     Runs created at or before W are ignored.",
        "",
        "STEP 2 one-byte change (ORCHESTRATOR ONLY):",
        _target_line(file_path, byte_offset, note),
        "  -> apply exactly one byte, commit it, and verify the resulting",
        "     push SHA before pushing; the runner remains read-only.",
        "",
        "STEP 3 push wall clock:",
        f"  $ time git push origin {push_ref}",
        "  -> t0 = the UTC instant git push returns.",
        "     Poll STEP-1's Actions query until a completed-listing entry has",
        "     a full 40-hex headSha beginning with <push_sha8> and",
        "     createdAt > W.",
        f"     caps: appear {appear_timeout_s:.0f}s, poll {POLL_INTERVAL_S:.0f}s.",
        "     queue_s = run.createdAt - t0.",
        "     Deadline breach => RunNotFoundTimeout.",
        "",
        "STEP 4 conclusion + cold build:",
        "  $ gh run watch <run_id> --exit-status",
        "  -> wait until the selected run is terminal.",
        "     cold_bb_s = run.updatedAt - run.createdAt.",
        f"     conclusion must equal {EXPECTED_CONCLUSION!r}; otherwise",
        "     NonSuccessConclusion and no telemetry row.",
        f"     terminal-wait cap: {conclude_timeout_s:.0f}s.",
        "",
        (
            "  $ gh api 'repos/{owner}/{repo}/actions/runs/<run_id>/jobs' "
            "--paginate --jq '.jobs[] | .id'"
        ),
        "  $ gh api 'repos/{owner}/{repo}/actions/jobs/<job_id>/logs'",
        "  -> inspect EVERY job log: miss marker REQUIRED and every hit marker",
        "     FORBIDDEN.",
        f"     miss={CACHE_MISS_MARKERS!r}",
        f"     hit={CACHE_HIT_MARKERS!r}",
        "     Assertion failure => CacheHitAssertFailed and no telemetry row.",
        "     Passing scan fixes cache_hit=false.",
        "",
        "STEP 5 record:",
        "  -> emit the D23 JSON record and exactly one TELEMETRY-format row.",
        "     The ORCHESTRATOR owns pasting the successful row into",
        "     STATE.md ## TELEMETRY + the D23 addendum.",
    ]


def metrics_schema_text() -> str:
    """Return the expected JSON schema of the recorded metrics block."""
    schema = {
        "cache_hit": "literal false (asserted against every job log)",
        "cold_bb_s": "float >= 0, run.updated_at - created_at seconds",
        "conclusion": f"str == {EXPECTED_CONCLUSION!r} or recording aborts",
        "date": "str YYYY-MM-DD (UTC day of the run)",
        "push_sha8": "str, 7-40 lowercase hex (ledger rows currently use 7)",
        "queue_s": "float >= 0, first-run createdAt - push-return instant",
        "run_url": "str, Actions html_url for provenance",
    }

    return json.dumps(schema, indent=2, sort_keys=True)


def sample_telemetry_row() -> str:
    """Placeholder row showing the paste shape, never fake probe data."""
    placeholder = {
        "date": "<YYYY-MM-DD>",
        "push_sha8": "<push_sha8>",
        "conclusion": EXPECTED_CONCLUSION,
        "cold_bb_s": 0.0,
        "queue_s": 0.0,
        "cache_hit": False,
    }

    return _TELEMETRY_ROW.format(**placeholder)


def print_plan_lines(lines: list[str]) -> None:
    """Print plan stages with loud boundaries."""
    print("== COLDPROBE-1 FIRING PLAN (plan only — zero side effects)")

    for line in lines:
        print(line)


# --- CLI wiring -------------------------------------------------------------


def _current_branch() -> str:
    """Return the checked-out branch, or ``main`` for detached HEAD.

    Detached HEAD is permitted for plan-only operation; the orchestrator is
    expected to provide the actual target branch explicitly in that case.
    """
    result = subprocess.run(
        ["git", "symbolic-ref", "--short", "HEAD"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    if result.returncode != 0:
        return "main"

    branch = result.stdout.strip()

    return branch or "main"


def _environment_branch() -> str | None:
    """Return the first non-empty branch supplied by the environment."""
    for variable in DEFAULT_BRANCH_ENV_VARS:
        value = os.environ.get(variable, "").strip()

        if value:
            return value

    return None


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse and sanity-check CLI arguments; usage errors exit 2."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0]
    )

    parser.add_argument(
        "--file",
        required=True,
        help="src file holding the byte",
    )
    parser.add_argument(
        "--byte-offset",
        type=int,
        required=True,
        help="0-based byte offset of the one-byte tweak",
    )
    parser.add_argument(
        "--note",
        required=True,
        help="verbatim description of the inert byte",
    )

    branch_default = _environment_branch() or _current_branch()

    parser.add_argument(
        "--branch",
        default=branch_default,
        help=(
            "target branch "
            f"(default: {branch_default})"
        ),
    )
    parser.add_argument(
        "--appear-timeout-s",
        type=float,
        default=RUN_APPEAR_TIMEOUT_S,
    )
    parser.add_argument(
        "--conclude-timeout-s",
        type=float,
        default=CONCLUDE_TIMEOUT_S,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="also print exact commands + metrics JSON schema",
    )

    args = parser.parse_args(argv)

    if args.byte_offset < 0:
        parser.error("--byte-offset must be >= 0")

    if min(args.appear_timeout_s, args.conclude_timeout_s) <= 0:
        parser.error("timeouts must be > 0")

    try:
        _validate_branch(args.branch)
    except ValueError as exc:
        parser.error(str(exc))

    if not args.file.strip():
        parser.error("--file must not be empty")

    if not args.note.strip():
        parser.error("--note must not be empty")

    return args


def main(argv: list[str] | None = None) -> int:
    """Emit the plan; --dry-run adds commands, schema, and sample row."""
    try:
        args = parse_args(argv)

        plan = build_plan(
            args.file,
            args.byte_offset,
            args.note,
            args.branch,
            args.appear_timeout_s,
            args.conclude_timeout_s,
        )

        print_plan_lines(plan)

        if args.dry_run:
            print("== EXACT COMMAND SEQUENCE (as embedded above, in order)")
            print(metrics_schema_text())
            print("== SAMPLE TELEMETRY ROW (placeholders)")
            print(sample_telemetry_row())

        print(
            "== PLAN COMPLETE — orchestrator owns execution; "
            "runner wrote nothing"
        )
        return 0

    except ColdprobeError as error:
        print(f"FAIL coldprobe: {error}", file=sys.stderr)
        return 1

    except (OSError, subprocess.SubprocessError, ValueError) as error:
        print(f"FAIL coldprobe plan: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
