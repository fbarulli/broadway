from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

_SNAPSHOT_DIRS = ["artifacts", "reports"]
# Only the CLI's configured process log directory is exempt: a real CLI
# invocation in a test (subprocess) writes it by design. Config SSOT:
# configs/environment/*.yaml `log_dir`.
_LOG_DIR = Path("artifacts/logs")
_snapshot: dict[str, dict[str, str]] = {}


@pytest.fixture(autouse=True)
def _isolate_tracking_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Redirect the tracking bundle out of the shared tree.

    The etl manifest writer resolves its output from BROADWAY_TRACKING_DIR
    at call time (default: artifacts/tracking). Any test running etl for
    real — e.g. the onboarding e2e full-pipeline run — would otherwise write
    artifacts/tracking/data_manifest.json into the repo tree and trip the
    sessionfinish custody guard below. Tests that need an explicit location
    keep overriding the env var themselves.
    """
    monkeypatch.setenv("BROADWAY_TRACKING_DIR", str(tmp_path / "tracking"))


@pytest.fixture(autouse=True)
def _isolate_process_logging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Keep process log files out of the shared tree.

    CLI entry points write the configured ``artifacts/logs/<log_file>`` by
    design; the custody guard below forbids test writes there. Redirect the
    logging owner's repo root into tmp_path and reset its one-shot sentinel so
    each test either sets logging up from scratch or exercises the no-op.
    """
    from broadway.observability import logging as broadway_logging

    root = logging.getLogger()
    level_before = root.level
    monkeypatch.setattr(broadway_logging, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(broadway_logging, "_configured", False)
    yield
    for handler in list(root.handlers):
        if type(handler) in (logging.StreamHandler, logging.FileHandler):
            root.removeHandler(handler)
            handler.close()
    root.setLevel(level_before)


def _tree_hashes(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not root.is_dir():
        return out
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.is_relative_to(_LOG_DIR):
            out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def pytest_sessionstart(session: pytest.Session) -> None:    # XDIST-1b amendment (a): custody snapshot is controller-only. Under xdist,
    # workers execute session hooks too; a worker-side RuntimeError here would
    # surface as noisy internal_error (xdist remote.py hookwrapper path).
    if hasattr(session.config, "workerinput"):  # xdist worker -> skip
        return
    for d in _SNAPSHOT_DIRS:
        _snapshot[d] = _tree_hashes(Path(d))


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if hasattr(session.config, "workerinput"):  # xdist worker -> skip
        return
    problems = []
    for d in _SNAPSHOT_DIRS:
        before = _snapshot.get(d, {})
        after = _tree_hashes(Path(d))
        added = sorted(set(after) - set(before))
        removed = sorted(set(before) - set(after))
        modified = sorted(p for p in set(before) & set(after) if before[p] != after[p])
        for p in added:
            problems.append(f"ADDED {p}")
        for p in removed:
            problems.append(f"REMOVED {p}")
        for p in modified:
            problems.append(f"MODIFIED {p}")
    if problems:
        raise RuntimeError(
            "test suite mutated shared runtime directories:\n" + "\n".join(problems)
        )
