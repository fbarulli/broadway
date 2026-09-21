from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

_SNAPSHOT_DIRS = ["artifacts", "reports"]
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


def _tree_hashes(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not root.is_dir():
        return out
    for p in sorted(root.rglob("*")):
        if p.is_file():
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
