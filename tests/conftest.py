from __future__ import annotations

import hashlib
from pathlib import Path

_SNAPSHOT_DIRS = ["artifacts", "reports"]
_snapshot: dict[str, dict[str, str]] = {}


def _tree_hashes(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not root.is_dir():
        return out
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def pytest_sessionstart(session: object) -> None:
    for d in _SNAPSHOT_DIRS:
        _snapshot[d] = _tree_hashes(Path(d))


def pytest_sessionfinish(session: object, exitstatus: int) -> None:
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
