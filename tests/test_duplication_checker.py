"""Public behavior: cross-file structural duplicates are reported, unique code is not."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_duplication import main

SHARED_BODY = """    total = 0
    count = 0
    peak = None
    floor = min(values)
    ceiling = max(values)
    for value in values:
        total += value
        count += 1
        if peak is None or value > peak:
            peak = value
    if count:
        total = total / count
    spread = ceiling - floor
    return total, count, peak, spread
"""

UNIQUE_BODY = """    ordered = sorted(values)
    first = ordered[0]
    last = ordered[-1]
    span = last - first
    midpoint = (first + last) / 2
    shifted = [item + 1 for item in ordered]
    doubled = [item * 2 for item in ordered]
    return first, last, span, midpoint, shifted, doubled
"""


def test_cross_file_duplicate_reported_and_unique_ignored(tmp_path: Path) -> None:
    """Two structurally identical functions group; a different function stays out."""
    left = tmp_path / "owners" / "left"
    right = tmp_path / "owners" / "right"
    left.mkdir(parents=True)
    right.mkdir()
    (left / "alpha.py").write_text(f"def shared(values):\n{SHARED_BODY}", encoding="utf-8")
    (right / "beta.py").write_text(
        f"def shared(values):\n{SHARED_BODY}\n\ndef unique(values):\n{UNIQUE_BODY}",
        encoding="utf-8",
    )
    report_path = tmp_path / "duplicates.json"
    config_path = tmp_path / "duplication.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "advisory": True,
                "min_statements": 8,
                "scan_roots": [str(left), str(right)],
                "exclude": ["**/__init__.py", "**/test_*.py"],
                "report": str(report_path),
            }
        ),
        encoding="utf-8",
    )

    assert main(["--config", str(config_path)]) == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    groups = payload["duplicate_groups"]
    assert len(groups) == 1
    assert {(site["file"], site["function"]) for site in groups[0]["sites"]} == {
        (str(left / "alpha.py"), "shared"),
        (str(right / "beta.py"), "shared"),
    }
    assert groups[0]["statements"] >= 8
    assert all(
        site["function"] != "unique" for group in groups for site in group["sites"]
    )
