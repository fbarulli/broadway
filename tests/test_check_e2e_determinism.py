"""scripts/check_e2e_determinism.sh — core comparator tests.

Synthetic fixtures only (tmp_path trees — no pipelines, no MLflow server):
the comparator is exercised via subprocess exactly as a developer would run
it. Same philosophy as tests/test_platform_hygiene.py: the standing bar is
enforced structurally, not by prose. Whitelist SSOT = the script's
EXACT/PATTERN table; this docstring restates it only as a pointer.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_e2e_determinism.sh"

# One canonical evaluation-style doc exercising every whitelisted field
# (trace.created_at, promote, reason, warnings, comparison.metrics.*.champion).
_BASE = {
    "metrics": {"mae": 3.9434, "rmse": 5.2435},
    "cv_metrics": {"mae": 6.6658, "rmse": 7.7246},
    "promote": True,
    "reason": "no champion model — promoting unconditionally",
    "warnings": ["no champion model found — candidate compared against none"],
    "comparison": {
        "metrics": {
            "mae": {
                "candidate": 3.9434,
                "champion": None,
                "delta": None,
                "delta_pct": None,
            },
            "rmse": {
                "candidate": 5.2435,
                "champion": None,
                "delta": None,
                "delta_pct": None,
            },
        }
    },
    "trace": {"created_at": "2026-08-22T13:43:14.434424Z", "commit": "c0fd39e"},
}


def _copy(payload: dict) -> dict:
    return json.loads(json.dumps(payload))


def _write_tree(root: Path, docs: dict[str, dict]) -> None:
    for rel, payload in docs.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _run(*dirs: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SCRIPT), *(str(d) for d in dirs)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=150,  # fail fast instead of hanging on a uv probe stall (2026-08-23)
    )


def test_identical_trees_report_ok(tmp_path: Path) -> None:
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    _write_tree(run1, {"evaluation/metrics.json": _BASE})
    _write_tree(run2, {"evaluation/metrics.json": _BASE})
    result = _run(run1, run2)
    assert result.returncode == 0
    assert "DETERMINISM OK" in result.stdout


def test_differing_numerics_fail_with_field_path(tmp_path: Path) -> None:
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    changed = _copy(_BASE)
    changed["metrics"]["rmse"] = 6.0
    _write_tree(run1, {"evaluation/metrics.json": _BASE})
    _write_tree(run2, {"evaluation/metrics.json": changed})
    result = _run(run1, run2)
    assert result.returncode == 1
    assert "evaluation/metrics.json: metrics.rmse" in result.stdout


def test_only_whitelisted_fields_may_differ(tmp_path: Path) -> None:
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    changed = _copy(_BASE)
    changed["trace"]["created_at"] = "2026-08-22T14:00:00.000000Z"
    changed["promote"] = False
    changed["reason"] = "candidate beat champion"
    changed["warnings"] = ["candidate compared against champion"]
    changed["comparison"]["metrics"]["rmse"]["champion"] = 5.1
    _write_tree(
        run1,
        {
            "evaluation/metrics.json": _BASE,
            "training/training_result.json": {
                "model_type": "linear",
                "artifact_path": "models:/m-one",
            },
        },
    )
    _write_tree(
        run2,
        {
            "evaluation/metrics.json": changed,
            "training/training_result.json": {
                "model_type": "linear",
                "artifact_path": "models:/m-two",
            },
        },
    )
    result = _run(run1, run2)
    assert result.returncode == 0
    assert "DETERMINISM OK" in result.stdout


def test_champion_derived_delta_fields_may_differ(tmp_path: Path) -> None:
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    changed = _copy(_BASE)
    changed["comparison"]["metrics"]["rmse"]["champion"] = 5.1
    changed["comparison"]["metrics"]["rmse"]["delta"] = 0.1435
    changed["comparison"]["metrics"]["rmse"]["delta_pct"] = 2.74
    _write_tree(run1, {"evaluation/metrics.json": _BASE})
    _write_tree(run2, {"evaluation/metrics.json": changed})
    result = _run(run1, run2)
    assert result.returncode == 0
    assert "DETERMINISM OK" in result.stdout


def test_missing_counterpart_file_is_reported(tmp_path: Path) -> None:
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    _write_tree(
        run1,
        {
            "evaluation/metrics.json": _BASE,
            "training/training_result.json": {
                "model_type": "linear",
                "artifact_path": "models:/m-one",
            },
        },
    )
    _write_tree(run2, {"evaluation/metrics.json": _BASE})
    result = _run(run1, run2)
    assert result.returncode == 1
    assert "training/training_result.json: missing counterpart" in result.stdout
