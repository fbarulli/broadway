"""Graphify surface-reconciliation gate (GATE-INFRA-149) — falsifiable probe.

Mirrors the probe-e pattern (test_governance_probes.py): a LIVE half that must
stay green at HEAD against the committed ``graphify-out/graph.json``, plus
falsifiability proofs that seed an unmapped ghost and prove the reconciler
goes RED. The reconciler itself is ``scripts/check_graphify_surfaces.py``;
this test imports its pure functions and exercises them against fixtures, so
the live half never mutates the tree.

Contract under test:
  * a governed file (``src/`` or ``project/``, non-test) that carries callable
    symbols and is referenced by NO gate ``owner:`` FAILS — unless it is in the
    ``KNOWN_UNMAPPED`` baseline;
  * FUNCTION-level "not individually named in an owner" is a WARN, never a
    failure;
  * the ``KNOWN_UNMAPPED`` baseline is the exact unmapped set at HEAD — if a
    file in it gains an owner (or a new file becomes unowned) the baseline must
    be updated or this suite goes red.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from check_graphify_surfaces import (
    KNOWN_UNMAPPED,
    callable_name,
    extract_callables,
    owner_paths,
    reconcile,
)

GRAPH = REPO / "graphify-out" / "graph.json"
GATES = REPO / "agents" / "ledger" / "gates.yaml"


def _graph_nodes() -> list[dict]:
    return json.loads(GRAPH.read_text(encoding="utf-8"))["nodes"]


def _gate_rows() -> list[dict]:
    return yaml.safe_load(GATES.read_text(encoding="utf-8"))["gates"]


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_callable_name_classifies_labels() -> None:
    assert callable_name("run()") == "run"
    assert callable_name("_assert_unique_merged_labels()") == "_assert_unique_merged_labels"
    assert callable_name(".predict()") == "predict"          # method (leading dot)
    assert callable_name(".__init__()") == "__init__"         # dunder method
    assert callable_name("ProjectPaths") == "ProjectPaths"    # class
    assert callable_name("_StopWalkthrough") == "_StopWalkthrough"
    assert callable_name("01_eda.py") is None                 # file-level node
    assert callable_name("analysis/__init__.py") is None      # package file node
    assert callable_name("broadway") is None                  # package/var, not callable


def test_owner_paths_splits_multi_path_owners() -> None:
    assert owner_paths("src/broadway/data/loader.py:124 load_with_audit()") == [
        "src/broadway/data/loader.py",
    ]
    # ' + <path>' boundary keeps the second owner path when a repo path follows
    multi = ("src/broadway/lineage/state.py:5 LINEAGE_STEPS "
             "+ src/broadway/baseline/module.py:87-91 parents")
    assert "src/broadway/lineage/state.py" in owner_paths(multi)
    assert "src/broadway/baseline/module.py" in owner_paths(multi)


# --------------------------------------------------------------------------- #
# LIVE probe — green at HEAD
# --------------------------------------------------------------------------- #
def test_live_reconciliation_green_at_head() -> None:
    nodes, gates = _graph_nodes(), _gate_rows()
    _, mapped, unmapped_fail, _ = reconcile(nodes, gates)
    assert unmapped_fail == [], f"unowned governed files at HEAD: {unmapped_fail}"
    assert mapped, "reconciliation found no owner-mapped governed files"


def test_known_unmapped_baseline_is_the_exact_unmapped_set() -> None:
    """Drift guard: KNOWN_UNMAPPED must equal the truly-unowned set at HEAD.

    With an EMPTY allowlist, the reconciler's failure set must be exactly the
    committed baseline — if a baseline file gains an owner, or a new file
    becomes unowned, this pins the drift so the baseline is updated in-step.
    """
    nodes, gates = _graph_nodes(), _gate_rows()
    _, _, unmapped_fail, _ = reconcile(nodes, gates, allowlist=())
    assert sorted(unmapped_fail) == sorted(KNOWN_UNMAPPED), (
        "KNOWN_UNMAPPED drift: the committed baseline no longer equals the "
        "unowned set — remove newly-owned files from KNOWN_UNMAPPED or add "
        "newly-unowned files to it"
    )


# --------------------------------------------------------------------------- #
# Falsifiability — a synthetic unmapped file must go RED
# --------------------------------------------------------------------------- #
def _ghost_nodes() -> list[dict]:
    return [
        {
            "id": "src_broadway_zzz_ghost_probe_zzz_ghost",
            "label": "zzz_ghost()",
            "file_type": "code",
            "source_file": "src/broadway/zzz_ghost_probe/zzz_ghost.py",
            "source_location": "L1",
            "_origin": "ast",
        },
    ]


def test_synthetic_unmapped_file_fails() -> None:
    nodes, gates = _graph_nodes(), _gate_rows()
    seeded = nodes + _ghost_nodes()
    _, _, unmapped_fail, _ = reconcile(seeded, gates)
    assert "src/broadway/zzz_ghost_probe/zzz_ghost.py" in unmapped_fail
    # and the baseline still holds: only the ghost is new
    _, _, baseline_fail, _ = reconcile(seeded, gates, allowlist=())
    assert set(baseline_fail) == set(KNOWN_UNMAPPED) | {
        "src/broadway/zzz_ghost_probe/zzz_ghost.py"
    }


def test_extract_callables_scopes_to_governed_non_test_files() -> None:
    nodes = _graph_nodes() + [
        {  # test file under a governed root — must NOT count as a callable surface
            "id": "project_tests_zzz",
            "label": "test_zzz()",
            "file_type": "code",
            "source_file": "project/tests/test_zzz.py",
            "source_location": "L1",
            "_origin": "ast",
        },
        {  # non-governed root — must NOT count either
            "id": "scripts_zzz",
            "label": "zzz()",
            "file_type": "code",
            "source_file": "scripts/zzz.py",
            "source_location": "L1",
            "_origin": "ast",
        },
    ]
    callables = extract_callables(nodes)
    assert "project/tests/test_zzz.py" not in callables
    assert "scripts/zzz.py" not in callables
    assert "src/broadway/zzz_ghost_probe/zzz_ghost.py" not in callables  # not in nodes


def test_warn_tier_never_fails() -> None:
    """A covered file with unnamed helpers warns but must not fail the gate."""
    nodes, gates = _graph_nodes(), _gate_rows()
    callables, mapped, unmapped_fail, unnamed = reconcile(nodes, gates)
    assert unmapped_fail == []
    # src/broadway/data/loader.py is owner-mapped; `load` is a helper that is
    # NOT individually named in any owner — so it must surface as a WARN only.
    assert "src/broadway/data/loader.py" in mapped
    assert "load" in callables["src/broadway/data/loader.py"]
    assert "load" in unnamed.get("src/broadway/data/loader.py", set())
