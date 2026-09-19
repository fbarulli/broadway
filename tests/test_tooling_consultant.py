"""Tooling consultant + deterministic-ops pins.

Pyright is a strict consultant (advisory-only, never blocking); all tool
values resolve from configs/tooling.yaml; the shared surface has exactly
one list (scripts/main_whitelist.txt).
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]

DATA_SAFETY_RULES = [
    "reportOptionalMemberAccess",
    "reportOptionalSubscript",
    "reportOptionalCall",
    "reportGeneralTypeIssues",
    "reportAssignmentType",
    "reportReturnType",
    "reportArgumentType",
    "reportAttributeAccessIssue",
]


def _tooling() -> dict:
    return yaml.safe_load((REPO / "configs" / "tooling.yaml").read_text(encoding="utf-8"))


def test_pyrightconfig_is_strict_json_strict_mode() -> None:
    cfg = json.loads((REPO / "pyrightconfig.json").read_text(encoding="utf-8"))
    assert cfg["typeCheckingMode"] == "strict"
    assert cfg["include"] == ["src/broadway"]
    for rule in DATA_SAFETY_RULES:
        assert cfg.get(rule) is True, f"pyrightconfig.json must set {rule}=true (consultant strictness)"


def test_tooling_ssot_matches_pyrightconfig() -> None:
    tooling = _tooling()
    cfg = json.loads((REPO / "pyrightconfig.json").read_text(encoding="utf-8"))
    assert str(tooling["pyright"]["version"]) != ""
    assert tooling["pyright"]["mode"] == "strict"
    assert [str(s) for s in tooling["pyright"]["scope"]] == cfg["include"]
    assert tooling["pyright"]["advisory_only"] is True
    assert tooling["shared_surface"]["file"] == "scripts/main_whitelist.txt"


def test_pyright_advisory_has_no_hardcoded_values_and_never_blocks() -> None:
    text = (REPO / "scripts" / "pyright_advisory.sh").read_text(encoding="utf-8")
    assert "configs/tooling.yaml" in text
    assert "pyright@1.1.414" not in text, "version must resolve from configs/tooling.yaml"
    assert "src/broadway" not in text, "scope must resolve from configs/tooling.yaml"
    # Consultant law: every path exits 0 (advisory-only).
    assert text.rstrip().endswith("exit 0")
    assert "advisory — not failing build" in text


def test_shared_surface_single_source_across_scripts() -> None:
    tooling = _tooling()
    whitelist_rel = tooling["shared_surface"]["file"]
    whitelist = (REPO / whitelist_rel).read_text(encoding="utf-8")
    assert "src/" in whitelist and "scripts/" in whitelist and "configs/" in whitelist
    for script in ["check_branch_parity.sh", "main_day_sync.sh", "promote_to_main.sh"]:
        text = (REPO / "scripts" / script).read_text(encoding="utf-8")
        assert "main_whitelist.txt" in text, script
        assert "configs/tooling.yaml" in text, script


def test_promote_script_is_dry_run_default_and_never_commits() -> None:
    text = (REPO / "scripts" / "promote_to_main.sh").read_text(encoding="utf-8")
    assert "--dry-run" in text and "DRY_RUN=1" in text
    assert "git commit" not in text and "git push" not in text, "custody: script never commits/pushes"
    assert "PARITY_MAIN_ANCHOR" in text, "must print the taxi-only anchor-bump instruction"
