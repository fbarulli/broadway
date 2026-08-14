from __future__ import annotations

from pathlib import Path

import pytest

from broadway.config.loader import load_config
from broadway.timeline import decide, module


def _analysis():
    return load_config("stats", dataset="taxi", analysis="taxi_hypothesis").analysis


def test_record_writes_valid_decision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "TIMELINE_DIR", tmp_path / "timeline")
    decision = decide.record(
        _analysis(),
        "omnibus",
        "welch",
        "non-normal",
        ["describe_groups", "normality", "variance"],
    )
    module.save_decision(decision)
    loaded = module.load_decision("taxi_hypothesis", "omnibus")
    assert loaded is not None
    assert loaded == decision
    assert loaded.kind == "omnibus"
    assert loaded.id == "omnibus"
    assert loaded.method == "welch"
    assert loaded.status == "resolved"
    assert loaded.reason == ["non-normal"]
    assert loaded.parents == ["describe_groups", "normality", "variance"]


def test_record_rejects_invalid_method() -> None:
    analysis = _analysis()
    with pytest.raises(ValueError):
        decide.record(analysis, "omnibus", "games_howell", "x", ["describe_groups"])
    with pytest.raises(ValueError):
        decide.record(analysis, "posthoc", "welch", "x", ["omnibus"])


def test_record_rejects_unknown_kind() -> None:
    analysis = _analysis()
    with pytest.raises(ValueError):
        decide.record(analysis, "bogus", "welch", "x", [])
