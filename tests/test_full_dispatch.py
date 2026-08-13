from __future__ import annotations

import pytest
from pydantic import ValidationError

from broadway.analysis.contracts import AnalysisContract, AnalysisMode
from broadway.config import loader
from broadway.config.loader import load_config, resolve_full_steps
from broadway.config.schema import FullStep


def test_flows_dispatch_per_mode() -> None:
    expected = {
        "test": ["discover", "etl", "contracts", "eda", "baseline", "features", "train", "evaluate"],
        "test_hypothesis": ["discover", "etl", "contracts", "eda", "baseline", "stats"],
        "test_causal": ["discover", "etl", "contracts", "eda", "baseline", "causal"],
    }
    for analysis_name, steps in expected.items():
        cfg = load_config("full", analysis=analysis_name)
        assert resolve_full_steps(cfg) == steps


def test_full_requires_analysis() -> None:
    with pytest.raises(ValueError, match="analysis contract"):
        resolve_full_steps(load_config("full"))


def test_full_step_rejects_invalid_flow_mode() -> None:
    with pytest.raises(ValidationError):
        FullStep(flows={"prediciton": "prediction"})


def test_full_dispatch_unknown_mode_raises() -> None:
    cfg = load_config("full", analysis="test")
    cfg = cfg.model_copy(
        update={
            "analysis": AnalysisContract(
                name="test",
                mode=AnalysisMode.HYPOTHESIS,
                goal="g",
                row_definition="r",
                decision_moment="d",
                available_info=["a"],
                leakage_notes=[],
                success_criterion="s",
                hypothesis={"group_column": "Borough", "group_values": ["Manhattan"]},
            ),
            "full": FullStep(flows={"prediction": "prediction"}),
        }
    )
    with pytest.raises(ValueError, match="no flow defined"):
        resolve_full_steps(cfg)


def test_full_dispatch_unknown_step_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config("full", analysis="test")
    monkeypatch.setattr(loader, "_load_yaml", lambda rel: {"steps": ["bogus_step"]})
    with pytest.raises(ValueError, match="unknown step"):
        resolve_full_steps(cfg)


def test_full_dispatch_missing_flow_file_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config("full", analysis="test")
    monkeypatch.setattr(
        loader, "_load_yaml", lambda rel: (_ for _ in ()).throw(FileNotFoundError("missing"))
    )
    with pytest.raises(ValueError, match="not found"):
        resolve_full_steps(cfg)
