from __future__ import annotations

import pytest
from pydantic import ValidationError

from broadway.analysis.contracts import AnalysisContract, AnalysisMode
from broadway.config.loader import load_config


def test_valid_contract_parses() -> None:
    contract = AnalysisContract(
        mode="prediction",
        goal="g",
        row_definition="r",
        decision_moment="d",
        available_info=["a"],
        leakage_notes=[],
        success_criterion="s",
    )
    assert contract.mode == AnalysisMode.PREDICTION


def test_missing_required_field_raises() -> None:
    with pytest.raises(ValidationError):
        AnalysisContract(
            mode="prediction",
            row_definition="r",
            decision_moment="d",
            available_info=["a"],
            leakage_notes=[],
            success_criterion="s",
        )


def test_empty_goal_rejected() -> None:
    with pytest.raises(ValidationError):
        AnalysisContract(
            mode="prediction",
            goal="   ",
            row_definition="r",
            decision_moment="d",
            available_info=["a"],
            leakage_notes=[],
            success_criterion="s",
        )


def test_empty_available_info_rejected() -> None:
    with pytest.raises(ValidationError):
        AnalysisContract(
            mode="prediction",
            goal="g",
            row_definition="r",
            decision_moment="d",
            available_info=[],
            leakage_notes=[],
            success_criterion="s",
        )


def test_invalid_mode_rejected() -> None:
    with pytest.raises(ValidationError):
        AnalysisContract(
            mode="bogus",
            goal="g",
            row_definition="r",
            decision_moment="d",
            available_info=["a"],
            leakage_notes=[],
            success_criterion="s",
        )


def test_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        AnalysisContract(
            mode="prediction",
            goal="g",
            row_definition="r",
            decision_moment="d",
            available_info=["a"],
            leakage_notes=[],
            success_criterion="s",
            unexpected="field",
        )


def test_loader_wires_analysis() -> None:
    cfg = load_config("stats", analysis="taxi")
    assert cfg.analysis is not None
    assert cfg.analysis.mode == AnalysisMode.PREDICTION
    assert cfg.analysis.goal
