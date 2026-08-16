from __future__ import annotations

import pytest
from pydantic import ValidationError

from broadway.analysis.contracts import AnalysisContract, AnalysisMode, require_mode
from broadway.config.loader import load_config


def test_valid_contract_parses() -> None:
    contract = AnalysisContract(
        name="test",
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
            name="test",
            mode="prediction",
            row_definition="r",
            decision_moment="d",
            available_info=["a"],
            leakage_notes=[],
            success_criterion="s",
        )


def test_missing_name_raises() -> None:
    with pytest.raises(ValidationError):
        AnalysisContract(
            mode="prediction",
            goal="g",
            row_definition="r",
            decision_moment="d",
            available_info=["a"],
            leakage_notes=[],
            success_criterion="s",
        )


def test_empty_goal_rejected() -> None:
    with pytest.raises(ValidationError):
        AnalysisContract(
            name="test",
            mode="prediction",
            goal="   ",
            row_definition="r",
            decision_moment="d",
            available_info=["a"],
            leakage_notes=[],
            success_criterion="s",
        )


def test_empty_name_rejected() -> None:
    with pytest.raises(ValidationError):
        AnalysisContract(
            name="   ",
            mode="prediction",
            goal="g",
            row_definition="r",
            decision_moment="d",
            available_info=["a"],
            leakage_notes=[],
            success_criterion="s",
        )


def test_empty_available_info_rejected() -> None:
    with pytest.raises(ValidationError):
        AnalysisContract(
            name="test",
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
            name="test",
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
            name="test",
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
    cfg = load_config("stats", analysis="test")
    assert cfg.analysis is not None
    assert cfg.analysis.mode == AnalysisMode.PREDICTION
    assert cfg.analysis.goal
    assert cfg.analysis.name == "test"


def test_require_mode_ok() -> None:
    contract = AnalysisContract(
        name="test",
        mode="prediction",
        goal="g",
        row_definition="r",
        decision_moment="d",
        available_info=["a"],
        leakage_notes=[],
        success_criterion="s",
    )
    assert require_mode(contract, AnalysisMode.PREDICTION) == contract


def test_require_mode_mismatch_raises() -> None:
    contract = AnalysisContract(
        name="test",
        mode="prediction",
        goal="g",
        row_definition="r",
        decision_moment="d",
        available_info=["a"],
        leakage_notes=[],
        success_criterion="s",
    )
    with pytest.raises(ValueError):
        require_mode(contract, AnalysisMode.HYPOTHESIS)


def test_require_mode_none_raises() -> None:
    with pytest.raises(ValueError):
        require_mode(None, AnalysisMode.PREDICTION)


def test_hypothesis_mode_requires_hypothesis_block() -> None:
    with pytest.raises(ValidationError, match="hypothesis"):
        AnalysisContract(
            name="test",
            mode="hypothesis",
            goal="g",
            row_definition="r",
            decision_moment="d",
            available_info=["a"],
            leakage_notes=[],
            success_criterion="s",
        )


def test_hypothesis_mode_with_block_parses() -> None:
    contract = AnalysisContract(
        name="test",
        mode="hypothesis",
        goal="g",
        row_definition="r",
        decision_moment="d",
        available_info=["a"],
        leakage_notes=[],
        success_criterion="s",
        hypothesis={
            "group_column": "Borough",
            "group_values": ["Manhattan", "Brooklyn"],
        },
    )
    assert contract.mode == AnalysisMode.HYPOTHESIS
    assert contract.hypothesis.group_column == "Borough"
    assert contract.hypothesis.group_values == ["Manhattan", "Brooklyn"]


def test_prediction_mode_allows_omitted_hypothesis() -> None:
    contract = AnalysisContract(
        name="test",
        mode="prediction",
        goal="g",
        row_definition="r",
        decision_moment="d",
        available_info=["a"],
        leakage_notes=[],
        success_criterion="s",
    )
    assert contract.hypothesis is None
