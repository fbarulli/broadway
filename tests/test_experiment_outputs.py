"""Output contract (EXPERIMENT_OUTPUT.md): enforced series publish a question + figure."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from broadway.reports.decision import Decision

_ROOT = Path(__file__).resolve().parents[1]


def _decision() -> dict[str, object]:
    return {
        "question": "Does the metric clear its threshold?",
        "metric": "association_target",
        "value": 0.94,
        "interval_low": 0.93,
        "interval_high": 0.95,
        "threshold": 0.05,
        "null_threshold": 0.0089,
        "direction": "max",
        "verdict": "act",
        "action": "keep the feature",
        "owner": "eda",
        "sample_size": 50000,
        "unit": None,
        "min_sample_size": 100,
        "evidence": {
            "baseline": "no-effect shuffle of the target at this n",
            "segments": [{"segment": "weekday", "value": 0.94, "sample_size": 30000}],
            "sensitivity": {"half_sample": 0.93},
            "falsifier": "the interval falls at or below the threshold",
            "provenance": {"dataset": "synthetic", "seed": "42"},
        },
    }


def _audit_module():
    """Load the gate script by path (scripts/ is not an importable package)."""
    path = _ROOT / "scripts" / "check_experiment_outputs.py"
    spec = importlib.util.spec_from_file_location("check_experiment_outputs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_experiment_outputs"] = module
    spec.loader.exec_module(module)
    return module


def test_enforced_series_steps_publish_a_question_and_a_figure() -> None:
    audit = _audit_module().ExperimentOutputAudit()

    assert audit.violations() == []
    # The Olist substrate publishes through the pipeline steps, not a script
    # series; the ratchet is empty until the modelling series lands.
    assert audit._enforced == ()
    enforced_steps = sum(
        len(series.steps) for series in audit.series() if series.series in audit._enforced
    )
    assert enforced_steps == 0


def test_audit_of_a_missing_experiments_root_is_empty(tmp_path: Path) -> None:
    audit = _audit_module().ExperimentOutputAudit(tmp_path / "missing")

    assert audit.series() == []


def test_project_checkout_without_results_root_still_enforces_eda(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    module = _audit_module()
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    layout = module.ProjectLayout(experiments=experiments, results=tmp_path / "results")

    assert "eda" in module._enforced_pipeline_steps(layout)

    monkeypatch.setattr(module, "resolve_project_layout", lambda: layout)
    assert module.main() == 1
    assert "FAIL pipeline/eda: missing published decision record" in capsys.readouterr().out

    skipped = module.ProjectLayout(None, None, "project unavailable")
    monkeypatch.setattr(module, "resolve_project_layout", lambda: skipped)
    assert module.main() == 0
    assert "no record is enforced (advisory)" in capsys.readouterr().out


def test_audit_flags_a_step_without_a_figure(tmp_path: Path) -> None:
    series = tmp_path / "demo"
    series.mkdir()
    (series / "01_question_only.py").write_text('"""A question."""\n', encoding="utf-8")
    (series / "02_compliant.py").write_text(
        '"""A question."""\nfigure.savefig("out.png")\n', encoding="utf-8"
    )

    audit = _audit_module().ExperimentOutputAudit(tmp_path, enforced=("demo",))

    assert [step.stem for step in audit.violations()] == ["01_question_only"]


def test_pipeline_audit_reports_missing_decisions(tmp_path: Path) -> None:
    module = tmp_path / "eda"
    module.mkdir()
    (module / "module.py").write_text("def run(cfg):\n    return cfg\n", encoding="utf-8")
    audit = _audit_module().PipelineDecisionAudit(
        tmp_path, results_root=tmp_path / "results"
    )

    assert audit.status() == {"eda": False}


def test_pipeline_enforced_set_is_pinned() -> None:
    """The enforced set is explicit: a step leaves it only with its dataset."""
    module = _audit_module()

    assert module.PIPELINE_DECISION_ENFORCED == ("etl", "eda", "features", "baseline")
    assert len(module.PIPELINE_DECISION_ENFORCED) == 4
    assert module.PIPELINE_NOT_APPLICABLE == {
        "causal": "the Olist project has no causal analysis config, so no record can exist"
    }


def test_a_code_call_alone_does_not_satisfy_the_record_gate(tmp_path: Path) -> None:
    """The loophole: a StepOutput call without a published record must fail."""
    module = tmp_path / "eda"
    module.mkdir()
    (module / "module.py").write_text(
        "def run(cfg):\n"
        "    StepOutput(results_dir='x', stem='eda', decision=None)\n"
        "    return cfg\n",
        encoding="utf-8",
    )
    audit = _audit_module().PipelineDecisionAudit(
        tmp_path, results_root=tmp_path / "results"
    )

    assert audit.code_publishes_decision("eda") is True
    assert audit.status() == {"eda": False}


def test_deleting_a_pipeline_record_turns_the_gate_red(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _audit_module()
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    results = tmp_path / "results"
    payload = json.dumps(_decision())
    for step in module.PIPELINE_DECISION_ENFORCED:
        record = results / "pipeline" / step / f"{step}_decision.json"
        record.parent.mkdir(parents=True)
        record.write_text(payload, encoding="utf-8")
    layout = module.ProjectLayout(experiments=experiments, results=results)
    monkeypatch.setattr(module, "resolve_project_layout", lambda: layout)

    assert module.main() == 0
    assert "FAIL" not in capsys.readouterr().out

    (results / "pipeline" / "eda" / "eda_decision.json").unlink()
    assert module.main() == 1
    assert "FAIL pipeline/eda: missing published decision record" in capsys.readouterr().out


def test_a_missing_non_eda_record_turns_the_gate_red(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The step module is resolved via STEP_MODULES, so etl's module.py is checked."""
    module = _audit_module()
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    results = tmp_path / "results"
    payload = json.dumps(_decision())
    for step in module.PIPELINE_DECISION_ENFORCED:
        if step == "etl":
            continue
        record = results / "pipeline" / step / f"{step}_decision.json"
        record.parent.mkdir(parents=True)
        record.write_text(payload, encoding="utf-8")
    layout = module.ProjectLayout(experiments=experiments, results=results)
    monkeypatch.setattr(module, "resolve_project_layout", lambda: layout)

    assert module.main() == 1
    assert "FAIL pipeline/etl: missing published decision record" in capsys.readouterr().out



def test_pipeline_audit_accepts_only_a_canonical_published_record(tmp_path: Path) -> None:
    module = tmp_path / "eda"
    module.mkdir()
    (module / "module.py").write_text("def run(cfg):\n    return cfg\n", encoding="utf-8")
    results = tmp_path / "results"
    record = results / "eda" / "eda_decision.json"
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps(Decision.model_validate(_decision()).model_dump(mode="json")),
        encoding="utf-8",
    )
    audit = _audit_module().PipelineDecisionAudit(tmp_path, results_root=results)

    assert audit.status() == {"eda": True}

    record.write_text("[]", encoding="utf-8")
    assert audit.status() == {"eda": False}


def test_decision_enforced_pair_needs_a_valid_record(tmp_path: Path) -> None:
    """A pair passes only when its published record validates (not just the AST call)."""
    audit = _audit_module()
    experiments = tmp_path / "experiments"
    series = experiments / "synthetic"
    series.mkdir(parents=True)
    (series / "01_step.py").write_text(
        '"""Does the metric clear its threshold?"""\n'
        "def main():\n"
        "    figure.savefig('01_step.png')\n"
        "    StepOutput(results_dir='.', stem='01_step', decision=None)\n",
        encoding="utf-8",
    )
    results = experiments / "results" / "synthetic"
    results.mkdir(parents=True)
    gate = audit.ExperimentOutputAudit(
        experiments,
        results_root=experiments / "results",
        decision_enforced=(("synthetic", "01_step"),),
    )
    assert [(item.series, item.stem) for item in gate.violations()] == [
        ("synthetic", "01_step")
    ]
    (results / "01_step_decision.json").write_text(
        json.dumps(_decision()), encoding="utf-8"
    )
    assert gate.violations() == []
