"""Output contract (EXPERIMENT_OUTPUT.md): enforced series publish a question + figure."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


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
    enforced_steps = sum(
        len(series.steps) for series in audit.series() if series.series in audit._enforced
    )
    assert enforced_steps >= 12


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
    audit = _audit_module().PipelineDecisionAudit(tmp_path)

    assert audit.status() == {"eda": False}
