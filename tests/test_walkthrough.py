from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from contract_fixture import categorical_column, target_column

from broadway.config.loader import load_config
from broadway.reports import paths
from broadway.timeline import decide, module, runners, walkthrough
from broadway.timeline.evidence import ConclusionEvidence, PosthocEvidence
from broadway.timeline.models import AnalysisDecision, AnalysisStep, StepStatus
from broadway.timeline.sequence import load_walkthrough_config


def _setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(module, "TIMELINE_DIR", tmp_path / "timeline")
    monkeypatch.setattr(walkthrough, "TIMELINE_DIR", tmp_path / "timeline")
    monkeypatch.setattr(paths, "TIMELINE_PATH", tmp_path / "reports" / "timeline.md")
    monkeypatch.setattr(paths, "INDEX_PATH", tmp_path / "reports" / "index.md")
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path / "reports" / "results")
    monkeypatch.setattr(paths, "FIGURES_DIR", tmp_path / "reports" / "figures")


def _demo_columns() -> tuple[str, str]:
    """(group column, target column) from the demo dataset contract."""
    cfg = load_config("contracts", dataset="test", experiment="baseline")
    assert cfg.dataset is not None
    group = categorical_column(cfg.dataset)
    assert group is not None
    return group, target_column(cfg.dataset)


def _make_canonical(tmp_path: Path, group: str, target: str) -> Path:
    rng = np.random.default_rng(0)
    specs = [
        ("A", 10.0, 1.0),
        ("B", 15.0, 1.5),
    ]
    frames = [
        pd.DataFrame(
            {group: [name] * 60, target: rng.normal(mean, std, 60)}
        )
        for name, mean, std in specs
    ]
    path = tmp_path / "test_canonical.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
    return path


def _counter_clock():
    state = {"n": 0}

    def clock() -> str:
        state["n"] += 1
        return f"2026-01-01T00:00:{state['n']:02d}.000000+00:00"

    return clock


def _load_cfg():
    return load_config("stats", dataset="test", analysis="test_hypothesis")


def test_walkthrough_stops_at_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _setup(monkeypatch, tmp_path)
    cfg = _load_cfg()
    monkeypatch.setattr(runners, "canonical_path", lambda d, e: _make_canonical(tmp_path, *_demo_columns()))

    walkthrough.run(cfg, None, force=False)

    steps_dir = tmp_path / "timeline" / "test_hypothesis" / "steps"
    evidence_dir = tmp_path / "timeline" / "test_hypothesis" / "evidence"
    for step_id in ("describe_groups", "normality", "variance"):
        assert module.load_step("test_hypothesis", step_id) is not None
        assert (steps_dir / f"{step_id}.json").exists()
    for evidence in ("describe.json", "normality.json", "variance.json"):
        assert (evidence_dir / evidence).exists()
    assert (tmp_path / "reports" / "figures" / "normality_qq.png").exists()
    assert (tmp_path / "reports" / "timeline.md").exists()
    assert module.load_step("test_hypothesis", "omnibus") is None
    assert not (steps_dir / "omnibus.json").exists()

    out = capsys.readouterr().out
    assert "DECISION REQUIRED" in out
    assert "decide_omnibus" in out
    assert "--method <method>" in out
    assert "--method welch" not in out


def test_decision_gate_humanizes_and_deprescribes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _setup(monkeypatch, tmp_path)
    analysis = _load_cfg().analysis
    steps = [
        AnalysisStep(
            analysis=analysis.name,
            step_id="describe_groups",
            order=1,
            question="q?",
            status=StepStatus.COMPLETED,
            method="describe",
            source="canonical",
            sample_name=None,
            evidence_refs=[],
            result_summary={
                "n_total": 4332549,
                "imbalance_ratio": 2162.6747,
                "absent_groups": 0,
            },
            ramification="r",
            decision_required=False,
            performed_at="2026-01-01T00:00:00Z",
        ),
        AnalysisStep(
            analysis=analysis.name,
            step_id="variance",
            order=3,
            question="q?",
            status=StepStatus.WARNING,
            method="levene",
            source="canonical",
            sample_name=None,
            evidence_refs=[],
            result_summary={"statistic": 4199.7167447099055, "p_value": 0.0},
            ramification="r",
            decision_required=False,
            performed_at="2026-01-01T00:00:00Z",
        ),
    ]

    walkthrough._print_decision_required(analysis, steps, load_walkthrough_config())
    out = capsys.readouterr().out

    assert "DECISION REQUIRED" in out
    assert "--method <method>" in out
    assert "--method welch" not in out
    assert "< 0.001" in out
    assert "4199.71" not in out
    assert "4.2e+03" in out
    assert "2.16e+03" in out


def test_walkthrough_resume_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch, tmp_path)
    cfg = _load_cfg()
    monkeypatch.setattr(runners, "canonical_path", lambda d, e: _make_canonical(tmp_path, *_demo_columns()))

    walkthrough.run(cfg, None, force=False)
    assert len(module.load_steps("test_hypothesis")) == 3

    walkthrough.run(cfg, None, force=False)
    assert len(module.load_steps("test_hypothesis")) == 3


def test_walkthrough_resume_past_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _setup(monkeypatch, tmp_path)
    cfg = _load_cfg()
    monkeypatch.setattr(runners, "canonical_path", lambda d, e: _make_canonical(tmp_path, *_demo_columns()))

    walkthrough.run(cfg, None, force=False)
    assert "DECISION REQUIRED" in capsys.readouterr().out

    decision = decide.record(
        cfg.analysis, "omnibus", "welch", "non-normal",
    )
    module.save_decision(decision)

    walkthrough.run(cfg, None, force=False)
    out = capsys.readouterr().out
    assert "DECISION REQUIRED" in out
    assert "decide_posthoc" in out
    assert "games_howell" in out
    assert module.load_step("test_hypothesis", "omnibus") is not None
    assert (tmp_path / "reports" / "timeline.md").exists()


def test_walkthrough_force_reruns_but_preserves_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch, tmp_path)
    cfg = _load_cfg()
    monkeypatch.setattr(runners, "canonical_path", lambda d, e: _make_canonical(tmp_path, *_demo_columns()))
    monkeypatch.setattr(runners, "now_iso", _counter_clock())

    walkthrough.run(cfg, None, force=False)
    before = {s.step_id: s.performed_at for s in module.load_steps("test_hypothesis")}

    decision = AnalysisDecision(
        analysis="test_hypothesis",
        id="omnibus",
        kind="omnibus",
        question="Which principal method should answer the question?",
        method="welch",
        reason=["non-normal"],
        status="resolved",
        parents=["normality"],
        decided_at="2026-01-01T00:00:00Z",
    )
    module.save_decision(decision)

    walkthrough.run(cfg, None, force=True)

    after = {s.step_id: s.performed_at for s in module.load_steps("test_hypothesis")}
    for step_id in ("describe_groups", "normality", "variance"):
        assert after[step_id] != before[step_id]
    assert module.load_decision("test_hypothesis", "omnibus") is not None


def test_run_variance_warning_on_unequal_variances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch, tmp_path)
    cfg = _load_cfg()
    groups = {
        "a": np.array([1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.0, 1.1]),
        "b": np.array([1.0, 50.0, 1.0, 40.0, 1.0, 60.0, 1.0, 45.0]),
    }
    step = runners.run_variance(
        cfg.analysis, 3, "q?", groups, tmp_path / "timeline" / "test_hypothesis",
        "canonical", None,
    )
    assert step.status == StepStatus.WARNING
    assert step.method == "levene"


def test_run_normality_writes_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch, tmp_path)
    cfg = _load_cfg()
    rng = np.random.default_rng(0)
    groups = {g: rng.normal(10.0, 2.0, 30) for g in ("A", "B")}
    figures_dir = tmp_path / "reports" / "figures"
    step = runners.run_normality(
        cfg.analysis, 2, "q?", groups, tmp_path / "timeline" / "test_hypothesis",
        figures_dir, "canonical", None,
    )
    assert (figures_dir / "normality_qq.png").exists()
    assert "figures/normality_qq.png" in step.evidence_refs


def test_load_frame_and_groups_from_canonical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch, tmp_path)
    cfg = _load_cfg()
    group, target = _demo_columns()
    canonical = _make_canonical(tmp_path, group, target)
    monkeypatch.setattr(runners, "canonical_path", lambda d, e: canonical)

    df, group_column, source_group_column, groups, attrition = runners.load_frame_and_groups(cfg, None)

    assert group_column == group
    assert source_group_column == group
    assert set(groups) == set(cfg.analysis.hypothesis.group_values)
    assert groups["A"].shape[0] == 60
    assert groups["B"].shape[0] == 60
    assert group in df.columns
    assert cfg.dataset.target in df.columns
    assert attrition["n_total"] == 120
    assert attrition["n_used"] == 120
    assert attrition["n_excluded"] == 0
    assert attrition["exclusion_reason"] == ""


def test_run_omnibus_welch_reports_effect_sizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch, tmp_path)
    cfg = _load_cfg()
    rng = np.random.default_rng(0)
    groups = {g: rng.normal(10.0 + i, 1.0, 30) for i, g in enumerate(("A", "B", "C"))}
    step = runners.run_omnibus(
        cfg.analysis, 5, "q?", groups, "welch",
        tmp_path / "timeline" / "test_hypothesis", "canonical", None,
    )
    assert step.method == "welch"
    assert "eta_squared" in step.result_summary
    assert "omega_squared" in step.result_summary
    assert (tmp_path / "timeline" / "test_hypothesis" / "evidence" / "omnibus.json").exists()


def test_run_omnibus_kruskal_epsilon_squared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch, tmp_path)
    cfg = _load_cfg()
    rng = np.random.default_rng(0)
    groups = {g: rng.normal(10.0 + i, 1.0, 30) for i, g in enumerate(("A", "B", "C"))}
    step = runners.run_omnibus(
        cfg.analysis, 5, "q?", groups, "kruskal",
        tmp_path / "timeline" / "test_hypothesis", "canonical", None,
    )
    assert "epsilon_squared" in step.result_summary
    assert "effect_size" not in step.result_summary
    assert "ε²" in step.ramification
    assert "deliberately not computed" not in step.ramification


def _omnibus_step(method: str) -> AnalysisStep:
    summary: dict = {
        "method": method,
        "statistic": 12.0,
        "p_value": 0.0004,
        "passed": True,
    }
    if method in ("anova", "welch"):
        summary["eta_squared"] = 0.982
        summary["omega_squared"] = 0.123
    else:
        summary["epsilon_squared"] = 0.456
    return AnalysisStep(
        analysis="test_hypothesis",
        step_id="omnibus",
        order=5,
        question="q?",
        status=StepStatus.COMPLETED,
        method=method,
        source="canonical",
        sample_name=None,
        evidence_refs=[],
        result_summary=summary,
        ramification="r",
        decision_required=True,
        performed_at="2026-01-01T00:00:00Z",
    )


def test_run_conclusion_copies_effect_sizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch, tmp_path)
    cfg = _load_cfg()
    step = runners.run_conclusion(
        cfg.analysis, 8, "q?", _omnibus_step("welch"), None,
        tmp_path / "timeline" / "test_hypothesis", "canonical", None,
    )
    rs = step.result_summary
    assert rs["eta_squared"] == 0.982
    assert rs["omega_squared"] == 0.123
    assert "effect_size" not in rs


def test_run_conclusion_kruskal_epsilon_squared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch, tmp_path)
    cfg = _load_cfg()
    step = runners.run_conclusion(
        cfg.analysis, 8, "q?", _omnibus_step("kruskal"), None,
        tmp_path / "timeline" / "test_hypothesis", "canonical", None,
    )
    rs = step.result_summary
    assert rs["epsilon_squared"] == pytest.approx(0.456)
    assert "effect_size" not in rs
    evidence = ConclusionEvidence.model_validate_json(
        (tmp_path / "timeline" / "test_hypothesis" / "evidence" / "conclusion.json").read_text()
    )
    assert "ε²" in evidence.effect_size


def test_run_posthoc_significant_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch, tmp_path)
    cfg = _load_cfg()
    rng = np.random.default_rng(42)
    frames = [
        pd.DataFrame({"group": g, "dv": rng.normal(0.0 + i * 10.0, 1.0, 30)})
        for i, g in enumerate(("A", "B", "C"))
    ]
    df = pd.concat(frames, ignore_index=True)
    step = runners.run_posthoc(
        cfg.analysis, 7, "q?", df, "group", "dv", "games_howell",
        tmp_path / "timeline" / "test_hypothesis", "canonical", None,
    )
    assert step.method == "games_howell"
    assert step.result_summary["pairs"] == 3
    assert step.result_summary["significant_pairs"] == 3
    details = step.result_summary["significant_pair_details"]
    assert len(details) == 3
    assert set(details[0]) == {"a", "b", "p_value", "cohens_d", "hedges_g", "effect_size_note"}
    evidence = PosthocEvidence.model_validate_json(
        (tmp_path / "timeline" / "test_hypothesis" / "evidence" / "posthoc.json").read_text()
    )
    assert evidence.significant_pairs == 3
    assert len(evidence.pairs) == 3


def test_walkthrough_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _setup(monkeypatch, tmp_path)
    cfg = _load_cfg()
    monkeypatch.setattr(runners, "canonical_path", lambda d, e: _make_canonical(tmp_path, *_demo_columns()))

    walkthrough.run(cfg, None, force=False)
    assert "DECISION REQUIRED" in capsys.readouterr().out

    omnibus = decide.record(
        cfg.analysis, "omnibus", "welch", "non-normal",
    )
    module.save_decision(omnibus)

    walkthrough.run(cfg, None, force=False)
    out = capsys.readouterr().out
    assert "DECISION REQUIRED" in out
    assert "decide_posthoc" in out
    assert "games_howell" in out

    posthoc = decide.record(
        cfg.analysis, "posthoc", "games_howell", "significant omnibus",
    )
    module.save_decision(posthoc)

    walkthrough.run(cfg, None, force=False)
    out = capsys.readouterr().out

    by_id = {s.step_id: s for s in module.load_steps("test_hypothesis")}
    for step_id in ("describe_groups", "normality", "variance", "omnibus", "posthoc", "conclusion"):
        assert step_id in by_id
    for decision_id in ("omnibus", "posthoc"):
        assert module.load_decision("test_hypothesis", decision_id) is not None

    assert by_id["omnibus"].decision_required is True
    assert by_id["posthoc"].decision_required is False
    assert by_id["conclusion"].method == "conclusion"
    assert "no significant difference" not in by_id["conclusion"].ramification

    timeline = (tmp_path / "reports" / "timeline.md").read_text()
    assert "conclusion" in timeline


def test_walkthrough_posthoc_gated_on_insignificant_omnibus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _setup(monkeypatch, tmp_path)
    cfg = _load_cfg()
    group, target = _demo_columns()
    rng = np.random.default_rng(1)
    frames = [
        pd.DataFrame(
            {group: [name] * 60, target: rng.normal(10.0, 1.0, 60)}
        )
        for name in ("A", "B")
    ]
    path = tmp_path / "test_canonical.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
    monkeypatch.setattr(runners, "canonical_path", lambda d, e: path)

    walkthrough.run(cfg, None, force=False)

    omnibus = decide.record(
        cfg.analysis, "omnibus", "welch", "normal",
    )
    module.save_decision(omnibus)

    walkthrough.run(cfg, None, force=False)
    capsys.readouterr()

    by_id = {s.step_id: s for s in module.load_steps("test_hypothesis")}
    assert "omnibus" in by_id
    assert "posthoc" not in by_id
    assert "conclusion" in by_id
    assert "no significant difference" in by_id["conclusion"].ramification


def test_stale_decision_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _setup(monkeypatch, tmp_path)
    cfg = _load_cfg()
    monkeypatch.setattr(runners, "canonical_path", lambda d, e: _make_canonical(tmp_path, *_demo_columns()))
    monkeypatch.setattr(runners, "now_iso", _counter_clock())

    walkthrough.run(cfg, None, force=False)

    omnibus = AnalysisDecision(
        analysis="test_hypothesis",
        id="omnibus",
        kind="omnibus",
        question="Which principal method should answer the question?",
        method="welch",
        reason=["non-normal"],
        status="resolved",
        parents=["describe_groups", "normality", "variance"],
        decided_at="2026-01-01T00:00:00Z",
    )
    module.save_decision(omnibus)
    posthoc = AnalysisDecision(
        analysis="test_hypothesis",
        id="posthoc",
        kind="posthoc",
        question="Which post-hoc comparison is appropriate?",
        method="games_howell",
        reason=["significant omnibus"],
        status="resolved",
        parents=["omnibus"],
        decided_at="2026-01-01T00:00:00Z",
    )
    module.save_decision(posthoc)

    walkthrough.run(cfg, None, force=True)
    out = capsys.readouterr().out
    assert "WARNING: decision 'omnibus' was made against earlier evidence" in out
    assert "WARNING: decision 'posthoc' was made against earlier evidence" in out


def test_run_omnibus_imbalanced_significant_is_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch, tmp_path)
    cfg = _load_cfg()
    rng = np.random.default_rng(0)
    groups = {
        "a": rng.normal(10.0, 1.0, 10),
        "b": rng.normal(15.0, 1.0, 200),
        "c": rng.normal(11.0, 1.0, 10),
    }
    step = runners.run_omnibus(
        cfg.analysis, 5, "q?", groups, "anova",
        tmp_path / "timeline" / "test_hypothesis", "canonical", None,
    )
    assert step.status == StepStatus.NOTE
    assert step.result_summary["passed"] is True


def test_run_describe_attrition_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch, tmp_path)
    cfg = _load_cfg()
    group, target = _demo_columns()
    df = pd.DataFrame(
        {group: ["A", "B"], target: [10.0, 20.0]}
    )
    step = runners.run_describe(
        cfg.analysis, 1, "q?", df, group, group,
        ["A", "B"], target, "x.parquet",
        None, "canonical", tmp_path / "timeline" / "test_hypothesis",
        tmp_path / "reports" / "figures",
    )
    rs = step.result_summary
    assert "total_n" not in rs
    assert rs["n_total"] == 2
    assert rs["n_used"] == 2
    assert rs["n_excluded"] == 0
    assert rs["exclusion_reason"] == ""


def test_run_describe_attrition_null_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch, tmp_path)
    cfg = _load_cfg()
    group, target = _demo_columns()
    df = pd.DataFrame(
        {
            group: ["A", "B", None],
            target: [10.0, 20.0, 30.0],
        }
    )
    step = runners.run_describe(
        cfg.analysis, 1, "q?", df, group, group,
        ["A", "B"], target, "x.parquet",
        None, "canonical", tmp_path / "timeline" / "test_hypothesis",
        tmp_path / "reports" / "figures",
    )
    rs = step.result_summary
    assert rs["n_excluded"] == 1
    assert "null group" in rs["exclusion_reason"]


def test_run_describe_attrition_unlisted_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch, tmp_path)
    cfg = _load_cfg()
    group, target = _demo_columns()
    df = pd.DataFrame(
        {
            group: ["A", "B", "C"],
            target: [10.0, 20.0, 30.0],
        }
    )
    step = runners.run_describe(
        cfg.analysis, 1, "q?", df, group, group,
        ["A", "B"], target, "x.parquet",
        None, "canonical", tmp_path / "timeline" / "test_hypothesis",
        tmp_path / "reports" / "figures",
    )
    rs = step.result_summary
    assert rs["n_excluded"] == 1
    assert "unlisted group" in rs["exclusion_reason"]


def test_walkthrough_failure_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _setup(monkeypatch, tmp_path)
    cfg = _load_cfg()
    monkeypatch.setattr(runners, "canonical_path", lambda d, e: _make_canonical(tmp_path, *_demo_columns()))

    original = runners.run_describe

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(runners, "run_describe", boom)

    walkthrough.run(cfg, None, force=False)

    failed = module.load_step("test_hypothesis", "describe_groups")
    assert failed is not None
    assert failed.status == StepStatus.FAILED
    assert failed.result_summary == {}
    assert failed.evidence_refs == []
    log = tmp_path / "timeline" / "test_hypothesis" / "failures" / "describe_groups.log"
    assert log.exists()
    assert "RuntimeError" in log.read_text()
    out = capsys.readouterr().out
    assert "failed while attempting describing the groups" in out
    assert "fix and rerun" in out

    monkeypatch.setattr(runners, "run_describe", original)
    walkthrough.run(cfg, None, force=True)

    replaced = module.load_step("test_hypothesis", "describe_groups")
    assert replaced is not None
    assert replaced.status != StepStatus.FAILED


# --- guard rails: config requirements, load branches, unknown methods ------


def test_load_frame_and_groups_requires_dataset_and_hypothesis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch, tmp_path)
    cfg = _load_cfg()
    no_dataset = cfg.model_copy(update={"dataset": None})
    with pytest.raises(ValueError, match="walkthrough requires dataset and hypothesis config"):
        runners.load_frame_and_groups(no_dataset, None)

    bare_analysis = cfg.analysis.model_copy(update={"hypothesis": None})
    no_hypothesis = cfg.model_copy(
        update={"analysis": cfg.analysis.model_copy(update={"hypothesis": None})}
    )
    assert bare_analysis.hypothesis is None
    with pytest.raises(ValueError, match="walkthrough requires dataset and hypothesis config"):
        runners.load_frame_and_groups(no_hypothesis, None)


def test_load_frame_and_groups_missing_canonical_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch, tmp_path)
    cfg = _load_cfg()
    missing = tmp_path / "no_such_canonical.parquet"
    monkeypatch.setattr(runners, "canonical_path", lambda d, e: missing)
    with pytest.raises(FileNotFoundError, match="canonical dataset not found"):
        runners.load_frame_and_groups(cfg, None)


def test_load_frame_and_groups_sample_branch_maps_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from broadway.lineage.models import SampleSpec

    _setup(monkeypatch, tmp_path)
    cfg = _load_cfg()
    group, target = _demo_columns()
    source_name = f"source_{group}"
    sample_path = tmp_path / "mapped.parquet"
    pd.DataFrame({source_name: ["A"] * 5 + ["B"] * 5, target: range(10)}).to_parquet(
        sample_path, index=False
    )
    sample = SampleSpec(
        name="test_diagnostic", role="diagnostic", path=str(sample_path),
        column_mapping={group: source_name},
    )

    _df, group_column, source_group_column, groups, attrition = (
        runners.load_frame_and_groups(cfg, sample)
    )
    assert group_column == group
    assert source_group_column == source_name
    assert set(groups) == {"A", "B"}
    assert attrition["n_total"] == 10


def test_load_frame_and_groups_missing_sample_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from broadway.lineage.models import SampleSpec

    _setup(monkeypatch, tmp_path)
    cfg = _load_cfg()
    sample = SampleSpec(
        name="test_diagnostic", role="diagnostic",
        path=str(tmp_path / "gone.parquet"),
    )
    with pytest.raises(FileNotFoundError, match="sample dataset not found"):
        runners.load_frame_and_groups(cfg, sample)


def test_load_frame_and_groups_missing_group_column_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch, tmp_path)
    cfg = _load_cfg()
    _, target = _demo_columns()
    canonical = tmp_path / "target_only.parquet"
    pd.DataFrame({target: [1.0, 2.0, 3.0]}).to_parquet(canonical, index=False)
    monkeypatch.setattr(runners, "canonical_path", lambda d, e: canonical)
    with pytest.raises(ValueError, match="not found in data"):
        runners.load_frame_and_groups(cfg, None)


def test_run_describe_attrition_null_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch, tmp_path)
    cfg = _load_cfg()
    group, target = _demo_columns()
    df = pd.DataFrame(
        {group: ["A", "B"], target: [10.0, None]}
    )
    step = runners.run_describe(
        cfg.analysis, 1, "q?", df, group, group,
        ["A", "B"], target, "x.parquet",
        None, "canonical", tmp_path / "timeline" / "test_hypothesis",
        tmp_path / "reports" / "figures",
    )
    rs = step.result_summary
    assert rs["n_excluded"] == 1
    assert "null target" in rs["exclusion_reason"]


def test_run_describe_flags_imbalance_and_absent_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both flagged branches of the describe ramification: an absent declared
    group AND an imbalance ratio above the configured threshold must produce a
    NOTE status whose ramification names both facts."""
    _setup(monkeypatch, tmp_path)
    cfg = _load_cfg()
    group, target = _demo_columns()
    df = pd.DataFrame(
        {group: ["A", "A", "B", "B", "B", "B", "B"], target: list(range(7))}
    )
    step = runners.run_describe(
        cfg.analysis, 1, "q?", df, group, group,
        ["A", "B", "C"], target, "x.parquet",
        None, "canonical", tmp_path / "timeline" / "test_hypothesis",
        tmp_path / "reports" / "figures",
    )
    assert step.status == StepStatus.NOTE
    assert "groups C have no observations" in step.ramification
    assert "group sizes are imbalanced" in step.ramification
    assert step.result_summary["absent_groups"] == 1


def test_run_omnibus_unknown_method_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch, tmp_path)
    cfg = _load_cfg()
    with pytest.raises(ValueError, match="unknown omnibus method 'mode'"):
        runners.run_omnibus(
            cfg.analysis, 5, "q?", {"A": np.array([1.0])}, "mode",
            tmp_path / "timeline" / "test_hypothesis", "canonical", None,
        )


def test_run_posthoc_unknown_method_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch, tmp_path)
    cfg = _load_cfg()
    df = pd.DataFrame({"group": ["A", "B"], "dv": [1.0, 2.0]})
    with pytest.raises(ValueError, match="unknown posthoc method 'anova'"):
        runners.run_posthoc(
            cfg.analysis, 7, "q?", df, "group", "dv", "anova",
            tmp_path / "timeline" / "test_hypothesis", "canonical", None,
        )


def test_walkthrough_run_requires_hypothesis_block_and_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch, tmp_path)
    cfg = _load_cfg()

    no_hypothesis = cfg.model_copy(
        update={"analysis": cfg.analysis.model_copy(update={"hypothesis": None})}
    )
    with pytest.raises(ValueError, match="requires a 'hypothesis' block"):
        walkthrough.run(no_hypothesis, None, force=False)

    no_dataset = cfg.model_copy(update={"dataset": None})
    with pytest.raises(ValueError, match="walkthrough requires a dataset config"):
        walkthrough.run(no_dataset, None, force=False)


def test_omnibus_executor_stops_when_decision_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The executor-level stop-guard is the last line before dispatching with
    an unresolved method (the sequence gate normally trips first): call it
    directly with a decision-free timeline and pin the loud stop."""
    _setup(monkeypatch, tmp_path)
    cfg = _load_cfg()
    ctx = walkthrough.StepContext(
        analysis=cfg.analysis, df=pd.DataFrame(), groups={},
        group_column="g", source_group_column="g", group_values=["A", "B"],
        target="t", source_path="x.parquet", sample_name=None,
        source="canonical", out_dir=tmp_path / "timeline" / "test_hypothesis",
        figures_dir=tmp_path / "figures", attrition={},
        sequence=walkthrough.load_walkthrough_sequence(),
        thresholds=load_walkthrough_config(),
    )
    step = SimpleNamespace(id="omnibus", label="omnibus", order=5,
                           question="q?", kind="analysis")
    with pytest.raises(walkthrough._StopWalkthrough):
        walkthrough._run_omnibus(cfg.analysis, step, ctx)
    assert "no omnibus decision recorded" in capsys.readouterr().out
