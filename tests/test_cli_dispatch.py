"""In-process dispatch tests for broadway.cli.

tests/test_cli.py exercises the CLI end-to-end via subprocess (exit codes,
stderr vocabulary); those runs cannot attribute coverage to cli.py, so the
parser+dispatch wiring itself had zero in-process verification. These tests
pin the other half of the contract: each subcommand parses its arguments and
dispatches them to exactly the right pipeline entry point with the right
values. Delegates are replaced with recording spies; nothing here executes a
real step.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from broadway import cli


def _main(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["ds-pipeline", *argv])
    cli.main()


class Spy:
    """Record call args; assert dispatch reached the right delegate."""

    def __init__(self, ret: Any = None) -> None:
        self.calls: list[tuple] = []
        self.kwargs_calls: list[dict] = []
        self.ret = ret

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(args)
        self.kwargs_calls.append(kwargs)
        return self.ret


def _patch(
    monkeypatch: pytest.MonkeyPatch, target: object, name: str, spy: Spy
) -> None:
    monkeypatch.setattr(target, name, spy)


def test_cli_discover_dispatches_csv_target_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from broadway.discover import module as discover_module

    spy = Spy()
    _patch(monkeypatch, discover_module, "run", spy)
    _main(
        monkeypatch, "discover", "--csv", "a.csv", "--target", "price",
        "--task", "regression",
    )
    assert spy.calls == [("a.csv", "price", "regression", None, [])]


def test_cli_discover_passes_optional_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from broadway.discover import module as discover_module

    spy = Spy()
    _patch(monkeypatch, discover_module, "run", spy)
    _main(
        monkeypatch, "discover", "--csv", "a.csv", "--target", "price",
        "--task", "regression", "--datetime-column", "dt",
        "--ignore-columns", "c1", "c2",
    )
    assert spy.calls == [("a.csv", "price", "regression", "dt", ["c1", "c2"])]


def test_cli_columns_dispatches_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    from broadway.discover import columns as columns_module

    spy = Spy()
    _patch(monkeypatch, columns_module, "run", spy)
    _main(monkeypatch, "columns", "--csv", "x.csv")
    assert spy.calls == [("x.csv",)]


def test_cli_lineage_dispatches_analysis_and_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from broadway.lineage import module as lineage_module

    spy = Spy()
    _patch(monkeypatch, lineage_module, "run", spy)
    _main(monkeypatch, "lineage", "--analysis", "test_hypothesis", "--dataset", "test")
    assert spy.calls == [("test_hypothesis", "test")]


def test_cli_report_without_history_prints_hint(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from broadway.timeline import module as timeline_module

    _patch(monkeypatch, timeline_module, "load_steps", Spy(ret=[]))
    _patch(monkeypatch, timeline_module, "load_decisions", Spy(ret=[]))
    write_results = Spy()
    from broadway.reports import results as results_module

    _patch(monkeypatch, results_module, "write_results", write_results)
    _main(monkeypatch, "report", "--analysis", "test_hypothesis", "--dataset", "test")
    assert "run the walkthrough first" in capsys.readouterr().out
    assert write_results.calls == []


def test_cli_report_writes_results(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from broadway.timeline import module as timeline_module

    steps = [SimpleNamespace(step_id="describe_groups")]
    decisions = [SimpleNamespace(kind="omnibus")]
    _patch(monkeypatch, timeline_module, "load_steps", Spy(ret=steps))
    _patch(monkeypatch, timeline_module, "load_decisions", Spy(ret=decisions))
    sequence_spy = Spy(ret="SEQ")
    from broadway.timeline import sequence as sequence_module

    _patch(monkeypatch, sequence_module, "load_walkthrough_sequence", sequence_spy)
    write_results = Spy()
    from broadway.reports import results as results_module

    _patch(monkeypatch, results_module, "write_results", write_results)
    _main(monkeypatch, "report", "--analysis", "test_hypothesis", "--dataset", "test")
    assert sequence_spy.calls == [()]
    assert write_results.calls == [
        ("test_hypothesis", "SEQ", steps, decisions),
    ]
    assert "wrote reports/results/" in capsys.readouterr().out


def test_cli_profile_dispatches_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    from broadway.discover import module as discover_module

    spy = Spy()
    _patch(monkeypatch, discover_module, "profile", spy)
    _main(monkeypatch, "profile", "--dataset", "test")
    assert spy.calls == [("test",)]


def test_cli_audit_dispatches_dataset_analysis_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from broadway.reports import audit as audit_module

    spy = Spy()
    _patch(monkeypatch, audit_module, "run", spy)
    _main(monkeypatch, "audit", "--dataset", "test", "--analysis", "test_hypothesis")
    assert spy.calls == [("test", "test_hypothesis", "development")]


def test_cli_init_forwards_all_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    from broadway.onboard import module as onboard_module

    spy = Spy()
    _patch(monkeypatch, onboard_module, "init", spy)
    _main(
        monkeypatch, "init", "data.csv", "--name", "mydata", "--target", "price",
        "--task", "regression", "--datetime-columns", "dt1", "dt2",
        "--ignore-columns", "junk", "--split-column", "region", "--mode", "full",
        "--goal", "g", "--row-definition", "r", "--decision-moment", "m",
        "--available-info", "a1", "--leakage-notes", "n1",
        "--success-criterion", "s",
    )
    assert spy.calls == [(
        "data.csv", "mydata", "price", "regression", ["dt1", "dt2"], ["junk"],
        "region", "full", "g", "r", "m", ["a1"], ["n1"], "s",
    )]


def _fake_cfg(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    cfg = SimpleNamespace(full=False)
    _patch(monkeypatch, cli, "load_config", Spy(ret=cfg))
    return cfg


def test_cli_ingest_dispatches_generic_etl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from broadway.etl import module as etl_module

    cfg = _fake_cfg(monkeypatch)
    run_spy = Spy()
    _patch(monkeypatch, etl_module, "run", run_spy)
    _main(monkeypatch, "ingest", "--dataset", "test", "--experiment", "baseline")

    assert run_spy.calls == [(cfg,)]
    assert cli.load_config.kwargs_calls == [{
        "step": "etl",
        "dataset": "test",
        "experiment": "baseline",
        "analysis": None,
        "environment": "development",
    }]


def test_cli_stats_run_dispatches_with_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from broadway.stats import module as stats_module

    sample = SimpleNamespace(name="s1")
    _patch(monkeypatch, cli, "load_sample", Spy(ret=sample))
    run_spy = Spy()
    _patch(monkeypatch, stats_module, "run", run_spy)
    _fake_cfg(monkeypatch)
    _main(
        monkeypatch, "stats", "run", "--dataset", "test",
        "--analysis", "test_hypothesis", "--sample", "s1",
    )
    cfg_arg = run_spy.calls[0][0]
    assert run_spy.calls[0][1] is sample
    assert cfg_arg.full is False  # the config loaded by cli.load_config


def test_cli_stats_describe_dispatches_with_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from broadway.stats import describe as describe_module

    sample = SimpleNamespace(name="s2")
    _patch(monkeypatch, cli, "load_sample", Spy(ret=sample))
    run_spy = Spy()
    _patch(monkeypatch, describe_module, "run", run_spy)
    _fake_cfg(monkeypatch)
    _main(
        monkeypatch, "stats", "describe", "--dataset", "test",
        "--analysis", "test_hypothesis", "--sample", "s2",
    )
    assert len(run_spy.calls) == 1
    assert run_spy.calls[0][1] is sample


def test_cli_walkthrough_dispatches_force_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from broadway.timeline import walkthrough as walkthrough_module

    sample = SimpleNamespace(name="s3")
    _patch(monkeypatch, cli, "load_sample", Spy(ret=sample))
    run_spy = Spy()
    _patch(monkeypatch, walkthrough_module, "run", run_spy)
    _fake_cfg(monkeypatch)
    _main(
        monkeypatch, "walkthrough", "--analysis", "test_hypothesis",
        "--dataset", "test", "--sample", "s3", "--force",
    )
    _cfg_arg, sample_arg, force = run_spy.calls[0]
    assert sample_arg is sample
    assert force is True


def test_cli_walkthrough_without_sample_passes_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from broadway.timeline import walkthrough as walkthrough_module

    run_spy = Spy()
    _patch(monkeypatch, walkthrough_module, "run", run_spy)
    _fake_cfg(monkeypatch)
    _main(
        monkeypatch, "walkthrough", "--analysis", "test_hypothesis",
        "--dataset", "test",
    )
    assert run_spy.calls[0][1] is None
    assert run_spy.calls[0][2] is False


def test_cli_decide_records_and_saves_decision(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from broadway.config.loader import load_config
    from broadway.timeline import decide as decide_module
    from broadway.timeline import module as timeline_module

    cfg = load_config("stats", dataset="test", analysis="test_hypothesis")
    _patch(monkeypatch, cli, "load_config", Spy(ret=cfg))
    decision = SimpleNamespace(kind="omnibus", method="welch")
    record_spy = Spy(ret=decision)
    _patch(monkeypatch, decide_module, "record", record_spy)
    save_spy = Spy()
    _patch(monkeypatch, timeline_module, "save_decision", save_spy)

    _main(
        monkeypatch, "decide", "--analysis", "test_hypothesis",
        "--method", "welch", "--reason", "non-normal residuals",
    )

    analysis_arg, kind_arg, method_arg, reason_arg = record_spy.calls[0]
    assert analysis_arg.mode.value == "hypothesis"  # require_mode ran on it
    assert kind_arg == "omnibus"
    assert method_arg == "welch"
    assert reason_arg == "non-normal residuals"
    assert save_spy.calls == [(decision,)]
    out = capsys.readouterr().out
    assert "recorded decision 'omnibus'" in out
    assert "next: ds-pipeline walkthrough" in out


def test_cli_decide_posthoc_kind_is_parsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from broadway.config.loader import load_config
    from broadway.timeline import decide as decide_module

    cfg = load_config("stats", dataset="test", analysis="test_hypothesis")
    _patch(monkeypatch, cli, "load_config", Spy(ret=cfg))
    record_spy = Spy(ret=SimpleNamespace(kind="posthoc", method="games_howell"))
    _patch(monkeypatch, decide_module, "record", record_spy)
    from broadway.timeline import module as timeline_module

    _patch(monkeypatch, timeline_module, "save_decision", Spy())

    _main(
        monkeypatch, "decide", "--analysis", "test_hypothesis",
        "--kind", "posthoc", "--method", "games_howell", "--reason", "omnibus passed",
    )
    assert record_spy.calls[0][1] == "posthoc"


def test_cli_generic_step_single_dispatches_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from broadway import pipeline as pipeline_module

    run_spy = Spy()
    _patch(monkeypatch, pipeline_module, "run", run_spy)
    cfg = _fake_cfg(monkeypatch)
    _main(
        monkeypatch, "etl", "--dataset", "test", "--experiment", "baseline",
    )
    cfg_arg, steps = run_spy.calls[0]
    assert cfg_arg is cfg
    assert steps == ["etl"]  # cfg.full False -> just the named step


def test_cli_full_pipeline_resolves_step_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from broadway import pipeline as pipeline_module

    run_spy = Spy()
    _patch(monkeypatch, pipeline_module, "run", run_spy)
    cfg = SimpleNamespace(full=True)
    _patch(monkeypatch, cli, "load_config", Spy(ret=cfg))
    _patch(monkeypatch, cli, "resolve_full_steps", Spy(ret=["etl", "train"]))
    _main(
        monkeypatch, "etl", "--dataset", "test", "--experiment", "baseline",
    )
    _, steps = run_spy.calls[0]
    assert steps == ["etl", "train"]
