"""Public behavior of the tracking bundle and ``ds-pipeline tracking``.

The bundle is the single owner of the ignored machine truth
(``baseline_stats.json``) and of the tracked published surface
(``data_card.md``); these tests pin its failure/skip paths and the pure-render
command end-to-end on a tmp tree.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from broadway.config.loader import load_config
from broadway.config.schema import TrackingConfig
from broadway.discover.profile import build_profile
from broadway.tracking import report as tracking_report
from broadway.tracking.bundle import TrackingBundle


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "cat": ["a", "b", "a", "a", "b", "c", "a", "b"],
        }
    )


def _config() -> TrackingConfig:
    cfg = load_config("train", dataset="test", environment="development")
    assert cfg.train is not None and cfg.train.tracking is not None
    return cfg.train.tracking


def _bundle(
    tmp_path: Path,
    *,
    config: TrackingConfig | None = None,
    clean_result_path: Path | None = None,
) -> TrackingBundle:
    return TrackingBundle.from_config(
        dataset="test",
        config=config or _config(),
        clean_result_path=clean_result_path or tmp_path / "absent_clean.json",
        machine_dir=tmp_path / "tracking",
        reports_dir=tmp_path / "reports",
    )


def test_tracking_report_renders_the_card_from_persisted_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _config()
    _bundle(tmp_path).record_training_truth(_frame())
    monkeypatch.setenv("BROADWAY_TRACKING_DIR", str(tmp_path / "tracking"))
    monkeypatch.setenv("BROADWAY_REPORTS_DIR", str(tmp_path / "reports"))

    tracking_report.run(dataset="test", environment="development")

    card = tmp_path / "reports" / config.data_card.output_file
    assert card.exists()
    assert card.read_text(encoding="utf-8").startswith("# Data Card — test")
    assert "wrote" in capsys.readouterr().out


def test_publish_data_card_without_a_baseline_fails_loud(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)

    with pytest.raises(FileNotFoundError, match="run the train step first"):
        bundle.publish_data_card()


def test_publish_data_card_ignores_absent_profile_and_clean_result(tmp_path: Path) -> None:
    config = _config().model_copy(
        update={"profile_file": str(tmp_path / "absent_profile.json")}
    )
    bundle = _bundle(tmp_path, config=config)
    bundle.record_training_truth(_frame())

    card = bundle.publish_data_card().read_text(encoding="utf-8")

    assert "profiled source: unknown" in card
    assert "## Exclusions" in card


def test_publish_data_card_ignores_a_foreign_profile(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        build_profile("houses", "houses.parquet", _frame()).model_dump_json(indent=2),
        encoding="utf-8",
    )
    config = _config().model_copy(update={"profile_file": str(profile_path)})
    bundle = _bundle(tmp_path, config=config)
    bundle.record_training_truth(_frame())

    card = bundle.publish_data_card().read_text(encoding="utf-8")

    assert "profiled source: unknown" in card


def test_publish_data_card_uses_a_matching_profile(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        build_profile("test", "test.parquet", _frame()).model_dump_json(indent=2),
        encoding="utf-8",
    )
    config = _config().model_copy(update={"profile_file": str(profile_path)})
    bundle = _bundle(tmp_path, config=config)
    bundle.record_training_truth(_frame())

    card = bundle.publish_data_card().read_text(encoding="utf-8")

    assert bundle.machine_dir == tmp_path / "tracking"
    assert "profiled source: 8" in card


def test_tracking_report_requires_a_tracking_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load_config("train", dataset="test", environment="development")
    monkeypatch.setattr(
        tracking_report,
        "load_config",
        lambda *args, **kwargs: cfg.model_copy(update={"train": None}),
    )

    with pytest.raises(ValueError, match="tracking section"):
        tracking_report.run(dataset="test", environment="development")
