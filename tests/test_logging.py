"""Public behavior of the canonical process logging setup (LOGGING.md §1-3)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

from broadway.config.loader import load_environment
from broadway.config.schema import EnvironmentConfig
from broadway.observability.logging import setup_logging


def _environment(tmp_path: Path, level: str) -> EnvironmentConfig:
    return load_environment("development").model_copy(
        update={"log_level": level, "log_dir": str(tmp_path / "logs")}
    )


def test_setup_is_idempotent_and_writes_the_configured_file(tmp_path: Path) -> None:
    environment = _environment(tmp_path, level="DEBUG")
    setup_logging(environment)
    setup_logging(environment)

    root = logging.getLogger()
    file_handlers = [h for h in root.handlers if type(h) is logging.FileHandler]
    assert len(file_handlers) == 1
    assert len([h for h in root.handlers if type(h) is logging.StreamHandler]) == 1
    assert root.level == logging.DEBUG

    log_file = environment.log_file
    assert log_file is not None
    log_path = tmp_path / "logs" / log_file
    assert log_path.exists()
    assert log_path.name == log_file
    logging.getLogger("broadway.test").info("logging-probe")
    assert "logging-probe" in log_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("field", ["log_dir", "log_file"])
def test_missing_log_path_field_raises_naming_it(tmp_path: Path, field: str) -> None:
    environment = _environment(tmp_path, level="INFO").model_copy(update={field: None})
    with pytest.raises(ValueError, match=field):
        setup_logging(environment)


def test_cli_main_writes_run_id_in_the_configured_log_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from broadway import cli
    from broadway.discover import module as discover_module

    def _log_probe(*_args: object) -> None:
        logging.getLogger("broadway.cli_probe").info("cli-run-probe")

    monkeypatch.setattr(discover_module, "run", _log_probe)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ds-pipeline", "discover", "--csv", "a.csv",
            "--target", "price", "--task", "regression",
        ],
    )
    monkeypatch.setenv("BROADWAY_RUN_ID", "cli-run-id")
    cli.main()

    log_text = (tmp_path / "artifacts" / "logs" / "broadway.log").read_text(encoding="utf-8")
    assert "run=cli-run-id" in log_text
    assert "cli-run-probe" in log_text
