from __future__ import annotations

from pathlib import Path

import pytest

from broadway.causal import module
from broadway.causal.contracts import load_design
from broadway.config.loader import load_config


def test_causal_run_persists_design(tmp_path: Path) -> None:
    cfg = load_config("causal")
    assert cfg.causal is not None

    cfg = cfg.model_copy(
        update={"causal": cfg.causal.model_copy(update={"output_dir": str(tmp_path)})}
    )

    module.run(cfg)

    design_path = tmp_path / cfg.causal.output_file
    assert design_path.exists()

    design = load_design(design_path)
    assert design.treatment_column == cfg.causal.treatment_column
