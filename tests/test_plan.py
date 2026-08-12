from __future__ import annotations

import json
from pathlib import Path

from broadway.stats.plan import AnalysisPlan, load_plan, save_plan


def _make_plan(next_step: str | None) -> AnalysisPlan:
    return AnalysisPlan(
        script="04_anova",
        analysis_type="group_comparison",
        test_name="one-way ANOVA",
        statistics={"p_value": 0.03, "statistic": 4.2},
        effect_sizes={"eta_squared": 0.21, "omega_squared": 0.15},
        threshold_context={"imbalance_ratio": 2.0, "any_small_group": False},
        reason=["p < 0.05", "large effect size"],
        warnings=["underpowered"],
        passed=True,
        next_step=next_step,
    )


def test_plan_round_trip_preserves_all_fields(tmp_path: Path) -> None:
    plan = _make_plan(None)
    path = tmp_path / "plan.json"
    save_plan(plan, path)
    loaded = load_plan(path)
    assert loaded == plan
    assert loaded.next_step is None


def test_plan_round_trip_with_next_step(tmp_path: Path) -> None:
    plan = _make_plan("05_post_hoc")
    path = tmp_path / "plan.json"
    save_plan(plan, path)
    assert load_plan(path) == plan


def test_save_plan_writes_indented_json_with_null(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    save_plan(_make_plan(None), path)
    raw = path.read_text()
    assert json.loads(raw)["next_step"] is None
    assert "\n" in raw
