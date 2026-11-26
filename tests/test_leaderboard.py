"""Leaderboard: one ranked figure per metric, top 3 highlighted per direction."""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import pandas as pd
import pytest

from broadway.evaluate.leaderboard import MetricDirection, MetricLeaderboard

_MUTED = "#bdbdbd"


def test_leaderboard_plots_one_ranked_figure_per_metric(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "run": ["a", "b", "c", "d", "e"],
            "rmse": [0.90, 0.20, 0.50, 0.70, 0.40],
            "r2": [0.60, 0.90, 0.70, 0.50, 0.80],
        }
    )
    board = MetricLeaderboard.from_frame(
        frame,
        id_column="run",
        metrics={"rmse": MetricDirection.MINIMIZE, "r2": MetricDirection.MAXIMIZE},
    )

    figures = board.figures()

    assert list(figures) == ["rmse", "r2"]
    assert board.top("rmse") == ["b", "e", "c"]
    assert board.top("r2") == ["b", "e", "c"]
    orderings = {
        "rmse": ["b", "e", "c", "d", "a"],
        "r2": ["b", "e", "c", "a", "d"],
    }
    for metric, expected in orderings.items():
        axis = figures[metric].axes[0]
        assert [tick.get_text() for tick in axis.get_yticklabels()] == expected
        colors = [mcolors.to_hex(patch.get_facecolor()) for patch in axis.patches]
        assert len(set(colors[:3])) == 3
        assert set(colors[3:]) == {_MUTED}
        assert [text.get_text().split()[0] for text in axis.texts[:3]] == ["#1", "#2", "#3"]
        assert all("#" not in text.get_text() for text in axis.texts[3:])
    assert "lower is better" in figures["rmse"].axes[0].get_title()
    assert "higher is better" in figures["r2"].axes[0].get_title()

    written = board.write(tmp_path, "leaderboard")

    assert [path.name for path in written.values()] == [
        "leaderboard_rmse.png",
        "leaderboard_r2.png",
    ]
    assert all(path.exists() for path in written.values())


def test_champion_picks_the_best_run_with_its_measured_interval() -> None:
    frame = pd.DataFrame(
        {
            "run": ["a", "b", "c"],
            "rmse": [0.90, 0.20, 0.50],
        }
    )
    board = MetricLeaderboard.from_frame(
        frame, id_column="run", metrics={"rmse": MetricDirection.MINIMIZE}
    )

    champion = board.champion(
        "rmse", {"a": (0.85, 0.95), "b": (0.18, 0.24), "c": (0.47, 0.53)}
    )

    assert champion.champion == "b"
    assert champion.value == 0.20
    assert (champion.interval_low, champion.interval_high) == (0.18, 0.24)
    assert champion.runner_up == "c"
    assert champion.margin == pytest.approx(0.30)
    assert champion.n_runs == 3
