"""06: model verdict scorecard — the three baselines and how to use them.

Reads the persisted baseline evidence (baseline_lightgbm.json from step 05;
fails loudly if missing) and packages the answers: each population's role and
verdict (analyst interpretation, from config) alongside its holdout metrics
(evidence). Renders a one-glance scorecard figure and writes the verdict JSON
plus a tracked CSV. Evidence and interpretation stay separated — metrics come
from the evidence file, roles/verdicts from config.
"""

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from _setup import RESULTS, load_config

CSV_STEM = Path(__file__).stem
COLORS = {"A": "#4C72B0", "B": "#DD8452", "C": "#55A868"}


def load_evidence(cfg: dict) -> dict:
    """Holdout metrics per population from the step-05 evidence file."""
    path = RESULTS / "baseline_lightgbm.json"
    if not path.exists():
        raise FileNotFoundError("baseline_lightgbm.json missing — run "
                                "05_lightgbm_baselines.py first")
    return {p["population"]: p
            for p in json.loads(path.read_text())["populations"]}


def build_rows(cfg: dict, evidence: dict) -> list[dict]:
    """Merge config roles/verdicts with evidence metrics per population."""
    rows = []
    for name, spec in cfg["model_verdicts"].items():
        ev = evidence[name]
        rows.append({
            "population": name,
            "role": spec["role"],
            "mae": ev["mae"],
            "rmse": ev["rmse"],
            "r2": ev["r2"],
            "tail_mae": ev["tail_mae"],
            "verdict": spec["verdict"],
        })
    return rows


def plot_scorecard(rows: list[dict], out: Path) -> None:
    """One-glance scorecard: role, key metrics, and verdict per model."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ax, row in zip(axes, rows):
        pop = row["population"]
        ax.axis("off")
        ax.add_patch(plt.Rectangle(
            (0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
            facecolor=COLORS[pop], alpha=0.12,
            edgecolor=COLORS[pop], linewidth=2))
        ax.text(0.5, 0.90, f"Model {pop}", ha="center", fontsize=15,
                fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.80, row["role"], ha="center", fontsize=13,
                color=COLORS[pop], transform=ax.transAxes)
        ax.text(0.5, 0.60, f"MAE  ${row['mae']:.2f}", ha="center",
                fontsize=16, transform=ax.transAxes)
        ax.text(0.5, 0.50, f"tail MAE  ${row['tail_mae']:.2f}", ha="center",
                fontsize=11, transform=ax.transAxes)
        ax.text(0.5, 0.41, f"R2  {row['r2']:.4f}", ha="center", fontsize=11,
                transform=ax.transAxes)
        verdict = "\n".join(textwrap.wrap(row["verdict"], 34))
        ax.text(0.5, 0.20, verdict, ha="center", va="center", fontsize=9,
                transform=ax.transAxes)
    fig.suptitle("Model roles — evidence-backed baselines "
                 "(total_amount holdout)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    RESULTS.mkdir(parents=True, exist_ok=True)

    rows = build_rows(cfg, load_evidence(cfg))
    for row in rows:
        print(f"Model {row['population']} ({row['role']}): "
              f"MAE ${row['mae']:.2f}, tail ${row['tail_mae']:.2f}, "
              f"R2 {row['r2']:.4f} — {row['verdict']}")

    plot_scorecard(rows, RESULTS / f"{CSV_STEM}.png")
    payload = {
        "note": ("metrics are evidence (baseline_lightgbm.json); "
                 "roles/verdicts are analyst interpretation "
                 "(config model_verdicts)"),
        "models": rows,
    }
    out = RESULTS / "model_verdict.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {CSV_STEM}.png / model_verdict.json")

    csv = RESULTS / f"{CSV_STEM}.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
