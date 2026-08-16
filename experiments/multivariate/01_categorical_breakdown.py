"""01: multivariate categorical breakdown — describe-style figures + evidence.

For every categorical column in the working dataset (auto-detected: object /
bool / category dtypes plus low-cardinality ints; constants and configured
exclusions skipped; derived time_bucket included), computes the value counts
(top-N) and per-group fare median (the analyst view), reuses the platform's
describe / plot_describe_figures for the legend_experiment/right/describe.png
style figure (boxplot of fare by group + group sizes/imbalance), and persists
a per-dataset JSON plus one tracked CSV per category. All knobs come from
config.yaml; the loader is owned by the univariate experiment.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from _setup import (
    RESULTS,
    WORKING_DATASET,
    load_config,
    load_manhattan_sample,
    require_keys,
)
from broadway.stats.describe import describe, plot_describe_figures

CSV_STEM = Path(__file__).stem


def detect_categoricals(df: pd.DataFrame, cfg: dict) -> list[str]:
    """Auto-detect categorical columns (config thresholds; constants skipped)."""
    max_unique = cfg["categorical"]["max_unique_for_int"]
    exclude = set(cfg["categorical"]["exclude"])
    detected = []
    for col in df.columns:
        if col in exclude:
            continue
        dtype = str(df[col].dtype)
        n_unique = int(df[col].nunique(dropna=True))
        is_cat = (df[col].dtype == object or dtype.startswith("bool")
                  or dtype == "category"
                  or (dtype.startswith("int") and 1 < n_unique <= max_unique))
        if is_cat and n_unique > 1:
            detected.append(col)
    for col in cfg["categorical"]["extra_categories"]:
        if col not in df.columns:
            raise ValueError(f"extra category '{col}' configured but not in data")
        if col not in detected and df[col].nunique() > 1:
            detected.append(col)
    if not detected:
        raise ValueError("no categorical columns detected — check config thresholds")
    return detected


def label(value: object, col: str, cfg: dict) -> str:
    """Human label for a category value when configured (else raw)."""
    labels = cfg["labels"].get(col)
    if labels is not None and value in labels:
        return f"{value} ({labels[value]})"
    return str(value)


def category_stats(df: pd.DataFrame, col: str, cfg: dict) -> dict:
    """Value counts (top-N) + median/mean/std per group, keyed by RAW value."""
    top_n = cfg["value_counts_head"]
    target = cfg["target"]
    counts = df[col].value_counts().head(top_n)
    if counts.empty:
        raise ValueError(f"category '{col}' produced no groups (all missing?)")
    groups = {g: df[df[col] == g][target].dropna() for g in counts.index}
    return {
        "column": col,
        "total_n": int(len(df)),
        "counts": {g: int(c) for g, c in counts.items()},
        "median_fare": {g: float(v.median()) for g, v in groups.items() if len(v)},
        "mean_fare": {g: float(v.mean()) for g, v in groups.items() if len(v)},
        "std_fare": {g: float(v.std()) for g, v in groups.items() if len(v)},
    }


def render_describe_figure(df: pd.DataFrame, col: str, cfg: dict,
                           out_path: Path) -> dict:
    """Platform describe-style figure (boxplot + group sizes); returns summary."""
    top_n = cfg["value_counts_head"]
    target = cfg["target"]
    # describe() requires string group keys, so work on a str-typed copy.
    plot_df = df.copy()
    plot_df[col] = plot_df[col].astype(str)
    group_values = [str(g) for g in df[col].value_counts().head(top_n).index]
    summary = describe(plot_df, col, col, group_values, target,
                       str(WORKING_DATASET), cfg["sample"]["name"],
                       cfg["sample_role"])
    plot_describe_figures(plot_df, col, col, group_values, target,
                          summary, out_path)
    return summary.model_dump()


def process_category(df: pd.DataFrame, col: str, cfg: dict,
                     evidence: dict) -> None:
    """Print, figure, CSV, and evidence block for one category."""
    stats = category_stats(df, col, cfg)
    print(f"\n--- {col} ---")
    for g, c in stats["counts"].items():
        print(f"  {label(g, col, cfg)}: {c}")
    print(pd.Series({label(g, col, cfg): v
                     for g, v in stats["median_fare"].items()})
          .rename("median fare").to_string())

    summary = render_describe_figure(df, col, cfg,
                                     RESULTS / f"{CSV_STEM}_{col}.png")
    print(f"imbalance ratio: {summary['imbalance_ratio']} | "
          f"groups: {len(summary['groups'])} | wrote {CSV_STEM}_{col}.png")

    stats["proportions"] = {g: summary["proportions"][str(g)]
                            for g in stats["counts"]}
    stats["imbalance_ratio"] = summary["imbalance_ratio"]
    evidence[col] = stats

    pd.DataFrame([
        {
            "value": label(g, col, cfg), "count": stats["counts"][g],
            "proportion": stats["proportions"][g],
            "median_fare": stats["median_fare"][g],
            "mean_fare": stats["mean_fare"][g],
            "std_fare": stats["std_fare"][g],
        }
        for g in stats["counts"]
    ]).to_csv(RESULTS / f"{CSV_STEM}_{col}.csv", index=False)


def main() -> None:
    cfg = load_config()
    require_keys(cfg, ["target", "sample", "sample_role"], "01 config")
    df = load_manhattan_sample(cfg)
    RESULTS.mkdir(parents=True, exist_ok=True)

    columns = detect_categoricals(df, cfg)
    print(f"detected categorical columns: {columns}")

    evidence = {}
    for col in columns:
        process_category(df, col, cfg, evidence)

    payload = {
        "dataset": WORKING_DATASET.name,
        "source_script": Path(__file__).name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": cfg["target"],
        "categories": evidence,
    }
    out = RESULTS / f"{cfg['sample']['name']}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")
    print(f"wrote CSVs: {sorted(p.name for p in RESULTS.glob(f'{CSV_STEM}_*.csv'))}")


if __name__ == "__main__":
    main()
