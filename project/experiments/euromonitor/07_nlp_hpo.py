"""07: NLP entity-resolution HPO — benchmark the embedding model zoo via Optuna.

Adapts the platform optuna bandit (broadway.training.hpo.run_hpo_bandit) to the
euromonitor matching task: each HPO "model" is a bi-encoder from the model zoo
(`project/config/experiments/nlp.yaml`), the objective scores title-pair entity
resolution (same-barcode positives vs cross-barcode negatives) as ROC AUC
(maximize), and every trial records the corpus encode latency. The bandit,
seeded TPE, mlflow per-trial callback, and RDB contract are reused unchanged.

Ground truth is the SAME 12,038 positive / 10,000 negative pair population as
step 04 (the TF-IDF baseline), so the two scoring layers are directly
comparable. The payload is title | brand | category. Cross-encoders are out of
scope here (they are the two-stage rerank, a different objective).

Writes (RESULTS = project/experiments/results/euromonitor/):
  07_nlp_hpo_benchmark.csv   per-model AUC / recall@5%FPR / encode latency
  07_nlp_hpo_pareto.png      AUC vs encode-seconds (the Pareto frontier)
MLflow runs (nested per trial) land in the ambient tracking store via the
shared hpo callback when mlflow_tracking is enabled.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from _blocking import build_pairs
from _common import PATHS, RESULTS, load_dataset

from broadway.config.schema import NLPConfig
from broadway.timing import TimingReport
from broadway.training import nlp
from broadway.training.mlflow_utils import setup_mlflow

CSV_BENCH = RESULTS / "07_nlp_hpo_benchmark.csv"
CSV_TIMING = RESULTS / "07_nlp_hpo_timing.csv"
PNG_PARETO = RESULTS / "07_nlp_hpo_pareto.png"
CONFIG_PATH = PATHS.experiment_configs / "nlp.yaml"

MAX_POS_PAIRS_PER_GROUP = 4
N_NEG_PAIRS = 10_000
EXPERIMENT = "euromonitor_nlp"
PAYLOAD_SEP = " | "


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    cfg = NLPConfig(**yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")))
    if cfg.cache_dir:
        cfg = cfg.model_copy(update={"cache_dir": str(PATHS.experiments.parent / cfg.cache_dir)})

    report = TimingReport()
    with report.record("data_load"):
        df = load_dataset()
    with report.record("payload_build"):
        payload = (
            df["title"].fillna("") + PAYLOAD_SEP + df["brand"].fillna("") + PAYLOAD_SEP + df["category"].fillna("")
        ).tolist()
    with report.record("pair_build"):
        pos_pairs, neg_pairs = build_pairs(df, cfg.seed, MAX_POS_PAIRS_PER_GROUP, N_NEG_PAIRS)
    print(f"payload: {len(payload):,} rows | "
          f"positive pairs: {len(pos_pairs):,} | negative pairs: {len(neg_pairs):,}")

    setup_mlflow(str(PATHS.root / "mlruns"), EXPERIMENT)
    with report.record("hpo_benchmark"):
        result = nlp.run_nlp(
            cfg,
            payload,
            pos_pairs,
            neg_pairs,
            mlflow_tracking=True,
            mlflow_tags={"experiment": EXPERIMENT, "seed": str(cfg.seed)},
        )

    timing_frame = pd.DataFrame(
        [{"phase": name, **entry} for name, entry in report.as_dict().items()]
    )
    timing_frame.to_csv(CSV_TIMING, index=False)
    print(f"wrote {CSV_TIMING} (phase timing profile, {len(timing_frame)} rows)")

    failed = result.get("failed", {})
    if failed:
        print(f"\nWARNING: {len(failed)} model(s) failed:")
        for name, err in sorted(failed.items()):
            print(f"  {name}: {err}")

    # ---- display table: per-model benchmark ---------------------------------
    rows = []
    for name, summary in result["models"].items():
        m = result["metrics"].get(name, {})
        rows.append({
            "model": name,
            "auc": summary["best_value"],
            "recall_at_5pct_fpr": m.get("recall_at_5pct_fpr"),
            "encode_s": m.get("encode_s"),
            "pos_median": m.get("pos_median"),
            "neg_p90": m.get("neg_p90"),
            "n_trials": summary["n_trials"],
        })
    frame = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    frame.to_csv(CSV_BENCH, index=False)
    print(f"\nwrote {CSV_BENCH} (display table, {len(frame)} rows)")

    # ---- sanity (fail loudly) -------------------------------------------------
    checks = [
        ("S1 every model produced a valid trial",
         len(result["models"]) == len(cfg.hpo.models)),
        ("S2 every AUC is a probability", bool(frame["auc"].between(0.0, 1.0).all())),
        ("S3 every encode latency is positive", bool((frame["encode_s"] > 0).all())),
        ("S4 a best model was selected", result["best_model"] in result["models"]),
    ]
    for name, ok in checks:
        if not ok:
            raise AssertionError(f"sanity check FAILED: {name}")

    # ---- Pareto frontier -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    ax.scatter(frame["encode_s"], frame["auc"], s=70, color="#4C72B0", zorder=3)
    for _, r in frame.iterrows():
        ax.annotate(r["model"], (r["encode_s"], r["auc"]),
                    fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("encode time (s, full corpus)")
    ax.set_ylabel("ROC AUC")
    ax.set_title("Embedding model zoo — AUC vs encode latency (Pareto frontier)")
    ax.set_xlim(0, frame["encode_s"].max() * 1.15)
    ax.set_ylim(0, 1.0)
    fig.savefig(PNG_PARETO, dpi=150)
    plt.close(fig)
    print(f"wrote {PNG_PARETO}")

    print("\n=== NLP embedding benchmark (zero-shot, same pairs as 04) ===")
    print(frame.to_string(index=False))
    print(f"\nbest model: {result['best_model']} (AUC={result['best_value']:.4f})")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


if __name__ == "__main__":
    main()
