"""07_report: report plots — score distribution, PR curve, threshold sweep,
error breakdown by attribute, field-ablation bars, and the data-scaling curve.

Reads the zero-shot bi-encoder embeddings (cached) for the first four; the
field-ablation and data-scaling plots read 07c_field_ablation.csv and
07d_data_scaling.csv when they exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _blocking import build_pairs
from _common import PATHS, SEED, load_dataset_deduped
from _text import MACRO_MAP, extract_volume_ml
from sklearn.metrics import average_precision_score, precision_recall_curve

from broadway.training.nlp import encode_corpus

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE = str(PATHS.experiments.parent / "data" / "euromonitor" / "embeddings_cache")
RESULTS = PATHS.experiments / "results" / "euromonitor"


def main() -> None:
    df = load_dataset_deduped()
    payload = (df["title"].fillna("") + " | " + df["brand"].fillna("") + " | " + df["category"].fillna("")).tolist()
    pos, neg = build_pairs(df, SEED, 4, 10_000)
    emb, _ = encode_corpus(MODEL, payload, batch_size=256, max_seq_length=128, cache_dir=CACHE)

    pos_s = (emb[pos[:, 0]] * emb[pos[:, 1]]).sum(axis=1)
    neg_s = (emb[neg[:, 0]] * emb[neg[:, 1]]).sum(axis=1)
    y = np.r_[np.ones(len(pos_s)), np.zeros(len(neg_s))]
    scores = np.r_[pos_s, neg_s]
    thr = float(np.quantile(neg_s, 0.95))

    # ---- 1. score distribution ----
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    bins = np.linspace(0, 1, 61)
    ax.hist(neg_s, bins=bins, alpha=0.6, color="#C44E52", label=f"negative (n={len(neg_s):,})")
    ax.hist(pos_s, bins=bins, alpha=0.6, color="#4C72B0", label=f"positive (n={len(pos_s):,})")
    ax.axvline(thr, color="k", ls="--", lw=1, label=f"thr@5%FPR={thr:.3f}")
    ax.set_xlabel("cosine"); ax.set_ylabel("pairs"); ax.legend()
    ax.set_title("Score distribution (zero-shot MiniLM-L6)")
    fig.savefig(RESULTS / "07_report_score_dist.png", dpi=150); plt.close(fig)

    # ---- 2. precision-recall curve ----
    precision, recall, _ = precision_recall_curve(y, scores)
    ap = average_precision_score(y, scores)
    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    ax.plot(recall, precision, color="#4C72B0", lw=2, label=f"AP = {ap:.4f}")
    ax.fill_between(recall, precision, alpha=0.15, color="#4C72B0")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve"); ax.legend(loc="lower left")
    fig.savefig(RESULTS / "07_report_pr_curve.png", dpi=150); plt.close(fig)

    # ---- 3. threshold sweep: precision / recall / F1 ----
    ths = np.linspace(0, 1, 201)
    tp = np.array([(pos_s >= t).sum() for t in ths], dtype=float)
    fp = np.array([(neg_s >= t).sum() for t in ths], dtype=float)
    fn = len(pos_s) - tp
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p + r)
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(ths, p, color="#4C72B0", label="precision")
    ax.plot(ths, r, color="#55A868", label="recall")
    ax.plot(ths, f1, color="#C44E52", label="F1")
    ax.axvline(thr, color="k", ls="--", lw=1, label=f"thr@5%FPR={thr:.3f}")
    ax.set_xlabel("cosine threshold"); ax.set_ylabel("rate")
    ax.set_title("Precision / recall / F1 vs threshold"); ax.legend()
    fig.savefig(RESULTS / "07_report_threshold_sweep.png", dpi=150); plt.close(fig)

    # ---- 4. error breakdown by attribute ----
    brand = df["brand"].fillna("").astype(str).to_numpy()
    cat = df["category"].fillna("").astype(str).to_numpy()
    macro = df["category"].fillna("").map(lambda c: MACRO_MAP.get(c, "?")).to_numpy()
    vol = df["title"].fillna("").map(extract_volume_ml).map(lambda t: t[0]).to_numpy()

    def shares(pairs, mask):
        a, b = pairs[mask][:, 0], pairs[mask][:, 1]
        va, vb = vol[a], vol[b]
        both = ~pd.isna(va) & ~pd.isna(vb)
        return {
            "same brand": (brand[a] == brand[b]).mean(),
            "same category": (cat[a] == cat[b]).mean(),
            "same macro": (macro[a] == macro[b]).mean(),
            "same volume": (both & (va == vb)).sum() / max(both.sum(), 1),
        }

    fn = pos_s < thr
    fp = neg_s > thr
    data = {
        "FN (missed matches)": shares(pos, fn),
        "FP (false matches)": shares(neg, fp),
    }
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    edf = pd.DataFrame(data).T[["same brand", "same category", "same macro", "same volume"]]
    edf.plot(kind="bar", ax=ax, rot=0)
    ax.set_ylabel("share of errors"); ax.set_xlabel("")
    ax.set_title("Error breakdown by attribute")
    fig.savefig(RESULTS / "07_report_error_breakdown.png", dpi=150); plt.close(fig)

    # ---- 5. field-ablation bars (if CSV exists) ----
    fab = RESULTS / "07c_field_ablation.csv"
    if fab.exists():
        fdf = pd.read_csv(fab).set_index("variant")
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        fdf[["average_precision", "precision_at_90pct_recall"]].plot(kind="bar", ax=ax, rot=0)
        ax.set_ylabel("score"); ax.set_xlabel("payload variant")
        ax.set_title("Field ablation — AP and precision@90%recall per variant")
        fig.savefig(RESULTS / "07_report_field_ablation.png", dpi=150); plt.close(fig)
        print(f"field ablation plot written (n={len(fdf)})", flush=True)
    else:
        print("field ablation CSV not ready yet", flush=True)

    # ---- 6. data-scaling curve (if CSV exists) ----
    dsc = RESULTS / "07d_data_scaling.csv"
    if dsc.exists():
        ddf = pd.read_csv(dsc).sort_values("n_triples")
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        ax.plot(ddf["n_triples"], ddf["average_precision"], marker="o", color="#4C72B0", label="AP")
        ax.plot(ddf["n_triples"], ddf["precision_at_90pct_recall"], marker="s", color="#C44E52", label="precision@90%recall")
        ax.set_xscale("log"); ax.set_xlabel("n_triples (log)"); ax.set_ylabel("score")
        ax.set_title("Data-scaling curve — hard-band performance vs train size"); ax.legend()
        fig.savefig(RESULTS / "07_report_data_scaling.png", dpi=150); plt.close(fig)
        print(f"data-scaling plot written (n={len(ddf)})", flush=True)
    else:
        print("data-scaling CSV not ready yet", flush=True)

    print("report plots done", flush=True)


if __name__ == "__main__":
    main()
