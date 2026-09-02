"""07e: Two-stage cross-encoder reranker on the euromonitor hard band.

Targeted A/B on top of the existing euromonitor NLP series (07_nlp_hpo,
07b_finetune, 07c_field_ablation, 07d_data_scaling): the bi-encoder
(sentence-transformers/all-MiniLM-L6-v2, already cached) encodes the corpus once
and scores candidate pairs by cosine. Only the pairs that land in the confusion
band are re-scored by a cross-encoder. Everything else — hard-negative mining,
ground-truth pairs, metric helpers — is reused unchanged from the existing infra.

Band discrepancy (per brief): the MINING band stays 0.45-0.80 (mine_hard_negatives
defaults, pinned by 07b's COSINE_LO/COSINE_HI) because its job is RECALL of the
negative pool — it over-collects the confusable region so the hard-negative eval
pool has enough volume. The reranker APPLICATION band is 0.50-0.75 because its
job is PRECISION spend on the highest-confidence confusion region. The two bands
serve different stages and do not coincide. The eval pool is drawn from the wider
0.45-0.80 band, but the cross-encoder only re-scores the 0.50-0.75 slice (plus any
positives that happen to fall in it); pairs in 0.45-0.50 and 0.75-0.80 remain
"bi-encoder-only" in the hybrid score, so the A/B is honest about what the
reranker actually touches.

Evaluation (auditable, bi-encoder-only vs hybrid, plus an in-band-only
cross-encoder row): precision@90%recall via the SHARED
broadway.training.nlp.precision_at_recall_breakdown, fixed-threshold precision
(tau=0.5) plus a 0.40/0.50/0.60/0.70/0.75 sweep, and band-crossing counts that
attribute the precision delta to named pairs. Every metric is also reported
per-country-stratum (in-country / cross-country / unknown-country positives,
never pooled only), with per-stratum pair counts and a bootstrap 95% CI on
precision@90%recall above a ~30-positive floor.

Default cross-encoder: cross-encoder/ms-marco-MiniLM-L-6-v2 (cached; 22M params,
6-layer BERT, regression head -> sigmoid scores in [0, 1]). Override with
--cross-model (e.g. BAAI/bge-reranker-v2-m3) to re-run the A/B against another
cached checkpoint without code edits.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from _blocking import build_pairs
from _common import PATHS, RESULTS, SEED, load_dataset_deduped
from _hard_negatives import mine_hard_negatives
from _text import MACRO_MAP

from broadway.training.nlp import (
    _cosine,
    calibrate_isotonic_heldout,
    encode_corpus,
    precision_at_recall_breakdown,
    precision_ci,
    split_pos_by_country,
)

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CACHE = str(PATHS.experiments.parent / "data" / "euromonitor" / "embeddings_cache")

# Mining band (pool-generation recall knob) vs application band (precision spend
# knob) are DIFFERENT stages and are not required to coincide — see docstring.
MINING_LO = 0.45
MINING_HI = 0.80
APP_LO = 0.50
APP_HI = 0.75
N_TARGET = 20_000
K = 40
BOOTSTRAP = 2_000
FIXED_TAU = 0.5
SWEEP_TAUS = (0.40, 0.50, 0.60, 0.70, 0.75)
CI_FLOOR = 30
TARGET_RECALL = 0.90

STRATA = ("pooled", "in-country", "cross-country", "unknown-country")

RERANK_COLUMNS = [
    "scorer", "metric", "country_stratum", "n_pos", "n_neg",
    "precision", "tp", "fp", "threshold", "recall", "ci_lo", "ci_hi",
    "tp_recovered", "fp_removed", "tp_lost", "fp_added", "tau_bi", "tau_hybrid",
    "final_score", "brier",
]

PAIRS_COLUMNS = [
    "title_a", "title_b", "brand_a", "brand_b", "barcode_a", "barcode_b",
    "macro_a", "macro_b", "country_a", "country_b", "country_stratum",
    "cosine", "in_band", "cross_score", "hybrid_score", "final_score", "label",
]


def _fixed_breakdown(
    pos_s: np.ndarray, neg_s: np.ndarray, tau: float
) -> tuple[float, int, int, float, float]:
    """(precision, TP, FP, tau, recall) at a fixed >= tau operating point."""
    if len(pos_s) == 0 or len(neg_s) == 0:
        return float("nan"), 0, 0, float(tau), float("nan")
    tp = int((pos_s >= tau).sum())
    fp = int((neg_s >= tau).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / len(pos_s)
    return precision, tp, fp, float(tau), recall


def country_stratum_vector(
    pairs: np.ndarray, country: np.ndarray, positive: bool
) -> np.ndarray:
    """Per-pair country_stratum label (positive pairs only; negatives are n/a).

    Delegates to the shared ``broadway.training.nlp.split_pos_by_country`` so
    the country-classification logic has a single source.
    """
    if not positive:
        return np.full(len(pairs), "n/a-for-negatives", dtype=object)
    same, cross, _ = split_pos_by_country(pairs, country)
    out = np.full(len(pairs), "unknown-country", dtype=object)
    out[cross] = "cross-country"
    out[same] = "in-country"
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Map cross-encoder raw logits to [0, 1] relevance probabilities.

    sentence-transformers 6.0.1 defaults ``CrossEncoder.predict`` to an
    Identity() activation (raw logits), NOT sigmoid — verified empirically for
    ms-marco-MiniLM-L-6-v2 (num_labels=1, activation_fn=Identity). ms-marco and
    bge-reranker cross-encoders are regression heads meant to be sigmoid-mapped,
    so apply it here rather than trusting the library's default.
    """
    return 1.0 / (1.0 + np.exp(-x))


def _assert_finite_in_unit(name: str, arr: np.ndarray) -> None:
    """Loud sanity check: scores are finite and within [0, 1]."""
    assert np.isfinite(arr).all(), f"{name} has non-finite scores"
    assert bool(((arr >= 0.0) & (arr <= 1.0)).all()), f"{name} scores out of [0, 1]"


def _hybrid_score(
    in_band: np.ndarray, ce_scores: np.ndarray, cosine_scores: np.ndarray
) -> np.ndarray:
    """In-band pairs take the cross-encoder score; out-of-band keep the cosine.

    The bi-encoder cosine is the raw dot product of L2-normalized float32
    embeddings, which can round just above 1.0 (or below 0.0) for a
    near-identical pair. Clamp it into [0, 1] here so the hybrid score is always
    a valid probability-like value. ``pos_s``/``hard_s`` are also clamped at
    their source so the bi-encoder scorer and the per-pair audit share the same
    scale.
    """
    return np.where(in_band, ce_scores, np.clip(cosine_scores, 0.0, 1.0))


def eval_scorer(
    scorer: str,
    pos_s: np.ndarray,
    neg_s: np.ndarray,
    pos_pairs: np.ndarray,
    country: np.ndarray,
) -> list[dict]:
    """One metric-row set (precision@90R + fixed-threshold sweep) per stratum."""
    rows: list[dict] = []
    same, cross, unknown = split_pos_by_country(pos_pairs, country)
    masks = {
        "pooled": np.ones(len(pos_pairs), dtype=bool),
        "in-country": same,
        "cross-country": cross,
        "unknown-country": unknown,
    }
    for stratum in STRATA:
        mask = masks[stratum]
        n_pos = int(mask.sum())
        n_neg = len(neg_s)
        if n_pos == 0:
            p90 = tp90 = fp90 = float("nan")
            thr90 = rec90 = ci_lo = ci_hi = float("nan")
        else:
            p90, tp90, fp90, thr90, rec90 = precision_at_recall_breakdown(
                pos_s[mask], neg_s, target_recall=TARGET_RECALL
            )
            if n_pos >= CI_FLOOR and np.isfinite(p90):
                _, ci_lo, ci_hi = precision_ci(
                    pos_s[mask], neg_s, target_recall=TARGET_RECALL,
                    n_boot=BOOTSTRAP, seed=SEED,
                )
            else:
                ci_lo = ci_hi = float("nan")
        rows.append({
            "scorer": scorer,
            "metric": "precision_at_90_recall",
            "country_stratum": stratum,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "precision": p90,
            "tp": tp90,
            "fp": fp90,
            "threshold": thr90,
            "recall": rec90,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        })
        for tau in SWEEP_TAUS:
            p, tp, fp, thr, rec = _fixed_breakdown(pos_s[mask], neg_s, tau)
            rows.append({
                "scorer": scorer,
                "metric": "precision_at_tau",
                "country_stratum": stratum,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "precision": p,
                "tp": tp,
                "fp": fp,
                "threshold": thr,
                "recall": rec,
                "ci_lo": float("nan"),
                "ci_hi": float("nan"),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-stage bi-encoder + cross-encoder rerank A/B on the hard band"
    )
    parser.add_argument("--cross-model", default=CROSS_MODEL,
                        help="cross-encoder HF repo id (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="cross-encoder predict batch size")
    parser.add_argument("--device", default="cpu", help="torch device for the cross-encoder")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    df = load_dataset_deduped()
    payload = (
        df["title"].fillna("") + " | " + df["brand"].fillna("")
        + " | " + df["category"].fillna("")
    ).tolist()
    country = df["country"].fillna("").astype(str).to_numpy()

    # Cross-country barcode-group census (01g expression), recomputed on the
    # deduped frame this step consumes (matching-stage input).
    barcodes = df["barcode"].fillna("").astype(str)
    known = df[barcodes.str.len() > 0]
    multi = known[known.groupby("barcode")["retailer"].transform("nunique") > 1]
    grp_countries = multi.groupby("barcode")["country"].nunique()
    n_cross_country_groups = int((grp_countries > 1).sum())
    print(f"cross-country barcode groups (deduped): {n_cross_country_groups:,}", flush=True)

    # ---- stage 1: bi-encoder, encode-once (cached) ----
    emb, encode_s = encode_corpus(MODEL, payload, batch_size=256,
                                  max_seq_length=128, cache_dir=CACHE)
    print(f"corpus encode_s = {encode_s:.1f}s", flush=True)

    # ---- ground truth + hard-negative pool (REUSED) ----
    pos, _neg_rand = build_pairs(df, SEED, 4, 10_000)
    t_mine = time.perf_counter()
    hard_pairs, _hard_cos = mine_hard_negatives(
        df, emb, n_target=N_TARGET, cosine_lo=MINING_LO, cosine_hi=MINING_HI, k=K
    )
    print(f"hard negatives mined (band {MINING_LO}-{MINING_HI}, target {N_TARGET:,}): "
          f"{len(hard_pairs):,} in {time.perf_counter() - t_mine:.1f}s", flush=True)

    # Bi-encoder cosine is a raw dot product of L2-normalized float32 embeddings:
    # a near-identical pair can round slightly above 1.0 (or negative). Clamp to
    # [0, 1] so the band mask, bi-encoder scorer, and per-pair audit stay on one
    # consistent probability-like scale.
    pos_s = np.clip(_cosine(emb, pos), 0.0, 1.0)
    hard_s = np.clip(_cosine(emb, hard_pairs), 0.0, 1.0)

    # ---- application band mask (0.5-0.75) over BOTH labels ----
    pos_in_band = (pos_s >= APP_LO) & (pos_s <= APP_HI)
    hard_in_band = (hard_s >= APP_LO) & (hard_s <= APP_HI)
    n_total_pool = len(pos) + len(hard_pairs)
    n_in_band = int(pos_in_band.sum() + hard_in_band.sum())
    print(f"eval pool: total {n_total_pool:,} | in-band reranked {n_in_band:,} "
          f"| out-of-band kept {n_total_pool - n_in_band:,}", flush=True)

    # ---- loud sanity checks (07_nlp_hpo S1-S4 fail-loud pattern) ----
    assert n_in_band > 0, "in-band subset empty — cross-encoder has nothing to re-score"
    assert np.isfinite(pos_s).all() and np.isfinite(hard_s).all(), \
        "bi-encoder cosine has non-finite scores"

    # ---- stage 2: cross-encoder re-scores ONLY in-band pairs ----
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder(args.cross_model, device=args.device, max_length=512)
    pos_in_idx = np.flatnonzero(pos_in_band)
    hard_in_idx = np.flatnonzero(hard_in_band)
    # build_pairs / mine_hard_negatives already emit pairs in a < b row order, so
    # no endpoint canonicalization is needed (order-sensitivity is negligible at
    # this scale — see spec §6).
    pos_pairs_in = [(payload[a], payload[b]) for a, b in pos[pos_in_idx]]
    hard_pairs_in = [(payload[a], payload[b]) for a, b in hard_pairs[hard_in_idx]]

    t_ce = time.perf_counter()
    ce_pos = _sigmoid(np.asarray(ce.predict(
        pos_pairs_in, batch_size=args.batch_size, show_progress_bar=False
    ), dtype=float)) if pos_pairs_in else np.empty(0)
    ce_hard = _sigmoid(np.asarray(ce.predict(
        hard_pairs_in, batch_size=args.batch_size, show_progress_bar=False
    ), dtype=float)) if hard_pairs_in else np.empty(0)
    print(f"cross-encoder predict_s = {time.perf_counter() - t_ce:.1f}s "
          f"({len(pos_pairs_in) + len(hard_pairs_in):,} pairs)", flush=True)

    # full-length cross-encoder arrays (NaN where out-of-band = not re-scored)
    ce_pos_full = np.full(len(pos_s), np.nan)
    ce_hard_full = np.full(len(hard_s), np.nan)
    ce_pos_full[pos_in_idx] = ce_pos
    ce_hard_full[hard_in_idx] = ce_hard

    # ---- hybrid score: in-band -> cross-encoder, out-of-band -> cosine ----
    pos_hybrid = _hybrid_score(pos_in_band, ce_pos_full, pos_s)
    hard_hybrid = _hybrid_score(hard_in_band, ce_hard_full, hard_s)

    _assert_finite_in_unit("pos hybrid", pos_hybrid)
    _assert_finite_in_unit("hard hybrid", hard_hybrid)
    _assert_finite_in_unit("pos cross", ce_pos)
    _assert_finite_in_unit("hard cross", ce_hard)

    # ---- isotonic calibration: monotone remap of hybrid scores -> [0, 1] ----
    # Honest held-out calibration: fit the monotone map on a deterministic 70%
    # fit split, apply it to the FULL set (rank order preserved -> ranking
    # metrics unchanged), and report Brier on the held-out 30% ONLY (never
    # in-sample).
    cal_scores = np.r_[pos_hybrid, hard_hybrid]
    cal_labels = np.r_[np.ones(len(pos_hybrid)), np.zeros(len(hard_hybrid))]
    cal_all, brier = calibrate_isotonic_heldout(cal_scores, cal_labels)
    pos_final = cal_all[: len(pos_hybrid)]
    hard_final = cal_all[len(pos_hybrid):]
    print(f"isotonic calibration: ranking preserved (AUC unchanged by monotonicity) "
          f"| held-out brier {brier:.4f}", flush=True)

    # ---- metric rows (bi-only vs hybrid vs in-band-only cross-encoder) ----
    scorers = {
        "bi_encoder": (pos_s, hard_s, pos),
        "hybrid": (pos_hybrid, hard_hybrid, pos),
        # scale-isolated row: re-score all in-band pairs, evaluate only them
        "cross_in_band": (ce_pos, ce_hard, pos[pos_in_idx]),
    }
    rows: list[dict] = []
    for scorer, (p_s, n_s, p_pairs) in scorers.items():
        rows.extend(eval_scorer(scorer, p_s, n_s, p_pairs, country))

    # ---- band-crossing diagnostic (bi -> hybrid operating-threshold delta) ----
    _, _, _, tau_bi, _ = precision_at_recall_breakdown(pos_s, hard_s,
                                                       target_recall=TARGET_RECALL)
    _, _, _, tau_hybrid, _ = precision_at_recall_breakdown(pos_hybrid, hard_hybrid,
                                                           target_recall=TARGET_RECALL)
    neg_in = hard_in_band
    fp_removed = int(((hard_s >= tau_bi) & (hard_hybrid <= tau_hybrid) & neg_in).sum())
    fp_added = int(((hard_s <= tau_bi) & (hard_hybrid >= tau_hybrid) & neg_in).sum())
    pos_in = pos_in_band
    tp_recovered = int(((pos_s <= tau_bi) & (pos_hybrid >= tau_hybrid) & pos_in).sum())
    tp_lost = int(((pos_s >= tau_bi) & (pos_hybrid <= tau_hybrid) & pos_in).sum())
    print(f"band crossing (tau_bi={tau_bi:.4f} -> tau_hybrid={tau_hybrid:.4f}): "
          f"TP recovered {tp_recovered} | FP removed {fp_removed} | "
          f"TP lost {tp_lost} | FP added {fp_added}", flush=True)
    rows.append({
        "scorer": "hybrid",
        "metric": "band_crossing",
        "country_stratum": "pooled",
        "n_pos": int(pos_in.sum()),
        "n_neg": int(neg_in.sum()),
        "tp_recovered": tp_recovered,
        "fp_removed": fp_removed,
        "tp_lost": tp_lost,
        "fp_added": fp_added,
        "tau_bi": tau_bi,
        "tau_hybrid": tau_hybrid,
    })
    # One held-out calibration-quality number on the rerank rollup (the per-pair
    # final_score lives in 07e_cross_encoder_pairs.csv).
    rows.append({
        "scorer": "hybrid",
        "metric": "isotonic_calibration",
        "country_stratum": "pooled",
        "n_pos": len(pos_hybrid),
        "n_neg": len(hard_hybrid),
        "brier": round(brier, 4),
    })

    rerank_df = pd.DataFrame(rows)
    for col in RERANK_COLUMNS:
        if col not in rerank_df.columns:
            rerank_df[col] = np.nan
    rerank_df = rerank_df[RERANK_COLUMNS]
    rerank_df.to_csv(RESULTS / "07e_cross_encoder_rerank.csv", index=False)

    # ---- per-pair audit CSV (pos=1 / hard-neg=0) ----
    title = df["title"].fillna("").astype(str).to_numpy()
    brand = df["brand"].fillna("").astype(str).to_numpy()
    barcode = df["barcode"].fillna("").astype(str).to_numpy()
    macro = df["category"].fillna("").map(lambda c: MACRO_MAP.get(c, "?")).to_numpy()

    def pair_frame(pairs, cosine, in_band, cross_full, hybrid, final, label):
        a, b = pairs[:, 0], pairs[:, 1]
        return pd.DataFrame({
            "title_a": title[a], "title_b": title[b],
            "brand_a": brand[a], "brand_b": brand[b],
            "barcode_a": barcode[a], "barcode_b": barcode[b],
            "macro_a": macro[a], "macro_b": macro[b],
            "country_a": country[a], "country_b": country[b],
            "country_stratum": country_stratum_vector(pairs, country, label == 1),
            "cosine": np.round(cosine, 4),
            "in_band": in_band.astype(bool),
            "cross_score": np.round(cross_full, 4),
            "hybrid_score": np.round(hybrid, 4),
            "final_score": np.round(final, 4),
            "label": label,
        })

    pos_df = pair_frame(pos, pos_s, pos_in_band, ce_pos_full, pos_hybrid, pos_final, 1)
    neg_df = pair_frame(hard_pairs, hard_s, hard_in_band, ce_hard_full, hard_hybrid, hard_final, 0)
    pairs_df = pd.concat([pos_df, neg_df], ignore_index=True)[PAIRS_COLUMNS]
    pairs_df.to_csv(RESULTS / "07e_cross_encoder_pairs.csv", index=False)

    # ---- concise stdout summary ----
    n_cross_country_pos = int((country_stratum_vector(pos, country, True) == "cross-country").sum())
    print(f"cross-country positive pairs: {n_cross_country_pos:,} "
          f"(groups {n_cross_country_groups:,})", flush=True)
    for scorer, (p_s, n_s, _p_pairs) in scorers.items():
        p90, tp, fp, thr, rec = precision_at_recall_breakdown(p_s, n_s,
                                                              target_recall=TARGET_RECALL)
        pfix, tpf, fpf, _, recf = _fixed_breakdown(p_s, n_s, FIXED_TAU)
        print(f"{scorer}: P@90R {p90:.4f} (TP {tp} FP {fp} thr {thr:.4f} recall {rec:.4f}) "
              f"| fix@0.5 P {pfix:.4f} (TP {tpf} FP {fpf} recall {recf:.4f})", flush=True)
    print(f"wrote {RESULTS / '07e_cross_encoder_rerank.csv'} ({len(rerank_df)} rows) "
          f"+ {RESULTS / '07e_cross_encoder_pairs.csv'} ({len(pairs_df):,} rows)", flush=True)
    print(f"TOTAL {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
