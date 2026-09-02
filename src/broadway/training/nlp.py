"""NLP entity-resolution HPO — adapt the optuna bandit to embedding models.

The sklearn HPO pipeline (broadway.training.hpo) treats each model as a
sklearn Pipeline built from a PipelineConfig and minimizes a regression metric.
This module adapts the SAME bandit machinery to NLP: each HPO "model" is a
bi-encoder (sentence-transformers), the objective scores text-pair entity
resolution (positive pairs vs negative pairs) as ROC AUC and
RECORDS the corpus encode latency, and the direction is "maximize" (higher AUC
is better). The bandit allocation, seeded TPE, mlflow per-trial callback, and
RDB storage contract are all reused unchanged from hpo.py / optuna.py.

The objective is the pure zero-shot benchmark: one trial per model that loads
+ encodes + scores (no fine-tune). The model zoo (short name -> HF repo id) is
config SSOT and passed in by the caller — this module never hardcodes a model
list.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from broadway.config.schema import HPOConfig, NLPConfig
from broadway.training.hpo import Objective, run_hpo_bandit

# Honest isotonic calibration: fit on a deterministic fraction of the
# calibration input and report Brier on the HELD-OUT fraction only, so the
# reported calibration quality is never in-sample (see calibrate_isotonic_heldout).
CALIBRATION_SEED = 42
CALIBRATION_FRAC = 0.70


def _cosine(emb: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Cosine similarity for index-aligned pair arrays (rows of a matrix)."""
    return (emb[pairs[:, 0]] * emb[pairs[:, 1]]).sum(axis=1)


def entity_resolution_metrics(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    fpr: float = 0.05,
    target_recall: float = 0.90,
) -> dict[str, float]:
    """Ranking metrics for a bi-encoder's pos-vs-neg score populations.

    pos_scores are true-pair similarities, neg_scores are non-pair similarities.
    Returns ROC AUC, PR-AUC (average precision), recall at a fixed FPR budget,
    precision at a target recall (operating-point precision via the positive
    score distribution), F1 at the operating threshold, and score-distribution
    summaries used for the threshold trade-off.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        raise ValueError("entity_resolution_metrics requires non-empty pos and neg scores")
    y = np.r_[np.ones(len(pos_scores)), np.zeros(len(neg_scores))]
    scores = np.r_[pos_scores, neg_scores]
    auc = float(roc_auc_score(y, scores))
    ap = float(average_precision_score(y, scores))

    # operating threshold: recall at fixed FPR budget
    thr = float(np.quantile(neg_scores, 1 - fpr))
    tp_op = float((pos_scores >= thr).sum())
    fp_op = float((neg_scores >= thr).sum())
    recall_op = tp_op / len(pos_scores)
    precision_op = tp_op / (tp_op + fp_op) if (tp_op + fp_op) > 0 else 0.0
    f1_op = (2 * precision_op * recall_op / (precision_op + recall_op)
             if (precision_op + recall_op) > 0 else 0.0)

    prec90, tp90, fp90, thr90, _ = precision_at_recall_breakdown(
        pos_scores, neg_scores, target_recall=target_recall
    )

    return {
        "auc": round(auc, 4),
        "average_precision": round(ap, 4),
        f"recall_at_{int(fpr * 100)}pct_fpr": round(recall_op, 4),
        f"precision_at_{int(target_recall * 100)}pct_recall": round(float(prec90), 4),
        "f1_at_5pct_fpr": round(f1_op, 4),
        # NEW — auditable raw counts / thresholds (backward-compatible additions):
        "tp_at_90pct_recall": round(float(tp90), 4),
        "fp_at_90pct_recall": round(float(fp90), 4),
        "threshold_at_90pct_recall": round(thr90, 4),
        "tp_at_5pct_fpr": round(float(tp_op), 4),
        "fp_at_5pct_fpr": round(float(fp_op), 4),
        "threshold_at_5pct_fpr": round(thr, 4),
        "pos_median": round(float(np.median(pos_scores)), 4),
        "neg_p90": round(float(np.quantile(neg_scores, 0.9)), 4),
    }


def precision_at_recall_breakdown(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    target_recall: float = 0.90,
) -> tuple[float, int, int, float, float]:
    """(precision, TP, FP, threshold, recall) at the target-recall operating point.

    Threshold is the positive-score quantile that retains exactly target_recall
    of positives; TP/FP are the >= threshold counts; precision = TP/(TP+FP)
    (0.0 when no pair clears the threshold); recall = TP/len(pos_scores).

    Empty-population semantics match precision_at_recall: an empty positive
    set has no threshold and an empty negative set would make precision
    spuriously perfect, so both return (NaN, 0, 0, NaN, NaN).
    """
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan"), 0, 0, float("nan"), float("nan")
    threshold = float(np.quantile(pos_scores, 1 - target_recall))
    tp = int((pos_scores >= threshold).sum())
    fp = int((neg_scores >= threshold).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / len(pos_scores)
    return precision, tp, fp, threshold, recall


def calibrate_isotonic(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Monotone isotonic regression of binary labels on scores -> probabilities.

    A thin post-processing layer for a reranker: fit sklearn's
    ``IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")`` on
    ``(scores, labels)`` with labels in {0, 1} and return the transformed
    scores. Isotonic regression is a non-parametric monotone fit, so it
    preserves the rank order of the input scores (ROC AUC is unchanged) while
    remapping the raw score axis onto calibrated probabilities in [0, 1].

    Degenerate fallback (documented): when ``scores`` is empty or constant
    (fewer than two distinct values) there is no monotone map to learn, so the
    input scores are returned unchanged instead of crashing.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if scores.size == 0 or np.unique(scores).size < 2:
        return scores
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(scores, labels)
    return iso.predict(scores)


def calibrate_isotonic_heldout(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int = CALIBRATION_SEED,
    frac: float = CALIBRATION_FRAC,
) -> tuple[np.ndarray, float]:
    """Honest isotonic calibration with a held-out Brier score.

    Splits the calibration input into a deterministic fit split (``frac`` of the
    rows) and a held-out split (the rest). Isotonic regression is fit on the fit
    split ONLY and then applied to the FULL input, so the returned ``final``
    scores are monotone in the input (rank order preserved -> ranking metrics
    unchanged) while the returned ``brier`` is computed exclusively on the
    held-out split — never in-sample.

    Degenerate fallback: when the input is empty, constant, or too small to
    leave a non-empty holdout there is no honest held-out estimate, so the input
    scores are returned unchanged with ``brier = NaN``.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    n = scores.size
    if n == 0 or np.unique(scores).size < 2:
        return scores, float("nan")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_fit = int(n * frac)
    if n_fit < 1 or n_fit >= n:
        return scores, float("nan")
    fit_idx = perm[:n_fit]
    hold_idx = perm[n_fit:]
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(scores[fit_idx], labels[fit_idx])
    final = iso.predict(scores)
    brier = float(np.mean((iso.predict(scores[hold_idx]) - labels[hold_idx]) ** 2))
    return final, brier


def precision_at_recall(pos_scores: np.ndarray, neg_scores: np.ndarray, target_recall: float = 0.90) -> float:
    """Precision at the score threshold keeping exactly target_recall of positives.

    See precision_at_recall_breakdown for the formula and empty-population
    NaN semantics. Backward-compatible wrapper returning only the precision.
    """
    precision, _, _, _, _ = precision_at_recall_breakdown(
        pos_scores, neg_scores, target_recall=target_recall
    )
    return precision


def precision_ci(
    pos_s: np.ndarray,
    neg_s: np.ndarray,
    target_recall: float = 0.90,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """precision@target-recall point estimate + bootstrap 95% CI.

    Resamples the positive and negative score populations independently
    ``n_boot`` times and reports the 2.5/97.5 percentiles of the resampled
    precision-at-recall. A non-finite point estimate short-circuits to
    ``(point, NaN, NaN)``. Single source for the euromonitor 07b/07e eval
    steps (previously duplicated as ``_precision_ci`` in both scripts).
    """
    point = float(precision_at_recall(pos_s, neg_s, target_recall=target_recall))
    if not np.isfinite(point):
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, len(neg_s), len(neg_s))
        j = rng.integers(0, len(pos_s), len(pos_s))
        boots[b] = precision_at_recall(pos_s[j], neg_s[i], target_recall=target_recall)
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def split_pos_by_country(
    pos_pairs: np.ndarray, country: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split positive pairs into (same, cross, unknown) country strata, in order.

    Per-pair, per spec: ``same`` (in-country) iff both sides have a non-empty
    country and the values are equal; ``cross`` (cross-country) iff both sides
    non-empty and different; ``unknown`` iff either side is empty. The unknown
    stratum is a SEPARATE scored stratum — empty country is never folded into
    in-country and never mislabeled as cross-country. Single source shared by
    the euromonitor 07b and 07e steps.
    """
    a = pos_pairs[:, 0]
    b = pos_pairs[:, 1]
    has_country = (country[a] != "") & (country[b] != "")
    cross = has_country & (country[a] != country[b])
    same = has_country & (country[a] == country[b])
    unknown = ~has_country
    return same, cross, unknown


def log_nlp_eval(
    metrics: dict[str, float],
    params: dict[str, float | int | str],
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> None:
    """Log one NLP eval run to MLflow; a no-op when no tracking URI is set.

    When tracking_uri is None, do nothing (the euromonitor experiments read
    MLFLOW_TRACKING_URI from the environment and must stay runnable without a
    server). Otherwise set up the store and log params + metrics.
    """
    if tracking_uri is None:
        return
    from broadway.training.mlflow_utils import log_metrics, log_params, setup_mlflow

    setup_mlflow(tracking_uri, experiment_name or "nlp-eval")
    log_params(params)
    log_metrics(metrics)


def _embedding_cache_path(
    cache_dir: str | None,
    model_id: str,
    payload: list[str],
    max_seq_length: int,
    batch_size: int,
    prompt: str | None = None,
) -> Path | None:
    """Deterministic .npz cache path for a model+payload (None disables cache)."""
    if cache_dir is None:
        return None
    digest = hashlib.sha1("\x1f".join(payload).encode("utf-8")).hexdigest()[:12]
    key = f"{model_id.replace('/', '_')}_{max_seq_length}_{batch_size}_{prompt or ''}_{digest}"
    return Path(cache_dir) / f"emb_{key}.npz"


def _encode_payload(
    model, payload: list[str], batch_size: int, cache_path: Path | None, prompt: str | None = None
):
    """Encode the corpus once; reuse a cached .npz (emb + encode_s) on re-run.

    Encode-once, score-many: the cache holds the raw embedding matrix and the
    original encode wall-time so a warm re-run reports the SAME latency instead
    of a misleading 0. Fine-tuned models never cache (params vary per trial).
    ``prompt`` (e.g. ``query: `` for e5-family models) is prepended to every
    text by sentence-transformers.
    """
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path) as cached:
            return cached["emb"], float(cached["encode_s"])
    t0 = time.perf_counter()
    emb = model.encode(
        payload,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        prompt=prompt,
    )
    encode_s = time.perf_counter() - t0
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, emb=emb, encode_s=encode_s)
    return emb, encode_s


def encode_corpus(
    model_id: str,
    payload: list[str],
    *,
    device: str = "cpu",
    batch_size: int = 256,
    max_seq_length: int = 128,
    cache_dir: str | None = None,
    prompt: str | None = None,
) -> tuple[np.ndarray, float]:
    """Encode a corpus once; reuse the .npz cache. Returns (emb, encode_s).

    The reusable, non-HPO entry point for the encode-once/score-many path that
    ``make_objective`` uses internally: load one bi-encoder, encode the payload,
    and return ``(embeddings, encode_seconds)``. With ``cache_dir`` set, a warm
    call returns the cached matrix (same payload + knobs) without re-encoding,
    while still reporting the original encode wall-time. Callers can persist the
    returned matrix as a first-class artifact and pass it to downstream scoring
    instead of re-encoding per experiment.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id, device=device, model_kwargs={"torch_dtype": torch.float32})
    model.max_seq_length = max_seq_length
    cache_path = _embedding_cache_path(
        cache_dir, model_id, payload, max_seq_length, batch_size, prompt
    )
    return _encode_payload(model, payload, batch_size, cache_path, prompt)


def _finetune(
    model,
    params: dict[str, float | int],
    examples,
    max_seq_length: int,
    loss: str = "mnrl",
    *,
    log_steps: bool = False,
) -> None:
    """Fine-tune a fresh base model in place (no disk save).

    The caller owns the model object; fit() mutates it, so encoding afterwards
    uses the fine-tuned weights. ``loss`` selects the bi-encoder objective:
    ``mnrl`` (MultipleNegativesRankingLoss — positive-only examples, negatives
    mined in-batch), ``contrastive`` (ContrastiveLoss — labeled pos/neg
    examples), or ``triplet`` (TripletLoss — anchor/positive/negative triples).
    The caller must build ``examples`` to match the chosen loss. ``learning_rate``
    is passed through optimizer_params (sentence-transformers has no top-level lr
    knob); when absent the library default is used, so the argument is omitted
    rather than passed as None.

    ``log_steps`` prints one ``[step i/N] loss X`` line per training batch by
    wrapping the loss module's forward — a live progress signal for long
    CPU-only runs where fit() otherwise emits nothing between folds (its tqdm
    bar shows position but not the loss).
    """
    from sentence_transformers.sentence_transformer import losses
    from torch.utils.data import DataLoader

    model.max_seq_length = max_seq_length
    epochs = int(params.get("epochs", 1))
    batch_size = int(params.get("batch_size", 32))
    warmup_steps = int(params.get("warmup_steps", 0))
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    train_loss: Any
    if loss == "contrastive":
        train_loss = losses.ContrastiveLoss(model)
    elif loss == "triplet":
        train_loss = losses.TripletLoss(model)
    else:
        train_loss = losses.MultipleNegativesRankingLoss(model)

    if log_steps:
        num_steps = max(1, (len(examples) + batch_size - 1) // batch_size) * epochs
        step_state = {"step": 0}
        _orig_forward = train_loss.forward

        def _logging_forward(sentence_features, labels):
            step_state["step"] += 1
            loss_val = _orig_forward(sentence_features, labels)
            print(f"    [step {step_state['step']}/{num_steps}] "
                  f"loss {float(loss_val.detach()):.4f}", flush=True)
            return loss_val

        train_loss.forward = _logging_forward

    fit_kwargs: dict = {
        "epochs": epochs,
        "warmup_steps": warmup_steps,
        "output_path": None,
        "show_progress_bar": False,
    }
    if "learning_rate" in params:
        fit_kwargs["optimizer_params"] = {"lr": float(params["learning_rate"])}
    model.fit(train_objectives=[(loader, train_loss)], **fit_kwargs)


def make_objective(
    model_id: str,
    payload: list[str],
    pos_pairs: np.ndarray,
    neg_pairs: np.ndarray,
    *,
    device: str = "cpu",
    batch_size: int = 256,
    max_seq_length: int = 128,
    cache_dir: str | None = None,
    prompt: str | None = None,
) -> Objective:
    """Build the zero-shot NLP objective for one bi-encoder: encode -> score -> AUC.

    The returned objective takes the active optuna trial as an optional second
    argument and, when given, attaches the full metric set (auc, recall at a
    5% FPR budget, score summaries, and the corpus encode latency in seconds)
    to the trial as the "broadway_metrics" user attr — the same contract the
    hpo.py mlflow callback logs for every trial.

    The objective is the pure zero-shot benchmark (load -> encode -> score).
    ``cache_dir`` enables encode-once/score-many: zero-shot embeddings are
    cached to .npz so a warm re-run skips the encode but still reports the
    original encode latency. ``prompt`` is prepended to every text before
    encoding (e5-family models).
    """
    from sentence_transformers import SentenceTransformer

    cache_path = _embedding_cache_path(
        cache_dir, model_id, payload, max_seq_length, batch_size, prompt
    )

    def objective(params: dict[str, float | int], trial=None) -> float:
        model = SentenceTransformer(model_id, device=device, model_kwargs={"torch_dtype": torch.float32})
        model.max_seq_length = max_seq_length
        emb, encode_s = _encode_payload(model, payload, batch_size, cache_path, prompt)
        metrics = entity_resolution_metrics(_cosine(emb, pos_pairs), _cosine(emb, neg_pairs))
        metrics["encode_s"] = round(encode_s, 3)
        if trial is not None:
            trial.set_user_attr("broadway_metrics", metrics)
        return metrics["auc"]

    return objective


def _extract_broadway_metrics(study) -> dict[str, float]:
    """Read the objective's broadway_metrics user attr from a study's best trial."""
    return {
        key: float(value)
        for key, value in study.best_trial.user_attrs.get("broadway_metrics", {}).items()
    }


def run_nlp_hpo(
    model_zoo: dict[str, str],
    hpo_cfg: HPOConfig,
    payload: list[str],
    pos_pairs: np.ndarray,
    neg_pairs: np.ndarray,
    *,
    seed: int,
    device: str = "cpu",
    batch_size: int = 256,
    max_seq_length: int = 128,
    cache_dir: str | None = None,
    prompts: dict[str, str] | None = None,
    mlflow_tracking: bool = False,
    mlflow_tags: dict[str, str] | None = None,
) -> dict:
    """Run the bandit HPO over the embedding model zoo (direction: maximize).

    model_zoo maps the HPO spec's short names to HF repo ids; every
    hpo_cfg.models entry must resolve through it (loud failure otherwise).
    ``prompts`` optionally maps a short name to a text prefix (e.g. ``query: ``
    for e5-family models) applied before encoding. The bandit, seeded TPE,
    mlflow callback, and RDB contract are reused from run_hpo_bandit unchanged.
    The returned dict adds a "metrics" map (name -> the best trial's
    broadway_metrics) so callers can build the benchmark table from plain data.
    """
    prompts = prompts or {}
    unknown = [spec.name for spec in hpo_cfg.models if spec.name not in model_zoo]
    if unknown:
        raise ValueError(f"hpo.models names missing from model_zoo: {sorted(unknown)}")
    objectives = {
        spec.name: make_objective(
            model_zoo[spec.name],
            payload,
            pos_pairs,
            neg_pairs,
            device=device,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            cache_dir=cache_dir,
            prompt=prompts.get(spec.name),
        )
        for spec in hpo_cfg.models
    }
    return run_hpo_bandit(
        objectives, hpo_cfg, seed, mlflow_tracking, mlflow_tags,
        metric_extractor=_extract_broadway_metrics,
    )


def run_nlp(
    cfg: NLPConfig,
    payload: list[str],
    pos_pairs: np.ndarray,
    neg_pairs: np.ndarray,
    *,
    mlflow_tracking: bool = False,
    mlflow_tags: dict[str, str] | None = None,
) -> dict:
    """Run the NLP HPO bandit from a typed config (data-agnostic entry point).

    Parallel to broadway.training.module.run: the typed NLPConfig carries the
    model zoo + bandit spec + encode knobs, and the caller supplies the payload
    and ground-truth pair indices, so this module stays dataset-agnostic. Returns
    the plain-data result ({models, best_model, best_params, best_value,
    metrics, and a failed map when some model raised}).
    """
    return run_nlp_hpo(
        cfg.model_zoo,
        cfg.hpo,
        payload,
        pos_pairs,
        neg_pairs,
        seed=cfg.seed,
        device=cfg.device,
        batch_size=cfg.batch_size,
        max_seq_length=cfg.max_seq_length,
        cache_dir=cfg.cache_dir,
        prompts=cfg.prompts,
        mlflow_tracking=mlflow_tracking,
        mlflow_tags=mlflow_tags,
    )


def load_pairs_csv(path: str | Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load a generic pairs CSV (title_a, title_b, label) into NLP inputs.

    Data-agnostic loader, the NLP analogue of demo/demo.csv: dedupes all titles
    into a payload and maps each row to payload indices; label == 1 rows become
    positive (match) pairs and label == 0 rows become negative (non-match)
    pairs. Empty title cells and non-numeric labels degrade to ""/0.
    """
    import pandas as pd

    frame = pd.read_csv(path)
    a = frame["title_a"].fillna("").astype(str).str.strip().tolist()
    b = frame["title_b"].fillna("").astype(str).str.strip().tolist()
    labels = pd.to_numeric(frame["label"], errors="coerce").fillna(0).astype(int).tolist()
    payload = list(dict.fromkeys(a + b))
    index = {text: i for i, text in enumerate(payload)}
    pos = [[index[x], index[y]] for x, y, lab in zip(a, b, labels) if lab == 1]
    neg = [[index[x], index[y]] for x, y, lab in zip(a, b, labels) if lab == 0]
    pos_arr = np.array(pos, dtype=int).reshape(-1, 2) if pos else np.zeros((0, 2), dtype=int)
    neg_arr = np.array(neg, dtype=int).reshape(-1, 2) if neg else np.zeros((0, 2), dtype=int)
    return payload, pos_arr, neg_arr
