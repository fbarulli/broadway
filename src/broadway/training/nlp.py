"""NLP entity-resolution HPO — adapt the optuna bandit to embedding models.

The sklearn HPO pipeline (broadway.training.hpo) treats each model as a
sklearn Pipeline built from a PipelineConfig and minimizes a regression metric.
This module adapts the SAME bandit machinery to NLP: each HPO "model" is a
bi-encoder (sentence-transformers), the objective scores text-pair entity
resolution (positive pairs vs negative pairs) as ROC AUC and
RECORDS the corpus encode latency, and the direction is "maximize" (higher AUC
is better). The bandit allocation, seeded TPE, mlflow per-trial callback, and
RDB storage contract are all reused unchanged from hpo.py / optuna.py.

The objective's search space drives a light contrastive fine-tune
(MultipleNegativesRankingLoss) when the spec declares fine-tune params and
training examples are supplied; an empty search space is the zero-shot
benchmark (one trial that just loads + encodes + scores). The model zoo
(short name -> HF repo id) is config SSOT and passed in by the caller — this
module never hardcodes a model list.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from broadway.config.schema import HPOConfig, NLPConfig
from broadway.training.hpo import Objective, run_hpo_bandit

# Fine-tune params that, when present in a trial's search space, switch the
# objective from zero-shot scoring to a contrastive fine-tune. Names match the
# sentence-transformers fit() contract (epochs, warmup_steps, batch_size,
# learning_rate via optimizer_params).
_FINETUNE_PARAMS = ("epochs", "learning_rate", "warmup_steps", "batch_size")


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

    return {
        "auc": round(auc, 4),
        "average_precision": round(ap, 4),
        f"recall_at_{int(fpr * 100)}pct_fpr": round(recall_op, 4),
        f"precision_at_{int(target_recall * 100)}pct_recall": round(
            float(precision_at_recall(pos_scores, neg_scores, target_recall=target_recall)), 4),
        "f1_at_5pct_fpr": round(f1_op, 4),
        "pos_median": round(float(np.median(pos_scores)), 4),
        "neg_p90": round(float(np.quantile(neg_scores, 0.9)), 4),
    }


def precision_at_recall(pos_scores: np.ndarray, neg_scores: np.ndarray, target_recall: float = 0.90) -> float:
    """Precision at the score threshold that keeps exactly target_recall of positives.

    The threshold is set on the POSITIVE score distribution (the quantile that
    retains target_recall of true pairs), then precision = TP / (TP + FP) at
    that threshold. This is the business-relevant number for the scoring stage:
    at target recall, how many of the flagged pairs are actually the same product.

    Returns NaN when either score population is empty: an empty negative set
    would otherwise make precision spuriously 1.0 (a division with no false
    positives), and an empty positive set has no threshold to define.
    """
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")
    threshold = float(np.quantile(pos_scores, 1 - target_recall))
    pos_at = float((pos_scores >= threshold).sum())
    neg_at = float((neg_scores >= threshold).sum())
    return pos_at / (pos_at + neg_at) if (pos_at + neg_at) > 0 else 0.0


def group_kfold(
    group_ids: Sequence[int],
    k: int,
    *,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Split pair indices into k folds such that no group spans two folds.

    Contract: ``group_ids[i]`` is the group label of the i-th PAIR — one label
    per pair. For entity resolution the group is the barcode: a positive
    (same-barcode) pair carries the single shared barcode; a negative
    (cross-barcode) pair is labeled by the barcode of its FIRST row (the
    caller decides this convention; the splitter only guarantees that pairs
    sharing a label stay together). Every unique group is assigned to exactly
    one fold, so two rows that share a barcode are never split across
    train/val.

    Folds are balanced by pair count, not group count: groups are ordered
    largest-first and greedily assigned to the currently smallest fold (the
    classic greedy number-partitioning heuristic). ``seed`` only shuffles
    ties (equal-size groups) for reproducibility — set it to get a
    deterministic fold layout, or leave it None for a deterministic
    first-appearance tie order.

    Returns a list of k boolean masks over ``group_ids``: ``masks[f][i]`` is
    True iff pair ``i`` belongs to fold ``f``. Each pair is in exactly one fold.
    """
    if k < 2:
        raise ValueError(f"group_kfold requires k >= 2, got {k}")
    ids = np.asarray(group_ids)
    if ids.ndim != 1:
        raise ValueError("group_ids must be a 1-dimensional sequence")
    if ids.size == 0:
        raise ValueError("group_ids must be non-empty")
    unique, counts = np.unique(ids, return_counts=True)
    if len(unique) < k:
        raise ValueError(
            f"cannot build {k} folds from only {len(unique)} distinct groups"
        )
    groups = list(zip(unique.tolist(), counts.tolist()))
    if seed is not None:
        rng = np.random.default_rng(seed)
        rng.shuffle(groups)
    # Stable sort keeps the (possibly shuffled) tie order while still putting
    # the largest groups first for greedy balance.
    groups.sort(key=lambda group_count: -group_count[1])
    fold_totals = np.zeros(k, dtype=int)
    group_to_fold: dict[int, int] = {}
    for group, count in groups:
        fold = int(np.argmin(fold_totals))
        fold_totals[fold] += count
        group_to_fold[group] = fold
    fold_of_pair = np.array([group_to_fold[int(g)] for g in ids], dtype=int)
    return [fold_of_pair == f for f in range(k)]


def _aggregate_fold_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Collapse per-fold metric dicts into mean/std plus per-fold values.

    Each per-fold dict has identical keys (entity_resolution_metrics + encode_s);
    the aggregate adds ``{key}_mean``, ``{key}_std`` and ``fold_{f}_{key}`` for
    every key, so MLflow logs the honest held-out estimate plus its variance.
    """
    if not fold_metrics:
        raise ValueError("fold_metrics must be non-empty")
    keys = list(fold_metrics[0])
    out: dict[str, float] = {}
    for key in keys:
        values = [fold[key] for fold in fold_metrics]
        out[f"{key}_mean"] = round(float(np.mean(values)), 4)
        out[f"{key}_std"] = round(float(np.std(values)), 4)
        for f, value in enumerate(values):
            out[f"fold_{f}_{key}"] = round(float(value), 4)
    return out


def _has_finetune_params(params: dict[str, float | int]) -> bool:
    return any(name in params for name in _FINETUNE_PARAMS)


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
    finetune_examples=None,
    device: str = "cpu",
    batch_size: int = 256,
    max_seq_length: int = 128,
    cache_dir: str | None = None,
    prompt: str | None = None,
    loss: str = "mnrl",
    cv_folds: int | None = None,
    pos_groups: Sequence[int] | None = None,
    neg_groups: Sequence[int] | None = None,
    finetune_groups: Sequence[int] | None = None,
    cv_seed: int | None = None,
) -> Objective:
    """Build the NLP objective for one bi-encoder: encode -> score -> AUC.

    The returned objective takes the active optuna trial as an optional second
    argument and, when given, attaches the full metric set (auc, recall at a
    5% FPR budget, score summaries, and the corpus encode latency in seconds)
    to the trial as the "broadway_metrics" user attr — the same contract the
    hpo.py mlflow callback logs for every trial.

    With finetune_examples and a search space carrying fine-tune params, each
    trial fine-tunes a FRESH base model before scoring; otherwise the objective
    is a zero-shot benchmark (load -> encode -> score). ``cache_dir`` enables
    encode-once/score-many: zero-shot embeddings are cached to .npz so a warm
    re-run skips the encode but still reports the original encode latency.
    ``prompt`` is prepended to every text before encoding (e5-family models).

    ``cv_folds >= 2`` switches from the single full-set score to group-aware
    K-fold cross-validation. The group is the barcode: ``pos_groups``/``neg_groups``
    carry one label per positive/negative pair (negatives labeled by the
    barcode of their first row — see group_kfold), and ``finetune_groups``
    carries one label per finetune example (required only for the fine-tune
    path). Each fold is scored on its held-out pairs alone — zero-shot: encode
    once, score the fold's pairs; fine-tune: fit on the OTHER folds'
    finetune_examples, then evaluate — and the objective returns the fold-AUC
    mean while reporting ``auc_mean``/``auc_std`` plus per-fold values (and the
    same mean/std for every metric) in ``broadway_metrics``. ``cv_seed`` seeds
    the fold assignment. With ``cv_folds=None`` the original byte-for-byte
    single-score path is used.
    """
    from sentence_transformers import SentenceTransformer

    cache_path = _embedding_cache_path(
        cache_dir, model_id, payload, max_seq_length, batch_size, prompt
    )

    folds: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None
    if cv_folds is not None:
        if cv_folds < 2:
            raise ValueError(f"cv_folds must be >= 2 or None, got {cv_folds}")
        if pos_groups is None or neg_groups is None:
            raise ValueError(
                "cv_folds requires pos_groups and neg_groups (one group label per pair)"
            )
        if finetune_examples is not None and finetune_groups is None:
            raise ValueError(
                "cv_folds fine-tune requires finetune_groups (one group label per example)"
            )
        pos_g = np.asarray(pos_groups)
        neg_g = np.asarray(neg_groups)
        if pos_g.ndim != 1 or len(pos_g) != len(pos_pairs):
            raise ValueError("pos_groups must have one label per positive pair")
        if neg_g.ndim != 1 or len(neg_g) != len(neg_pairs):
            raise ValueError("neg_groups must have one label per negative pair")
        ft_g = np.asarray(finetune_groups) if finetune_groups is not None else np.empty(0, dtype=int)
        if ft_g.ndim != 1 or (finetune_groups is not None and len(ft_g) != len(finetune_examples)):
            raise ValueError("finetune_groups must have one label per finetune example")
        combined = np.concatenate([pos_g, neg_g, ft_g]).tolist()
        masks = group_kfold(combined, cv_folds, seed=cv_seed)
        n_pos, n_neg = len(pos_g), len(neg_g)
        folds = [
            (masks[f][:n_pos], masks[f][n_pos:n_pos + n_neg], masks[f][n_pos + n_neg:])
            for f in range(cv_folds)
        ]

    def objective(params: dict[str, float | int], trial=None) -> float:
        if folds is None:
            model = SentenceTransformer(model_id, device=device, model_kwargs={"torch_dtype": torch.float32})
            if finetune_examples is not None and _has_finetune_params(params):
                _finetune(model, params, finetune_examples, max_seq_length, loss)
                emb, encode_s = _encode_payload(model, payload, batch_size, None, prompt)
            else:
                model.max_seq_length = max_seq_length
                emb, encode_s = _encode_payload(model, payload, batch_size, cache_path, prompt)
            metrics = entity_resolution_metrics(_cosine(emb, pos_pairs), _cosine(emb, neg_pairs))
            metrics["encode_s"] = round(encode_s, 3)
            if trial is not None:
                trial.set_user_attr("broadway_metrics", metrics)
            return metrics["auc"]

        if finetune_examples is not None and _has_finetune_params(params):
            fold_metrics: list[dict[str, float]] = []
            for pos_mask, neg_mask, ft_mask in folds:
                train_mask = ~ft_mask
                train_examples = [
                    example for example, keep in zip(finetune_examples, train_mask) if keep
                ]
                model = SentenceTransformer(model_id, device=device, model_kwargs={"torch_dtype": torch.float32})
                _finetune(model, params, train_examples, max_seq_length, loss)
                emb, encode_s = _encode_payload(model, payload, batch_size, None, prompt)
                metrics = entity_resolution_metrics(
                    _cosine(emb, pos_pairs[pos_mask]), _cosine(emb, neg_pairs[neg_mask])
                )
                metrics["encode_s"] = round(encode_s, 3)
                fold_metrics.append(metrics)
        else:
            model = SentenceTransformer(model_id, device=device, model_kwargs={"torch_dtype": torch.float32})
            model.max_seq_length = max_seq_length
            emb, encode_s = _encode_payload(model, payload, batch_size, cache_path, prompt)
            fold_metrics = []
            for pos_mask, neg_mask, _ft_mask in folds:
                metrics = entity_resolution_metrics(
                    _cosine(emb, pos_pairs[pos_mask]), _cosine(emb, neg_pairs[neg_mask])
                )
                metrics["encode_s"] = round(encode_s, 3)
                fold_metrics.append(metrics)
        aggregated = _aggregate_fold_metrics(fold_metrics)
        if trial is not None:
            trial.set_user_attr("broadway_metrics", aggregated)
        return aggregated["auc_mean"]

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
    finetune_examples=None,
    device: str = "cpu",
    batch_size: int = 256,
    max_seq_length: int = 128,
    cache_dir: str | None = None,
    prompts: dict[str, str] | None = None,
    loss: str = "mnrl",
    cv_folds: int | None = None,
    pos_groups: Sequence[int] | None = None,
    neg_groups: Sequence[int] | None = None,
    finetune_groups: Sequence[int] | None = None,
    cv_seed: int | None = None,
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
    ``cv_folds``/``pos_groups``/``neg_groups``/``finetune_groups``/``cv_seed``
    are forwarded to make_objective (see its docstring for the group-aware CV
    contract); they default to None = the original full-set scoring.
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
            finetune_examples=finetune_examples,
            device=device,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            cache_dir=cache_dir,
            prompt=prompts.get(spec.name),
            loss=loss,
            cv_folds=cv_folds,
            pos_groups=pos_groups,
            neg_groups=neg_groups,
            finetune_groups=finetune_groups,
            cv_seed=cv_seed,
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
    finetune_examples=None,
    pos_groups: Sequence[int] | None = None,
    neg_groups: Sequence[int] | None = None,
    finetune_groups: Sequence[int] | None = None,
    mlflow_tracking: bool = False,
    mlflow_tags: dict[str, str] | None = None,
) -> dict:
    """Run the NLP HPO bandit from a typed config (data-agnostic entry point).

    Parallel to broadway.training.module.run: the typed NLPConfig carries the
    model zoo + bandit spec + encode knobs, and the caller supplies the payload
    and ground-truth pair indices, so this module stays dataset-agnostic. Returns
    the plain-data result ({models, best_model, best_params, best_value,
    metrics, and a failed map when some model raised}).

    When ``cfg.cv_folds`` is set, ``pos_groups``/``neg_groups`` carry one
    barcode label per pair (and ``finetune_groups`` one per finetune example)
    and the CV fold split is seeded with ``pair_seed`` (falling back to ``seed``)
    so it is reproducible and independent of the TPE sampler seed.
    """
    return run_nlp_hpo(
        cfg.model_zoo,
        cfg.hpo,
        payload,
        pos_pairs,
        neg_pairs,
        seed=cfg.seed,
        finetune_examples=finetune_examples,
        device=cfg.device,
        batch_size=cfg.batch_size,
        max_seq_length=cfg.max_seq_length,
        cache_dir=cfg.cache_dir,
        prompts=cfg.prompts,
        loss=getattr(cfg, "loss", "mnrl"),
        cv_folds=cfg.cv_folds,
        pos_groups=pos_groups,
        neg_groups=neg_groups,
        finetune_groups=finetune_groups,
        cv_seed=cfg.pair_seed if cfg.pair_seed is not None else cfg.seed,
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
