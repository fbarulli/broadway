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
from pathlib import Path
from typing import Any

import numpy as np

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

    model = SentenceTransformer(model_id, device=device)
    model.max_seq_length = max_seq_length
    cache_path = _embedding_cache_path(
        cache_dir, model_id, payload, max_seq_length, batch_size, prompt
    )
    return _encode_payload(model, payload, batch_size, cache_path, prompt)


def _finetune(model, params: dict[str, float | int], examples, max_seq_length: int, loss: str = "mnrl") -> None:
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
    """
    from sentence_transformers import SentenceTransformer

    cache_path = _embedding_cache_path(
        cache_dir, model_id, payload, max_seq_length, batch_size, prompt
    )

    def objective(params: dict[str, float | int], trial=None) -> float:
        model = SentenceTransformer(model_id, device=device)
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
            finetune_examples=finetune_examples,
            device=device,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            cache_dir=cache_dir,
            prompt=prompts.get(spec.name),
            loss=loss,
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
        finetune_examples=finetune_examples,
        device=cfg.device,
        batch_size=cfg.batch_size,
        max_seq_length=cfg.max_seq_length,
        cache_dir=cfg.cache_dir,
        prompts=cfg.prompts,
        loss=getattr(cfg, "loss", "mnrl"),
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
