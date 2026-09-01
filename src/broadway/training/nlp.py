"""NLP entity-resolution HPO — adapt the optuna bandit to embedding models.

The sklearn HPO pipeline (broadway.training.hpo) treats each model as a
sklearn Pipeline built from a PipelineConfig and minimizes a regression metric.
This module adapts the SAME bandit machinery to NLP: each HPO "model" is a
bi-encoder (sentence-transformers), the objective scores title-pair entity
resolution (same-barcode positives vs cross-barcode negatives) as ROC AUC and
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

import numpy as np

from broadway.training.hpo import HPOConfig, Objective, run_hpo_bandit

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
) -> dict[str, float]:
    """Ranking metrics for a bi-encoder's pos-vs-neg score populations.

    pos_scores are true-pair (same-barcode) cosine similarities, neg_scores are
    cross-barcode non-pairs. Returns AUC (probability a random positive outscores
    a random negative) plus the recall at a fixed false-positive budget and the
    score-distribution summaries used for the threshold trade-off.
    """
    from sklearn.metrics import roc_auc_score

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        raise ValueError("entity_resolution_metrics requires non-empty pos and neg scores")
    y = np.r_[np.ones(len(pos_scores)), np.zeros(len(neg_scores))]
    auc = float(roc_auc_score(y, np.r_[pos_scores, neg_scores]))
    thr = float(np.quantile(neg_scores, 1 - fpr))
    return {
        "auc": round(auc, 4),
        f"recall_at_{int(fpr * 100)}pct_fpr": round(float((pos_scores >= thr).mean()), 4),
        "pos_median": round(float(np.median(pos_scores)), 4),
        "neg_p90": round(float(np.quantile(neg_scores, 0.9)), 4),
    }


def _has_finetune_params(params: dict[str, float | int]) -> bool:
    return any(name in params for name in _FINETUNE_PARAMS)


def _embedding_cache_path(
    cache_dir: str | None,
    model_id: str,
    payload: list[str],
    max_seq_length: int,
    batch_size: int,
) -> Path | None:
    """Deterministic .npz cache path for a model+payload (None disables cache)."""
    if cache_dir is None:
        return None
    digest = hashlib.sha1("\x1f".join(payload).encode("utf-8")).hexdigest()[:12]
    key = f"{model_id.replace('/', '_')}_{max_seq_length}_{batch_size}_{digest}"
    return Path(cache_dir) / f"emb_{key}.npz"


def _encode_payload(model, payload: list[str], batch_size: int, cache_path: Path | None):
    """Encode the corpus once; reuse a cached .npz (emb + encode_s) on re-run.

    Encode-once, score-many: the cache holds the raw embedding matrix and the
    original encode wall-time so a warm re-run reports the SAME latency instead
    of a misleading 0. Fine-tuned models never cache (params vary per trial).
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
    )
    encode_s = time.perf_counter() - t0
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, emb=emb, encode_s=encode_s)
    return emb, encode_s


def _finetune(model, params: dict[str, float | int], examples, max_seq_length: int) -> None:
    """Contrastive fine-tune a fresh base model in place (no disk save).

    The caller owns the model object; fit() mutates it, so encoding afterwards
    uses the fine-tuned weights. ``examples`` must be POSITIVE-ONLY
    ``InputExample`` pairs (same product): MultipleNegativesRankingLoss builds
    the negatives in-batch, so a label-bearing example would be silently
    misused as a positive. ``learning_rate`` is passed through optimizer_params
    (sentence-transformers has no top-level lr knob); when absent the library
    default is used, so the argument is omitted rather than passed as None.
    """
    from sentence_transformers.sentence_transformer import losses
    from torch.utils.data import DataLoader

    model.max_seq_length = max_seq_length
    epochs = int(params.get("epochs", 1))
    batch_size = int(params.get("batch_size", 32))
    warmup_steps = int(params.get("warmup_steps", 0))
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
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
    """
    from sentence_transformers import SentenceTransformer

    cache_path = _embedding_cache_path(cache_dir, model_id, payload, max_seq_length, batch_size)

    def objective(params: dict[str, float | int], trial=None) -> float:
        model = SentenceTransformer(model_id, device=device)
        if finetune_examples is not None and _has_finetune_params(params):
            _finetune(model, params, finetune_examples, max_seq_length)
            emb, encode_s = _encode_payload(model, payload, batch_size, None)
        else:
            model.max_seq_length = max_seq_length
            emb, encode_s = _encode_payload(model, payload, batch_size, cache_path)
        metrics = entity_resolution_metrics(_cosine(emb, pos_pairs), _cosine(emb, neg_pairs))
        metrics["encode_s"] = round(encode_s, 3)
        if trial is not None:
            trial.set_user_attr("broadway_metrics", metrics)
        return metrics["auc"]

    return objective


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
    mlflow_tracking: bool = False,
    mlflow_tags: dict[str, str] | None = None,
) -> dict:
    """Run the bandit HPO over the embedding model zoo (direction: maximize).

    model_zoo maps the HPO spec's short names to HF repo ids; every
    hpo_cfg.models entry must resolve through it (loud failure otherwise). The
    bandit, seeded TPE, mlflow callback, and RDB contract are reused from
    run_hpo_bandit unchanged. The returned dict adds a "metrics" map (name ->
    the best trial's broadway_metrics: auc, recall@5%FPR, encode latency, and
    score summaries) so callers can build the benchmark table from plain data.
    """
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
        )
        for spec in hpo_cfg.models
    }
    result = run_hpo_bandit(
        objectives, hpo_cfg, seed, mlflow_tracking, mlflow_tags, return_studies=True
    )
    studies = result.pop("studies")
    result["metrics"] = {
        name: {
            key: float(value)
            for key, value in study.best_trial.user_attrs.get("broadway_metrics", {}).items()
        }
        for name, study in studies.items()
        if name in result["models"]
    }
    return result
