"""NLP HPO tests — entity-resolution metrics + the optuna bandit over embeddings.

Hermetic: the SentenceTransformer import inside the objective is stubbed via a
fake module in sys.modules, so no torch weights are loaded. The fake encoder
places each payload index on a unit circle, so cosine similarity is a pure
function of index distance; a per-model noise level then controls how well the
pos/neg pair populations separate — giving a deterministic AUC per model that
exercises the maximize direction and metric extraction.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

from broadway.config.schema import HPOConfig, ModelHPOSpec, NLPConfig
from broadway.training import nlp
from broadway.training.nlp import entity_resolution_metrics, load_pairs_csv


class _CaptureTrial:
    """Minimal optuna-trial stand-in that records the broadway_metrics user attr."""

    def __init__(self) -> None:
        self.attrs: dict = {}

    def set_user_attr(self, key, value) -> None:
        self.attrs[key] = value


def _circle_embeddings(n: int) -> np.ndarray:
    """Replicate the fake encoder's noise-free embeddings (index on the unit circle)."""
    angle = 2.0 * np.pi * np.arange(n) / n
    return np.column_stack([np.cos(angle), np.sin(angle)])


def _install_fake_sentence_transformers(monkeypatch: pytest.MonkeyPatch, noise_map: dict) -> None:
    """Stub the sentence_transformers module with a deterministic fake encoder."""
    st = types.ModuleType("sentence_transformers")

    class _FakeModel:
        def __init__(self, model_id: str, device: str = "cpu", model_kwargs=None) -> None:
            self.model_id = model_id

        def encode(self, payload, **kwargs):
            del kwargs
            n = len(payload)
            angle = 2.0 * np.pi * np.arange(n) / n
            rng = np.random.default_rng(0)
            emb = np.column_stack([np.cos(angle), np.sin(angle)])
            emb = emb + noise_map[self.model_id] * rng.normal(size=(n, 2))
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            return emb

        def fit(self, **kwargs) -> None:  # fine-tune branch no-op
            del kwargs

    st.SentenceTransformer = _FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)


def test_entity_resolution_metrics_perfect_separation() -> None:
    pos = np.linspace(0.6, 1.0, 100)
    neg = np.linspace(0.0, 0.2, 100)
    m = entity_resolution_metrics(pos, neg)
    assert m["auc"] == 1.0
    assert m["average_precision"] == 1.0
    assert m["recall_at_5pct_fpr"] == 1.0
    assert m["precision_at_90pct_recall"] == 1.0
    assert m["f1_at_5pct_fpr"] > 0.9  # 5% FPR budget admits ~5% false positives
    assert 0.5 < m["pos_median"] < 1.0
    assert m["neg_p90"] < 0.3


def test_entity_resolution_metrics_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        entity_resolution_metrics(np.array([]), np.array([1.0]))
    with pytest.raises(ValueError, match="non-empty"):
        entity_resolution_metrics(np.array([1.0]), np.array([]))


def test_precision_at_recall_breakdown_nonempty() -> None:
    """The breakdown returns precision/TP/FP/threshold/recall; the wrapper matches."""
    from broadway.training.nlp import precision_at_recall, precision_at_recall_breakdown

    pos = np.array([0.60, 0.70, 0.80, 0.90])
    neg = np.array([0.10, 0.20])
    threshold = float(np.quantile(pos, 0.10))
    tp = int((pos >= threshold).sum())
    fp = int((neg >= threshold).sum())
    precision, tp_out, fp_out, thr_out, recall_out = precision_at_recall_breakdown(pos, neg, 0.90)
    assert precision == pytest.approx(tp / (tp + fp))
    assert tp_out == tp
    assert fp_out == fp
    assert thr_out == pytest.approx(threshold)
    assert recall_out == pytest.approx(tp / len(pos))
    # wrapper parity: precision_at_recall delegates to the breakdown
    assert precision_at_recall(pos, neg, 0.90) == precision


def test_precision_at_recall_breakdown_empty_populations_nan() -> None:
    """An empty positive or negative population yields (nan, 0, 0, nan, nan)."""
    from broadway.training.nlp import precision_at_recall_breakdown

    pos = np.array([0.60, 0.70, 0.80])
    neg = np.array([0.10, 0.20])
    for empty_pos, empty_neg in ((np.array([]), neg), (pos, np.array([]))):
        precision, tp, fp, thr, recall = precision_at_recall_breakdown(empty_pos, empty_neg, 0.90)
        assert np.isnan(precision)
        assert tp == 0 and fp == 0
        assert np.isnan(thr)
        assert np.isnan(recall)


def test_entity_resolution_metrics_new_audit_keys() -> None:
    """The six auditable TP/FP/threshold keys match the breakdown + 5% FPR formula."""
    from broadway.training.nlp import precision_at_recall_breakdown

    pos = np.array([0.60, 0.70, 0.80, 0.90])
    neg = np.array([0.10, 0.20, 0.30, 0.40])
    m = entity_resolution_metrics(pos, neg)
    prec90, tp90, fp90, thr90, _ = precision_at_recall_breakdown(pos, neg, 0.90)
    assert m["precision_at_90pct_recall"] == pytest.approx(round(float(prec90), 4))
    assert m["tp_at_90pct_recall"] == pytest.approx(float(tp90))
    assert m["fp_at_90pct_recall"] == pytest.approx(float(fp90))
    assert m["threshold_at_90pct_recall"] == pytest.approx(round(thr90, 4))
    thr = float(np.quantile(neg, 0.95))
    tp_op = float((pos >= thr).sum())
    fp_op = float((neg >= thr).sum())
    assert m["tp_at_5pct_fpr"] == pytest.approx(float(tp_op))
    assert m["fp_at_5pct_fpr"] == pytest.approx(float(fp_op))
    assert m["threshold_at_5pct_fpr"] == pytest.approx(round(thr, 4))


def test_make_objective_returns_auc_and_attaches_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sentence_transformers(monkeypatch, {"m": 0.0})
    payload = [f"s{i}" for i in range(20)]
    pos = np.array([[i, i + 1] for i in range(0, 18, 2)])       # adjacent -> similar
    neg = np.array([[i, i + 10] for i in range(10)])            # opposite -> dissimilar
    objective = nlp.make_objective("m", payload, pos, neg)

    class _Trial:
        def __init__(self) -> None:
            self.attrs: dict = {}

        def set_user_attr(self, key, value) -> None:
            self.attrs[key] = value

    trial = _Trial()
    value = objective({}, trial)
    assert isinstance(value, float) and 0.0 <= value <= 1.0
    assert set(trial.attrs["broadway_metrics"]) == {
        "auc", "average_precision", "recall_at_5pct_fpr",
        "precision_at_90pct_recall", "f1_at_5pct_fpr",
        "tp_at_90pct_recall", "fp_at_90pct_recall", "threshold_at_90pct_recall",
        "tp_at_5pct_fpr", "fp_at_5pct_fpr", "threshold_at_5pct_fpr",
        "pos_median", "neg_p90", "encode_s",
    }


def test_run_nlp_hpo_maximize_selects_best_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sentence_transformers(monkeypatch, {"good": 0.0, "bad": 5.0})
    payload = [f"s{i}" for i in range(20)]
    pos = np.array([[i, i + 1] for i in range(0, 18, 2)])
    neg = np.array([[i, i + 10] for i in range(10)])
    hpo_cfg = HPOConfig(
        engine="optuna",
        direction="maximize",
        target_metric="auc",
        total_trials=2,
        initial_trials_per_model=1,
        top_k=2,
        models=[
            ModelHPOSpec(name="good", search_space={}),
            ModelHPOSpec(name="bad", search_space={}),
        ],
    )
    result = nlp.run_nlp_hpo(
        {"good": "good", "bad": "bad"}, hpo_cfg, payload, pos, neg, seed=42
    )
    assert set(result["models"]) == {"good", "bad"}
    assert result["best_model"] == "good"
    assert result["best_value"] == pytest.approx(result["metrics"]["good"]["auc"])
    assert result["metrics"]["good"]["auc"] > result["metrics"]["bad"]["auc"]
    # encode_s is a wall-clock reading; the fake encoder is instant, so only
    # assert presence + non-negativity (real models yield positive values).
    assert all("encode_s" in m and m["encode_s"] >= 0 for m in result["metrics"].values())


def test_run_nlp_typed_config_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """The data-agnostic entry point (run_nlp + NLPConfig) drives the bandit."""
    _install_fake_sentence_transformers(monkeypatch, {"good": 0.0, "bad": 5.0})
    payload = [f"s{i}" for i in range(20)]
    pos = np.array([[i, i + 1] for i in range(0, 18, 2)])
    neg = np.array([[i, i + 10] for i in range(10)])
    cfg = NLPConfig(
        seed=42,
        model_zoo={"good": "good", "bad": "bad"},
        hpo=HPOConfig(
            engine="optuna",
            direction="maximize",
            target_metric="auc",
            total_trials=2,
            initial_trials_per_model=1,
            top_k=2,
            models=[
                ModelHPOSpec(name="good", search_space={}),
                ModelHPOSpec(name="bad", search_space={}),
            ],
        ),
    )
    result = nlp.run_nlp(cfg, payload, pos, neg)
    assert result["best_model"] == "good"
    assert set(result["models"]) == {"good", "bad"}


def test_make_objective_passes_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """A per-model prompt (e.g. e5 'query: ') reaches sentence-transformers."""
    captured: dict = {}
    st = types.ModuleType("sentence_transformers")

    class _PromptModel:
        def __init__(self, model_id: str, device: str = "cpu", model_kwargs=None) -> None:
            self.model_id = model_id

        def encode(self, payload, **kwargs):
            captured["prompt"] = kwargs.get("prompt")
            emb = np.random.default_rng(0).normal(size=(len(payload), 2))
            return emb / np.linalg.norm(emb, axis=1, keepdims=True)

    st.SentenceTransformer = _PromptModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)

    payload = [f"s{i}" for i in range(8)]
    pos = np.array([[0, 1], [2, 3]])
    neg = np.array([[4, 5], [6, 7]])
    objective = nlp.make_objective("m", payload, pos, neg, prompt="query: ")
    objective({})
    assert captured["prompt"] == "query: "


def _install_fake_finetune_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub sentence_transformer.losses AND the whole torch package chain.

    Stubbing torch.utils.data alone still imports the real ``torch`` package
    (the parent), so we also stub ``torch`` and ``torch.utils`` for a fully
    hermetic unit (no torch/ST import, only stdlib + numpy).
    """
    losses_mod = types.ModuleType("losses")

    class _Loss:
        def __init__(self, model) -> None:
            self.model = model

    losses_mod.MultipleNegativesRankingLoss = _Loss
    st_mod = types.ModuleType("sentence_transformer")
    st_mod.losses = losses_mod
    monkeypatch.setitem(sys.modules, "sentence_transformers.sentence_transformer", st_mod)

    class _Loader:
        def __init__(self, examples, shuffle=True, batch_size=32) -> None:
            self.examples, self.shuffle, self.batch_size = examples, shuffle, batch_size

    torch_data = types.ModuleType("torch.utils.data")
    torch_data.DataLoader = _Loader
    torch_utils = types.ModuleType("torch.utils")
    torch_utils.data = torch_data
    torch_mod = types.ModuleType("torch")
    torch_mod.utils = torch_utils
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", torch_data)


def test_make_objective_finetune_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fine-tune branch (finetune_examples + fine-tune params) still scores."""
    _install_fake_sentence_transformers(monkeypatch, {"m": 0.0})
    _install_fake_finetune_modules(monkeypatch)
    payload = [f"s{i}" for i in range(20)]
    pos = np.array([[i, i + 1] for i in range(0, 18, 2)])
    neg = np.array([[i, i + 10] for i in range(10)])
    objective = nlp.make_objective("m", payload, pos, neg, finetune_examples=["a", "b"])
    value = objective({"epochs": 1, "learning_rate": 1e-4, "batch_size": 8, "warmup_steps": 0})
    assert isinstance(value, float) and 0.0 <= value <= 1.0


def test_group_kfold_no_group_spans_two_folds() -> None:
    """The splitter puts each group (barcode) in exactly one fold, ~balanced."""
    from broadway.training.nlp import group_kfold

    group_ids = [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 4, 4]
    k = 3
    masks = group_kfold(group_ids, k, seed=42)
    assert len(masks) == k
    # every pair belongs to exactly one fold
    membership = np.column_stack([m.astype(int) for m in masks])
    assert np.all(membership.sum(axis=1) == 1)
    # no group is split across folds: each group's pairs sit in a single fold
    for group in sorted(set(group_ids)):
        idx = np.array([i for i, g in enumerate(group_ids) if g == group])
        folds_with_group = [m for m in masks if m[idx].any()]
        assert len(folds_with_group) == 1
        assert folds_with_group[0][idx].all()
    # ~balanced: greedy largest-first on these counts yields 4/4/4
    sizes = [int(m.sum()) for m in masks]
    assert max(sizes) - min(sizes) <= 1
    assert all(size > 0 for size in sizes)
    # deterministic for a fixed seed
    assert all(
        np.array_equal(a, b)
        for a, b in zip(masks, group_kfold(group_ids, k, seed=42), strict=True)
    )


def test_make_objective_cv_returns_mean_std(monkeypatch: pytest.MonkeyPatch) -> None:
    """CV objective scores each held-out fold and reports mean/std of the AUCs."""
    from broadway.training.nlp import _cosine, group_kfold

    _install_fake_sentence_transformers(monkeypatch, {"m": 0.0})
    n = 24
    payload = [f"s{i}" for i in range(n)]
    pos_rows: list[list[int]] = []
    neg_rows: list[list[int]] = []
    pos_groups: list[int] = []
    neg_groups: list[int] = []
    for g in range(6):
        base = g * 4
        pos_rows += [[base, base + 1], [base + 2, base + 3]]
        pos_groups += [g, g]
        neg_rows.append([base, (base + 8) % n])
        neg_groups.append(g)
    pos_pairs = np.array(pos_rows)
    neg_pairs = np.array(neg_rows)

    k = 3
    objective = nlp.make_objective(
        "m", payload, pos_pairs, neg_pairs,
        cv_folds=k, pos_groups=pos_groups, neg_groups=neg_groups, cv_seed=7,
    )
    trial = _CaptureTrial()
    value = objective({}, trial)
    metrics = trial.attrs["broadway_metrics"]

    # expected per-fold AUCs: score the fold's held-out pairs with the same
    # noise-free circle embeddings the fake encoder produces.
    combined = np.concatenate([np.asarray(pos_groups), np.asarray(neg_groups)]).tolist()
    masks = group_kfold(combined, k, seed=7)
    emb = _circle_embeddings(n)
    p, nneg = len(pos_groups), len(neg_groups)
    fold_aucs = []
    for f in range(k):
        fold_metrics = entity_resolution_metrics(
            _cosine(emb, pos_pairs[masks[f][:p]]), _cosine(emb, neg_pairs[masks[f][p:p + nneg]])
        )
        fold_aucs.append(fold_metrics["auc"])

    expected_mean = round(float(np.mean(fold_aucs)), 4)
    expected_std = round(float(np.std(fold_aucs)), 4)
    assert value == pytest.approx(expected_mean)
    assert metrics["auc_mean"] == pytest.approx(expected_mean)
    assert metrics["auc_std"] == pytest.approx(expected_std)
    for f in range(k):
        assert metrics[f"fold_{f}_auc"] == pytest.approx(round(fold_aucs[f], 4))
    # mean/std (and per-fold) are reported for every fold metric, not just auc
    assert "average_precision_mean" in metrics
    assert "average_precision_std" in metrics
    assert "fold_0_average_precision" in metrics


def test_make_objective_cv_folds_none_preserves_single_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cv_folds=None keeps the original full-set single-score contract."""
    from broadway.training.nlp import _cosine

    _install_fake_sentence_transformers(monkeypatch, {"m": 0.0})
    payload = [f"s{i}" for i in range(20)]
    pos = np.array([[i, i + 1] for i in range(0, 18, 2)])
    neg = np.array([[i, i + 10] for i in range(10)])
    objective = nlp.make_objective("m", payload, pos, neg, cv_folds=None)
    trial = _CaptureTrial()
    value = objective({}, trial)
    metrics = trial.attrs["broadway_metrics"]

    expected = entity_resolution_metrics(_cosine(_circle_embeddings(20), pos), _cosine(_circle_embeddings(20), neg))
    assert value == pytest.approx(expected["auc"])
    assert metrics["auc"] == pytest.approx(expected["auc"])
    assert set(metrics) == {
        "auc", "average_precision", "recall_at_5pct_fpr",
        "precision_at_90pct_recall", "f1_at_5pct_fpr",
        "tp_at_90pct_recall", "fp_at_90pct_recall", "threshold_at_90pct_recall",
        "tp_at_5pct_fpr", "fp_at_5pct_fpr", "threshold_at_5pct_fpr",
        "pos_median", "neg_p90", "encode_s",
    }
    assert "auc_mean" not in metrics and "auc_std" not in metrics


def test_make_objective_cv_finetune_fits_on_other_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fine-tune CV fits each fold on every example EXCEPT the held-out group's."""
    from broadway.training.nlp import group_kfold

    _install_fake_sentence_transformers(monkeypatch, {"m": 0.0})
    _install_fake_finetune_modules(monkeypatch)

    n = 24
    payload = [f"s{i}" for i in range(n)]
    pos_rows: list[list[int]] = []
    neg_rows: list[list[int]] = []
    pos_groups: list[int] = []
    neg_groups: list[int] = []
    for g in range(6):
        base = g * 4
        pos_rows += [[base, base + 1], [base + 2, base + 3]]
        pos_groups += [g, g]
        neg_rows.append([base, (base + 8) % n])
        neg_groups.append(g)
    pos_pairs = np.array(pos_rows)
    neg_pairs = np.array(neg_rows)

    finetune_examples = [f"ex_{g}" for g in range(6)]
    finetune_groups = list(range(6))

    # record the example list each fold's fit() receives
    loader_calls: list[list[str]] = []

    class _RecLoader:
        def __init__(self, examples, shuffle=True, batch_size=32) -> None:
            loader_calls.append(list(examples))

    sys.modules["torch.utils.data"].DataLoader = _RecLoader

    k = 3
    objective = nlp.make_objective(
        "m", payload, pos_pairs, neg_pairs,
        finetune_examples=finetune_examples,
        finetune_groups=finetune_groups,
        cv_folds=k, pos_groups=pos_groups, neg_groups=neg_groups, cv_seed=7,
    )
    trial = _CaptureTrial()
    value = objective({"epochs": 1, "learning_rate": 1e-4, "batch_size": 8, "warmup_steps": 0}, trial)
    metrics = trial.attrs["broadway_metrics"]

    assert value == pytest.approx(metrics["auc_mean"])
    assert "auc_std" in metrics and "fold_0_auc" in metrics

    combined = np.concatenate([
        np.asarray(pos_groups), np.asarray(neg_groups), np.asarray(finetune_groups),
    ]).tolist()
    masks = group_kfold(combined, k, seed=7)
    n_ft = len(finetune_groups)
    assert len(loader_calls) == k
    for f in range(k):
        ft_mask = masks[f][-n_ft:]
        expected_train = [ex for i, ex in enumerate(finetune_examples) if not ft_mask[i]]
        held_out = {ex for i, ex in enumerate(finetune_examples) if ft_mask[i]}
        assert set(loader_calls[f]) == set(expected_train)
        assert held_out.isdisjoint(loader_calls[f])


def test_finetune_calls_model_fit_with_params(monkeypatch: pytest.MonkeyPatch) -> None:
    """_finetune maps search-space params onto sentence-transformers fit() and
    passes learning_rate through optimizer_params (no top-level lr knob)."""
    from broadway.training.nlp import _finetune, _has_finetune_params

    captured: dict = {}

    class _Loss:
        def __init__(self, model) -> None:
            self.model = model

    losses_mod = types.ModuleType("losses")
    losses_mod.MultipleNegativesRankingLoss = _Loss
    st_mod = types.ModuleType("sentence_transformer")
    st_mod.losses = losses_mod
    monkeypatch.setitem(sys.modules, "sentence_transformers.sentence_transformer", st_mod)

    class _Loader:
        def __init__(self, examples, shuffle=True, batch_size=32) -> None:
            self.examples, self.shuffle, self.batch_size = examples, shuffle, batch_size

    torch_data = types.ModuleType("torch.utils.data")
    torch_data.DataLoader = _Loader
    monkeypatch.setitem(sys.modules, "torch.utils.data", torch_data)

    class _Model:
        def __init__(self) -> None:
            self.max_seq_length = None

        def fit(self, **kwargs) -> None:
            captured.update(kwargs)

    model = _Model()
    _finetune(model, {"epochs": 2, "learning_rate": 1e-4, "batch_size": 16, "warmup_steps": 10},
              examples=["a", "b"], max_seq_length=128)
    assert model.max_seq_length == 128
    assert captured["epochs"] == 2
    assert captured["warmup_steps"] == 10
    assert captured["optimizer_params"] == {"lr": 1e-4}
    assert captured["output_path"] is None
    assert _has_finetune_params({"epochs": 1}) is True
    assert _has_finetune_params({}) is False


def test_finetune_without_learning_rate_omits_optimizer_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a search space without learning_rate must NOT pass
    optimizer_params=None (sentence-transformers would do **None -> TypeError)."""
    from broadway.training.nlp import _finetune

    _install_fake_finetune_modules(monkeypatch)
    captured: dict = {}

    class _Model:
        def __init__(self) -> None:
            self.max_seq_length = None

        def fit(self, **kwargs) -> None:
            captured.update(kwargs)

    model = _Model()
    _finetune(model, {"epochs": 1, "batch_size": 8}, examples=["a"], max_seq_length=128)
    assert "optimizer_params" not in captured
    assert captured["epochs"] == 1


def test_encode_payload_cache_roundtrip(tmp_path) -> None:
    """Encode-once/score-many: a warm re-run reads the .npz and keeps encode_s."""
    from broadway.training.nlp import _embedding_cache_path, _encode_payload

    class _Encoder:
        def encode(self, payload, **kwargs):
            del kwargs
            emb = np.random.default_rng(0).normal(size=(len(payload), 3))
            return emb / np.linalg.norm(emb, axis=1, keepdims=True)

    payload = [f"s{i}" for i in range(8)]
    cache_path = _embedding_cache_path(str(tmp_path), "model/x", payload, 128, 32)
    assert cache_path is not None and not cache_path.exists()

    model = _Encoder()
    emb1, s1 = _encode_payload(model, payload, 32, cache_path)
    assert cache_path.exists()
    emb2, s2 = _encode_payload(model, payload, 32, cache_path)
    assert s2 == s1  # cached latency preserved, not reset to 0
    assert np.allclose(emb1, emb2)
    assert _embedding_cache_path(None, "m", payload, 128, 32) is None


def test_encode_corpus_returns_embeddings_and_reuses_cache(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """encode_corpus is the reusable non-HPO encode entry point."""
    from broadway.training.nlp import encode_corpus

    st = types.ModuleType("sentence_transformers")

    class _Model:
        def __init__(self, model_id: str, device: str = "cpu", model_kwargs=None) -> None:
            self.model_id = model_id
            self.max_seq_length = None

        def encode(self, payload, **kwargs):
            del kwargs
            emb = np.random.default_rng(0).normal(size=(len(payload), 3))
            return emb / np.linalg.norm(emb, axis=1, keepdims=True)

    st.SentenceTransformer = _Model
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)

    payload = [f"s{i}" for i in range(8)]
    emb1, s1 = encode_corpus("m", payload, cache_dir=str(tmp_path), batch_size=32)
    assert emb1.shape == (8, 3)
    assert s1 >= 0
    emb2, s2 = encode_corpus("m", payload, cache_dir=str(tmp_path), batch_size=32)
    assert s2 == s1  # cached latency preserved, not reset to 0
    assert np.allclose(emb1, emb2)


def test_run_nlp_hpo_unknown_model_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sentence_transformers(monkeypatch, {})
    hpo_cfg = HPOConfig(
        engine="optuna",
        direction="maximize",
        target_metric="auc",
        total_trials=1,
        initial_trials_per_model=1,
        top_k=1,
        models=[ModelHPOSpec(name="missing", search_space={})],
    )
    payload = ["a", "b"]
    pos = np.array([[0, 1]])
    neg = np.array([[0, 1]])
    with pytest.raises(ValueError, match="missing from model_zoo"):
        nlp.run_nlp_hpo({}, hpo_cfg, payload, pos, neg, seed=1)


def test_run_nlp_end_to_end_generic_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Config-driven smoke: configs/nlp.yaml + demo/nlp_pairs.csv -> run_nlp.

    Mirrors the tabular configs gate (load a data-agnostic config against a
    generic fixture) but as a hermetic pytest: the encoder is stubbed so no
    torch weights or network are needed. Asserts the pipeline runs end-to-end
    and returns a result for every model in the zoo.
    """
    root = Path(__file__).resolve().parents[1]
    cfg = NLPConfig(**yaml.safe_load((root / "configs" / "nlp.yaml").read_text(encoding="utf-8")))
    payload, pos, neg = load_pairs_csv(str(root / "demo" / "nlp_pairs.csv"))
    assert payload and len(pos) > 0 and len(neg) > 0

    _install_fake_sentence_transformers(monkeypatch, {repo: 0.0 for repo in cfg.model_zoo.values()})
    result = nlp.run_nlp(cfg, payload, pos, neg)
    assert set(result["models"]) == set(cfg.model_zoo)
    assert result["best_model"] in cfg.model_zoo
    assert set(result["metrics"]) == set(cfg.model_zoo)
    assert all("auc" in m and "encode_s" in m for m in result["metrics"].values())


def test_precision_at_recall_empty_negatives_returns_nan() -> None:
    """An empty negative population must yield NaN, not a spuriously-perfect 1.0."""
    from broadway.training.nlp import precision_at_recall

    assert np.isnan(precision_at_recall(np.array([0.6, 0.7, 0.8]), np.array([])))


def test_precision_at_recall_empty_positives_returns_nan() -> None:
    """An empty positive population has no threshold to define -> NaN."""
    from broadway.training.nlp import precision_at_recall

    assert np.isnan(precision_at_recall(np.array([]), np.array([0.1, 0.2])))


def test_log_nlp_eval_noop_without_tracking_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    """A None tracking URI short-circuits before the lazy mlflow_utils import."""
    calls: list = []
    mlflow_utils = types.ModuleType("broadway.training.mlflow_utils")
    mlflow_utils.setup_mlflow = lambda *a, **k: calls.append("setup")
    mlflow_utils.log_params = lambda *a, **k: calls.append("params")
    mlflow_utils.log_metrics = lambda *a, **k: calls.append("metrics")
    monkeypatch.setitem(sys.modules, "broadway.training.mlflow_utils", mlflow_utils)

    nlp.log_nlp_eval({"auc": 1.0}, {"seed": 42}, tracking_uri=None)
    assert calls == []


def test_log_nlp_eval_happy_path_mocked_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """setup_mlflow -> log_params -> log_metrics in order; None name falls back."""
    calls: list = []
    mlflow_utils = types.ModuleType("broadway.training.mlflow_utils")
    mlflow_utils.setup_mlflow = lambda uri, name: calls.append(("setup", uri, name))
    mlflow_utils.log_params = lambda params: calls.append(("params", params))
    mlflow_utils.log_metrics = lambda metrics: calls.append(("metrics", metrics))
    monkeypatch.setitem(sys.modules, "broadway.training.mlflow_utils", mlflow_utils)

    nlp.log_nlp_eval({"auc": 0.9}, {"seed": 42}, "file:///tmp/mlruns", "test-exp")
    assert calls == [
        ("setup", "file:///tmp/mlruns", "test-exp"),
        ("params", {"seed": 42}),
        ("metrics", {"auc": 0.9}),
    ]

    calls.clear()
    nlp.log_nlp_eval({"auc": 0.9}, {"seed": 42}, "file:///tmp/mlruns", None)
    assert calls == [
        ("setup", "file:///tmp/mlruns", "nlp-eval"),
        ("params", {"seed": 42}),
        ("metrics", {"auc": 0.9}),
    ]


def _load_07b_finetune_module():
    import importlib.util

    path = (
        Path(__file__).resolve().parents[1]
        / "project" / "experiments" / "euromonitor" / "07b_finetune.py"
    )
    spec = importlib.util.spec_from_file_location("_07b_finetune_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_pos_by_country_stratifies_pairs() -> None:
    """same -> same; cross -> cross; empty-country side -> unlabeled (neither)."""
    fb = _load_07b_finetune_module()
    split = fb.split_pos_by_country
    country = np.array(["US", "US", "GB", "GB", "", "US"])
    pairs = np.array([[0, 1], [0, 2], [0, 4]])
    cross, same, unlabeled = split(pairs, country)
    assert np.array_equal(same, [True, False, False])       # [0,1] both "US"
    assert np.array_equal(cross, [False, True, False])      # [0,2] "US" vs "GB"
    assert np.array_equal(unlabeled, [False, False, True])  # [0,4] side 4 empty
    # unlabeled lands in NEITHER stratum
    assert not (same & unlabeled).any()
    assert not (cross & unlabeled).any()


def test_cosine_shared_scorer() -> None:
    """The shared _cosine scorer equals the per-pair dot product on unit vectors."""
    from broadway.training.nlp import _cosine

    emb = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    pairs = np.array([[0, 1], [0, 2]])
    scores = _cosine(emb, pairs)
    assert scores[0] == pytest.approx(0.0)
    assert scores[1] == pytest.approx(1.0)


def _load_07e_module():
    import importlib.util

    path = (
        Path(__file__).resolve().parents[1]
        / "project" / "experiments" / "euromonitor" / "07e_cross_encoder_rerank.py"
    )
    spec = importlib.util.spec_from_file_location("_07e_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hybrid_score_clamps_float32_cosine_overflow() -> None:
    """Identical float32 unit vectors round their dot product above 1.0.

    The bi-encoder cosine is the raw dot product of L2-normalized float32
    embeddings; a near-identical pair yields 1.0000002 (> 1.0). That value used
    to escape through the out-of-band branch of the hybrid ``np.where`` and trip
    ``_assert_finite_in_unit("pos hybrid")`` before either CSV was written. The
    hybrid must clamp the cosine branch into [0, 1].
    """
    from broadway.training.nlp import _cosine

    e = _load_07e_module()
    # L2-normalized float32 vector whose self-dot rounds to 1.0000002 (> 1.0).
    v = np.array([-0.7591758966445923, 0.6508856415748596], dtype=np.float32)
    emb = np.vstack([v, v, np.array([0.0, 0.0], dtype=np.float32)])
    pairs = np.array([[0, 1], [0, 2]])

    cosine = _cosine(emb, pairs)
    assert cosine[0] > 1.0  # documents the float32 overflow the clamp guards

    # Identical pair sits OUT of the re-score band, so the out-of-band branch
    # carries the overflowing cosine — the exact path that failed before the fix.
    in_band = np.array([False, True])
    ce_scores = np.array([np.nan, 0.9])
    hybrid = e._hybrid_score(in_band, ce_scores, cosine)

    e._assert_finite_in_unit("hybrid", hybrid)
    assert hybrid[0] == pytest.approx(1.0)  # overflow cosine clamped to 1.0
    assert hybrid[1] == pytest.approx(0.9)  # in-band cross-encoder score passes through

    # The source-level clamp (applied in main) also pins the bi-encoder scorer
    # itself within [0, 1].
    clamped = np.clip(cosine, 0.0, 1.0)
    e._assert_finite_in_unit("bi-encoder cosine", clamped)
    assert clamped[0] == pytest.approx(1.0)
