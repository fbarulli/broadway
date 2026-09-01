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

import numpy as np
import pytest

from broadway.config.schema import HPOConfig, ModelHPOSpec, NLPConfig
from broadway.training import nlp
from broadway.training.nlp import entity_resolution_metrics


def _install_fake_sentence_transformers(monkeypatch: pytest.MonkeyPatch, noise_map: dict) -> None:
    """Stub the sentence_transformers module with a deterministic fake encoder."""
    st = types.ModuleType("sentence_transformers")

    class _FakeModel:
        def __init__(self, model_id: str, device: str = "cpu") -> None:
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
    assert m["recall_at_5pct_fpr"] == 1.0
    assert 0.5 < m["pos_median"] < 1.0
    assert m["neg_p90"] < 0.3


def test_entity_resolution_metrics_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        entity_resolution_metrics(np.array([]), np.array([1.0]))
    with pytest.raises(ValueError, match="non-empty"):
        entity_resolution_metrics(np.array([1.0]), np.array([]))


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
        "auc", "recall_at_5pct_fpr", "pos_median", "neg_p90", "encode_s",
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


def _install_fake_finetune_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the sentence_transformer.losses and torch.utils.data submodules."""
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
