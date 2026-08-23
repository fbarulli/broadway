"""Include-order == producer-order parity guard (CONTRACT FIX_2).

The ordered read schema of the engineered read contract
(``engineered_schema_for`` with ``ordered=True``) rebuilds its column order
from the experiment config's ``features.include`` list, while the features step
preserves the ETL frame's column order and appends derived then encoded columns
(``FeaturePipeline.transform``). If an experiment's include order ever drifts
from the producer's stable write order, ANY real read through the
training/evaluate loaders fails loud with a false-positive ``SchemaError`` on
an otherwise conformant frame.

Binding direction (human-ratified, fail-loud philosophy): the CONFIG aligns to
the producer; the producer does NOT change to follow the config, and the read
guards are never weakened. This suite asserts that the specs (schema) column
order equals the real features-step output order for every in-scope pair, so
the drift class cannot silently recur. ``SchemaError`` is never swallowed here:
the real producer chain runs as-is, and the assertion is order-equality only
(Ratified decision 5 — no dtype assertions).

In-scope rule (Ratified decision 2): a dataset x experiment pair is in scope
iff every name the experiment's feature graph references (include columns,
derived ``source``s, encoding ``columns``) exists in the dataset contract's
``columns``; out-of-scope pairs are skipped and counted
(``_OUT_OF_SCOPE_PAIRS``), and the in-scope set is asserted non-empty so the
guard can never silently protect nothing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from broadway.config.loader import load_config
from broadway.config.schema import PipelineConfig
from broadway.features.generic import build_generic_feature_specs
from broadway.features.pipeline import FeaturePipeline

_CONFIGS_DIR = Path("configs")


def _all_pairs() -> list[tuple[str, str]]:
    """Every dataset x experiment pair, enumerated and sorted — never a
    hardcoded list, so a new config joins the parity surface automatically."""
    datasets = sorted(p.stem for p in (_CONFIGS_DIR / "dataset").glob("*.yaml"))
    experiments = sorted(p.stem for p in (_CONFIGS_DIR / "experiment").glob("*.yaml"))
    return [(dataset, experiment) for dataset in datasets for experiment in experiments]


def _referenced_names(cfg: PipelineConfig) -> set[str]:
    """Every name the experiment's feature graph references: include columns,
    derived ``source``s, and encoding ``columns``."""
    assert cfg.experiment is not None
    features = cfg.experiment.features
    names = set(features.include)
    names.update(derived.source for derived in features.derived)
    for encoding in features.encodings:
        names.update(encoding.columns)
    return names


def _pair_in_scope(cfg: PipelineConfig) -> bool:
    """Ratified decision 2: in scope iff every referenced name exists in the
    dataset contract's ``columns``."""
    assert cfg.dataset is not None
    return _referenced_names(cfg) <= set(cfg.dataset.columns)


def _in_scope_pairs() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Partition all enumerated pairs into in-scope / out-of-scope."""
    in_scope: list[tuple[str, str]] = []
    out_of_scope: list[tuple[str, str]] = []
    for dataset, experiment in _all_pairs():
        cfg = load_config("features", dataset=dataset, experiment=experiment)
        (in_scope if _pair_in_scope(cfg) else out_of_scope).append((dataset, experiment))
    return in_scope, out_of_scope


_IN_SCOPE_PAIRS, _OUT_OF_SCOPE_PAIRS = _in_scope_pairs()

assert _IN_SCOPE_PAIRS, (
    "no dataset x experiment pair is in scope for the include-order parity "
    "guard — the parametrized cases would silently guard nothing"
)


def _contract_declared_frame(cfg: PipelineConfig, n: int = 16) -> pd.DataFrame:
    """ETL-split-shaped frame for the pair's dataset: every contract column in
    config order with its declared dtype, then the experiment's builder-param
    columns (the joined-lookup columns multi-input builders read). Names and
    dtypes derive from the configs — nothing hardcoded."""
    assert cfg.dataset is not None
    data: dict[str, object] = {}
    for idx, (name, column) in enumerate(cfg.dataset.columns.items()):
        if column.dtype == "object":
            data[name] = [f"cat_{k % 4}" for k in range(n)]
        elif column.dtype == "datetime64":
            data[name] = pd.date_range("2024-01-01", periods=n, freq="h")
        elif column.dtype.startswith("int"):
            data[name] = list(range(1 + idx, n + 1 + idx))
        else:
            data[name] = [float(k) + 0.5 for k in range(1 + idx, n + 1 + idx)]
    frame = pd.DataFrame(data)
    if cfg.experiment is not None and cfg.experiment.features.builder_params is not None:
        params = cfg.experiment.features.builder_params
        frame[params.group_col] = [f"grp_{k % 3}" for k in range(n)]
        frame[params.lookup_col] = [f"grp_{k % 3}" for k in range(n)]
    return frame


@pytest.mark.parametrize(
    ("dataset", "experiment"),
    _IN_SCOPE_PAIRS,
    ids=[f"{dataset} x {experiment}" for dataset, experiment in _IN_SCOPE_PAIRS],
)
def test_specs_order_matches_producer_write_order(dataset: str, experiment: str) -> None:
    """Specs (schema) column order == features-step output order for the pair.

    Runs the exact producer chain the features step executes
    (``features/module.py`` fit + transform) on a contract-declared-dtype
    frame, then asserts order-equality only — no dtype assertions per Ratified
    decision 5, and no ``SchemaError`` is swallowed.
    """
    cfg = load_config("features", dataset=dataset, experiment=experiment)
    assert cfg.dataset is not None and cfg.experiment is not None
    specs = build_generic_feature_specs(cfg.dataset, cfg.experiment.features)
    frame = _contract_declared_frame(cfg)
    pipeline = FeaturePipeline(encodings=cfg.experiment.features.encodings)
    pipeline.fit(frame, cfg.dataset.target, cfg.features.encoding_smoothing)
    transformed = pipeline.transform(
        frame, cfg.experiment.features, cfg.dataset.target, cfg.features.frequency_fill
    )
    assert list(specs) == [column for column in transformed.columns if column in specs]
