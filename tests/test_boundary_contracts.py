"""Parametrized dtype-drift mutation tripwires at every DataFrame boundary.

For each writer→reader boundary in ``src/broadway``, a mutation case flips ONE
column's dtype behind the boundary's back — written through the boundary's real
writer and read through its real reader — and asserts the drift fails LOUD
(pandera ``SchemaError``/``SchemaErrors``, or an explicit guard). A flip that
survives silently fails the test with a message naming the leaky boundary.

Boundaries with no loud failure today are GAP FINDINGS (CONTRACT_G report).
They stay in the suite as ``xfail(strict=True)`` tripwires: the flip still
fails the assertion (documented-leaky), and the day a future contract adds a
guard the case turns XPASS and the suite goes red, forcing the xfail to be
converted into a real assertion. Every xfail reason names the boundary and the
suggested future contract.

Tripwire ledger (current): 8 strict-xfail tripwires, 8 converted, 0 remaining.
The 4 engineered-read dtype flips on B3/B4, the column-reorder, and the
target-dtype gaps were converted by the read-side engineered contract (CONTRACT
H). The raw→canonical int/float coercion mask (G1) is closed by evidence-tagged
coercion (FIX_4, Option E): ``parse_numeric`` records every astype-back-to-
declared event — {column, declared dtype, arriving dtype, rows affected} — into
the run's audit channel instead of silently masking it. The extra-column gap
(G2) is closed by the joined-loader unique-label guard (SchemaError at the
merge point) plus the engineered read-path extra-column guard.

Boundary map (writer → reader; who declares / produces / validates today):

  B1 raw source → etl canonical   (etl/module.py:95)  declares: dataset YAML
      via build_raw_schema; produces: external -> loader READERS[] ->
      canonicalize coercion; validates: build_raw_schema AFTER coercion.
      loud: NaN / object / non-coercible flips, and coercible int<->float
      flips — the latter are now recorded as coercion evidence instead of
      being masked (FIX_4 G1 / Option E, astype-back events surface via etl).
      silent: none.
  B2 etl split -> features         (features/module.py:25,41)  validates:
      engineered schema on features OUTPUT (included columns only).
  B3 train_features -> training    (training/module.py:42-51)  validates:
      engineered schema on read.
  B4 val_features -> evaluate      (evaluate/module.py:49-59)  validates:
      engineered schema on read.
  B5 sample source -> named sample (samples/generate.py:129 -> samples/loader.py:79)
      validates: sample-config schema block on read; else digest/row-count only.
  B6 canonical -> stats            (stats/module.py:34)  no validation (analysis-only)
  B7 canonical -> timeline         (timeline/runners.py:97)  no validation
  B8 raw -> read_sample            (data/loader.py:83)  no validation
  B9 raw -> mlflow lineage         (training/mlflow_utils.py:104)  metadata-only
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

import broadway.evaluate.module as evaluate_module
import broadway.samples.generate as sample_generate
import broadway.samples.loader as sample_loader
import broadway.training.module as training_module
from broadway.cleaning.structural import CoercionRecord
from broadway.config.loader import load_config
from broadway.config.schema import ColumnRole, ColumnSchema, DatasetContract, LookupSpec, TaskType
from broadway.contracts.pandera import build_raw_schema
from broadway.contracts.selectors import datetime_columns, numeric_columns
from broadway.data.cleaner import canonicalize
from broadway.data.loader import load_with_audit
from broadway.features.generic import build_generic_feature_specs
from broadway.features.module import _load_split
from broadway.features.pipeline import FeaturePipeline
from broadway.features.schema import build_engineered_schema
from broadway.lineage.models import DatasetRef, DerivedSpec, SampleSpec


def _contract_frame(contract: DatasetContract, n: int = 40) -> pd.DataFrame:
    """Minimal frame whose names/dtypes match the dataset contract.

    Names and dtypes come from ``configs/dataset/*.yaml`` (derive, don't
    maintain) — a demo rename flows through automatically.
    """
    data: dict[str, object] = {}
    for i, (name, col) in enumerate(contract.columns.items()):
        if col.dtype == "object":
            cats = ["A", "B", "C", "D"] * ((n + 3) // 4)
            data[name] = cats[:n]
        else:
            data[name] = list(range(1 + i, n + 1 + i))
    return pd.DataFrame(data)


def _int64_feature_cols(frame: pd.DataFrame, contract: DatasetContract) -> list[str]:
    """Feature-role columns the fixture currently holds as int64, in contract
    order. Runtime-based (not declaration-based): the tripwire flips whichever
    dtype the frame actually holds, so a drifted declaration cannot silently
    move the mutation onto a different column."""
    return [
        name
        for name, col in contract.columns.items()
        if col.role == ColumnRole.FEATURE and str(frame[name].dtype) == "int64"
    ]


def _assert_loud(
    boundary: str,
    variation: str,
    exc_types: tuple[type[Exception], ...],
    thunk: Callable[[], Any],
) -> None:
    """Run the write→read chain and require a loud failure.

    A flip that survives silently fails the test with a message naming the
    leaky boundary; a loud failure of an unexpected class is flagged too.
    """
    try:
        thunk()
    except exc_types:
        return
    except Exception as exc:  # loud, but the wrong class — still a signal
        pytest.fail(
            f"boundary '{boundary}' ({variation}): failed loud with unexpected "
            f"{type(exc).__name__} (expected {[e.__name__ for e in exc_types]})"
        )
    pytest.fail(
        f"boundary '{boundary}' is LEAKY: {variation} survived silently — no "
        f"dtype guard fired (expected a loud failure: "
        f"{[e.__name__ for e in exc_types]})"
    )


# --------------------------------------------------------------------------- #
# B1 — raw source -> etl canonical (writer-side guard: canonicalize + schema)
# --------------------------------------------------------------------------- #


def _mutate_nan_into_null_free_int(frame: pd.DataFrame, contract: DatasetContract) -> None:
    col = next(iter(_int64_feature_cols(frame, contract)))
    frame[col] = frame[col].astype("float64")
    frame.loc[frame.index[0], col] = float("nan")  # NaN into a null-free column


def _mutate_object_to_int(frame: pd.DataFrame, contract: DatasetContract) -> None:
    col = next(
        name for name, c in contract.columns.items() if str(frame[name].dtype) == "object"
    )
    frame[col] = [i % 4 + 1 for i in range(len(frame))]  # object -> int64


def _mutate_int_to_non_numeric_object(frame: pd.DataFrame, contract: DatasetContract) -> None:
    col = next(iter(_int64_feature_cols(frame, contract)))
    frame[col] = ["non-numeric"] * len(frame)  # int64 -> object (parse failure)


@pytest.mark.parametrize(
    "mutator",
    [
        _mutate_nan_into_null_free_int,
        _mutate_object_to_int,
        _mutate_int_to_non_numeric_object,
    ],
    ids=[
        "nan_into_null_free_int",
        "object_to_int64",
        "int64_to_non_numeric_object",
    ],
)
def test_raw_canonical_boundary_rejects_dtype_mutation(
    mutator: Callable[[pd.DataFrame, DatasetContract], None],
) -> None:
    cfg = load_config("etl", dataset="test", experiment="baseline")
    ds = cfg.dataset
    frame = _contract_frame(ds)
    mutator(frame, ds)

    def thunk() -> None:
        cleaned, *_ = canonicalize(
            frame,
            ds.target,
            datetime_columns(ds),
            cfg.etl.missing_encodings,
            {col: ds.columns[col].dtype for col in numeric_columns(ds)},
        )
        build_raw_schema(ds).validate(cleaned)

    _assert_loud(
        "raw source -> canonical (etl writer guard)",
        f"{mutator.__name__} on a declared column",
        (SchemaError,),
        thunk,
    )


# --------------------------------------------------------------------------- #
# B2 — etl split files -> features step (output engineered-schema guard)
# --------------------------------------------------------------------------- #


def test_split_to_features_boundary_rejects_dtype_mutation(tmp_path: Path) -> None:
    cfg = load_config("features", dataset="test", experiment="baseline")
    cfg = cfg.model_copy(
        update={"environment": cfg.environment.model_copy(update={"data_dir": str(tmp_path)})}
    )
    out_dir = tmp_path / cfg.environment.processed_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = _contract_frame(cfg.dataset)
    flipped = next(iter(_int64_feature_cols(frame, cfg.dataset)))
    frame[flipped] = frame[flipped].astype("float64")
    frame.to_parquet(out_dir / cfg.etl.train_file, index=False)

    def thunk() -> None:
        train, _ = _load_split(cfg)
        pipeline = FeaturePipeline(encodings=cfg.experiment.features.encodings)
        pipeline.fit(train, cfg.dataset.target, cfg.features.encoding_smoothing)
        transformed = pipeline.transform(
            train, cfg.experiment.features, cfg.dataset.target, cfg.features.frequency_fill
        )
        specs = build_generic_feature_specs(cfg.dataset, cfg.experiment.features)
        build_engineered_schema(specs).validate(transformed)

    _assert_loud(
        "etl split -> features (engineered output guard)",
        f"{flipped} int64 -> float64 in {cfg.etl.train_file}",
        (SchemaError,),
        thunk,
    )


# --------------------------------------------------------------------------- #
# B3/B4 — engineered feature files -> training / evaluate (read-side guards)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("boundary", "step", "file_attr", "reader", "col_idx"),
    [
        pytest.param(
            "features -> training (train_features read guard)",
            "train",
            "train_features_file",
            training_module._load_features,
            0,
            id="train_features_feature_1",
        ),
        pytest.param(
            "features -> training (train_features read guard)",
            "train",
            "train_features_file",
            training_module._load_features,
            1,
            id="train_features_feature_2",
        ),
        pytest.param(
            "features -> evaluate (val_features read guard)",
            "evaluate",
            "val_features_file",
            evaluate_module._load_val_features,
            0,
            id="val_features_feature_1",
        ),
        pytest.param(
            "features -> evaluate (val_features read guard)",
            "evaluate",
            "val_features_file",
            evaluate_module._load_val_features,
            1,
            id="val_features_feature_2",
        ),
    ],
)
def test_engineered_read_boundary_rejects_dtype_mutation(
    boundary: str,
    step: str,
    file_attr: str,
    reader: Callable[[Any], Any],
    col_idx: int,
    tmp_path: Path,
) -> None:
    cfg = load_config(step, dataset="test", experiment="baseline", analysis="test")
    cfg = cfg.model_copy(
        update={"environment": cfg.environment.model_copy(update={"data_dir": str(tmp_path)})}
    )
    out_dir = tmp_path / cfg.environment.processed_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = _contract_frame(cfg.dataset)
    flipped = _int64_feature_cols(frame, cfg.dataset)[col_idx]
    frame[flipped] = frame[flipped].astype("float64")
    frame.to_parquet(out_dir / getattr(cfg.etl, file_attr), index=False)

    _assert_loud(
        boundary,
        f"{flipped} int64 -> float64 in {getattr(cfg.etl, file_attr)}",
        (SchemaError,),
        lambda: reader(cfg),
    )


# --------------------------------------------------------------------------- #
# B5 — named-sample generation -> read_named_sample (schema-block guard)
# --------------------------------------------------------------------------- #

_SAMPLE_SCHEMA = {
    "fare_amount": {"dtype": "float64", "nullable": False},
    "trip_distance": {"dtype": "float64", "nullable": False},
    "trip_duration_minutes": {"dtype": "float64", "nullable": False},
    "speed_mph": {"dtype": "float64", "nullable": False},
}


def _sample_spec(source_path: str) -> SampleSpec:
    return SampleSpec(
        name="tripwire_sample",
        role="estimation",
        path="unused",
        version="v1",
        seed=7,
        size=12,
        source=DatasetRef(name="src", path=source_path),
        columns=["fare_amount", "trip_distance", "trip_duration_minutes"],
        derived=[
            DerivedSpec(
                name="speed_mph",
                formula="rate_per_hour",
                columns={
                    "distance": "trip_distance",
                    "duration_minutes": "trip_duration_minutes",
                },
            )
        ],
        schema=_SAMPLE_SCHEMA,
    )


def test_named_sample_boundary_rejects_source_dtype_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source.parquet"
    source = pd.DataFrame(
        {
            "fare_amount": [5.0, 6.0, 7.0, 8.0] * 6,
            "trip_distance": [1.0, 2.0, 3.0, 4.0] * 6,
            "trip_duration_minutes": [10.0, 20.0, 30.0, 40.0] * 6,
        }
    )
    source.to_parquet(source_path, index=False)
    spec = _sample_spec(str(source_path))
    monkeypatch.setattr(sample_generate, "load_sample", lambda name: spec)
    monkeypatch.setattr(sample_loader, "load_sample", lambda name: spec)

    # conformant control: the fixture itself must survive the real chain
    samples_dir = tmp_path / "samples"
    sample_generate.generate_sample("tripwire_sample", samples_dir=samples_dir)
    sample_loader.read_named_sample("tripwire_sample", samples_dir=samples_dir)

    # flip the producer side: fare_amount declared float64 is emitted as int64
    source["fare_amount"] = source["fare_amount"].astype("int64")
    source.to_parquet(source_path, index=False)
    flipped_dir = tmp_path / "samples_flipped"

    def thunk() -> None:
        sample_generate.generate_sample("tripwire_sample", samples_dir=flipped_dir)
        sample_loader.read_named_sample("tripwire_sample", samples_dir=flipped_dir)

    _assert_loud(
        "sample source -> artifact -> read_named_sample",
        "fare_amount float64 -> int64 in the generator's source parquet",
        (SchemaErrors,),
        thunk,
    )


# --------------------------------------------------------------------------- #
# FORMER GAP FINDINGS — closed by FIX_4 and now real passing assertions.
# G1 (raw→canonical int/float coercion mask): Option E makes the astype-back
# evidence-tagged, so the flip produces a recorded, visible coercion event
# instead of silence. G2 (extra-column strictness): the joined loader now
# raises SchemaError on duplicate labels at the merge point, and the
# engineered read path rejects columns outside the declared surface. The
# column-reorder and target-dtype gaps were closed earlier by the read-side
# engineered contract and run below as real assertions. See CONTRACT_G report
# for the original gap table and FIX_4 for the closures.
# --------------------------------------------------------------------------- #


def test_gap_raw_boundary_int_float_coercion_mask() -> None:
    """G1 converted (FIX_4 Option E): the flip is no longer masked — it is
    recorded evidence. Exercises the exact canonicalize call the pipeline
    performs (etl/module.py) with the evidence collector, then asserts the
    SPECIFIC record contents ({column, int64 declared, float64 arriving, rows
    affected}) — not merely that some coercion record exists."""
    cfg = load_config("etl", dataset="test", experiment="baseline")
    ds = cfg.dataset
    frame = _contract_frame(ds)
    flipped = next(iter(_int64_feature_cols(frame, ds)))
    frame[flipped] = frame[flipped].astype("float64")  # whole numbers, no NaN

    coercions: list[CoercionRecord] = []
    cleaned, *_ = canonicalize(
        frame,
        ds.target,
        datetime_columns(ds),
        cfg.etl.missing_encodings,
        {col: ds.columns[col].dtype for col in numeric_columns(ds)},
        coercions=coercions,
    )

    # the repair still happens exactly as before: post-coercion validation
    # passes on the restored dtype (zero behavior change to outputs)
    build_raw_schema(ds).validate(cleaned)
    assert str(cleaned[flipped].dtype) == ds.columns[flipped].dtype

    # anti-vacuous: the evidence record names the exact column and dtype pair
    record = next(r for r in coercions if r.column == flipped)
    assert record.declared_dtype == "int64"
    assert record.arriving_dtype == "float64"
    assert record.rows_affected == len(frame)


def test_gap_engineered_boundary_extra_column(tmp_path: Path) -> None:
    """G2 converted (FIX_4): the engineered read path rejects columns outside
    the declared surface (contract ∪ joined-lookup ∪ feature specs)."""
    cfg = load_config("train", dataset="test", experiment="baseline", analysis="test")
    cfg = cfg.model_copy(
        update={"environment": cfg.environment.model_copy(update={"data_dir": str(tmp_path)})}
    )
    out_dir = tmp_path / cfg.environment.processed_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = _contract_frame(cfg.dataset)
    frame["stowaway"] = "zzz"  # unexpected column — the joined-loader class
    frame.to_parquet(out_dir / cfg.etl.train_features_file, index=False)

    _assert_loud(
        "features -> training",
        "unexpected extra column 'stowaway'",
        (SchemaError,),
        lambda: training_module._load_features(cfg),
    )


def test_joined_loader_rejects_pre_existing_lookup_suffix_collision(
    tmp_path: Path,
) -> None:
    """NEW tripwire (FIX_4 G2): the 2026-08-23 incident shape at the joined-
    loader boundary — raw frame carries BOTH ``Borough`` and ``Borough_lookup``
    while the lookup contributes ``Borough`` (renamed to ``Borough_lookup`` by
    the merge) — must raise ``SchemaError`` from the loader, not the accidental
    deep failure the duplicate-label frame tripped inside ``audit_lookup_values``
    (measured pre-fix: ``ValueError: cannot reindex on an axis with duplicate
    labels`` on this exact shape)."""
    cfg = load_config("etl", dataset="taxi", experiment="taxi")
    ds = cfg.dataset
    frame = _contract_frame(ds)
    frame["pickup_datetime"] = pd.to_datetime("2024-01-01")  # contract datetime
    frame["Borough"] = "x"
    frame["Borough_lookup"] = "y"  # pre-existing _lookup column — incident shape
    raw_path = tmp_path / "incident_raw.csv"
    frame.to_csv(raw_path, index=False)

    lookup_path = tmp_path / "zones.csv"
    pd.DataFrame(
        {
            "LocationID": list(range(6)),
            "Borough": ["A"] * 6,
            "Zone": ["Z1"] * 6,
            "service_zone": ["S1"] * 6,
        }
    ).to_csv(lookup_path, index=False)

    ds = ds.model_copy(
        update={
            "path": str(raw_path),
            "lookup_tables": {
                "pickup_location_id": LookupSpec(path=str(lookup_path), key="LocationID")
            },
        }
    )

    _assert_loud(
        "joined loader -> raw ingest",
        "pre-existing 'Borough_lookup' collides with lookup 'Borough' renamed by the merge",
        (SchemaError,),
        lambda: load_with_audit(ds),
    )


def test_joined_loader_duplicate_label_error_names_all_collisions(
    tmp_path: Path,
) -> None:
    """NEW tripwire (FIX_4 G2): TWO distinct lookup columns colliding in the
    SAME merge — the SchemaError message must name ALL duplicated labels, not
    just the first found."""
    raw_path = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "area": [1, 2, 3],
            "price": [10, 20, 30],
            "X": [1, 2, 3],
            "X_lookup": ["a", "b", "c"],
            "Y": [1, 2, 3],
            "Y_lookup": ["d", "e", "f"],
        }
    ).to_csv(raw_path, index=False)

    lookup_path = tmp_path / "lookup.csv"
    pd.DataFrame({"LocationID": [1, 2, 3], "X": [4, 5, 6], "Y": [7, 8, 9]}).to_csv(
        lookup_path, index=False
    )

    ds = DatasetContract(
        name="collision",
        path=str(raw_path),
        target="price",
        task=TaskType.REGRESSION,
        datetime_column=None,
        columns={
            "area": ColumnSchema(dtype="int64", null_count=0, role=ColumnRole.FEATURE),
            "price": ColumnSchema(dtype="int64", null_count=0, role=ColumnRole.TARGET),
        },
        lookup_tables={"area": LookupSpec(path=str(lookup_path), key="LocationID")},
    )

    with pytest.raises(SchemaError) as exc_info:
        load_with_audit(ds)
    message = str(exc_info.value)
    assert "duplicate" in message
    assert "X_lookup" in message
    assert "Y_lookup" in message


def test_gap_engineered_boundary_column_reorder(tmp_path: Path) -> None:
    cfg = load_config("train", dataset="test", experiment="baseline", analysis="test")
    cfg = cfg.model_copy(
        update={"environment": cfg.environment.model_copy(update={"data_dir": str(tmp_path)})}
    )
    out_dir = tmp_path / cfg.environment.processed_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = _contract_frame(cfg.dataset)
    reordered = list(frame.columns)
    frame = frame[reordered[1:] + reordered[:1]]  # positional-consumer reorder
    frame.to_parquet(out_dir / cfg.etl.train_features_file, index=False)

    _assert_loud(
        "features -> training",
        "column reorder (first column moved to last)",
        (SchemaError,),
        lambda: training_module._load_features(cfg),
    )


def test_gap_target_dtype_flip_through_training(tmp_path: Path) -> None:
    cfg = load_config("train", dataset="test", experiment="baseline", analysis="test")
    cfg = cfg.model_copy(
        update={"environment": cfg.environment.model_copy(update={"data_dir": str(tmp_path)})}
    )
    out_dir = tmp_path / cfg.environment.processed_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = _contract_frame(cfg.dataset)
    frame[cfg.dataset.target] = frame[cfg.dataset.target].astype("float64")
    frame.to_parquet(out_dir / cfg.etl.train_features_file, index=False)

    _assert_loud(
        "features -> training",
        f"target '{cfg.dataset.target}' int64 -> float64",
        (SchemaError,),
        lambda: training_module._load_features(cfg),
    )
