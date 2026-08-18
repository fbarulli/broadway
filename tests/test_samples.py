"""Named-sample registry tests — synthetic data only (no taxi coupling).

Covers the definition → generation → immutable artifact → validated
consumption loop: generate/read round-trip, immutability, and every loader
guard (integrity, version resolution, row count, definition digest, schema).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from pandera.errors import SchemaErrors

from broadway.config import loader
from broadway.lineage.sample import load_sample
from broadway.samples import generate_sample, read_named_sample
from broadway.samples.models import Sample

SCHEMA = {
    "a": {"dtype": "float64", "nullable": False, "checks": [{"op": ">", "value": 0.0}]},
    "b": {"dtype": "float64", "nullable": False},
}
PROVENANCE_KEYS = {
    "name", "version", "source", "seed", "size", "row_count", "columns",
    "dtypes", "filters", "definition_sha256", "artifact_sha256", "created_at",
}


def _write_config(
    tmp_path: Path, name: str, source_path: Path, **overrides: object
) -> None:
    configs = tmp_path / "configs"
    (configs / "sample").mkdir(parents=True, exist_ok=True)
    spec = {
        "name": name,
        "role": "estimation",
        "version": "v1",
        "path": f"data/samples/{name}@v1.parquet",
        "description": "synthetic sample",
        "source": {"name": "src", "path": str(source_path)},
        "seed": 7,
        "size": 30,
        "columns": ["a", "b"],
        "filters": [{"column": "a", "op": ">", "value": 0.0}],
        "schema": SCHEMA,
        **overrides,
    }
    (configs / "sample" / f"{name}.yaml").write_text(
        yaml.safe_dump(spec), encoding="utf-8"
    )


@pytest.fixture
def synthetic_source(tmp_path: Path) -> Path:
    src = tmp_path / "src.parquet"
    pd.DataFrame({"a": [float(i) for i in range(50)], "b": [float(i) for i in range(50)]}).to_parquet(
        src, index=False
    )
    return src


def _generated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, source_path: Path
) -> Path:
    monkeypatch.setattr(loader, "CONFIGS_DIR", tmp_path / "configs")
    samples_dir = tmp_path / "samples"
    generate_sample(name, samples_dir=samples_dir)
    return samples_dir


def test_generate_creates_artifact_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_source: Path
) -> None:
    _write_config(tmp_path, "gen_sample", synthetic_source)
    samples_dir = _generated(tmp_path, monkeypatch, "gen_sample", synthetic_source)

    artifact = samples_dir / "gen_sample@v1.parquet"
    assert artifact.exists()
    provenance = json.loads(artifact.with_suffix(".json").read_text(encoding="utf-8"))
    assert set(provenance) == PROVENANCE_KEYS
    assert provenance["name"] == "gen_sample"
    assert provenance["version"] == "v1"
    assert provenance["row_count"] == len(pd.read_parquet(artifact))
    assert provenance["columns"] == ["a", "b"]
    assert provenance["filters"] == [{"column": "a", "op": ">", "value": 0.0}]
    # Row count honors the filter: every artifact row satisfies it, and the
    # sample cannot exceed the requested size.
    df = pd.read_parquet(artifact)
    assert (df["a"] > 0.0).all()
    assert len(df) <= 30


def test_read_named_sample_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_source: Path
) -> None:
    _write_config(tmp_path, "roundtrip", synthetic_source)
    samples_dir = _generated(tmp_path, monkeypatch, "roundtrip", synthetic_source)

    sample = read_named_sample("roundtrip", samples_dir=samples_dir)

    assert isinstance(sample, Sample)
    assert sample.spec.version == "v1"
    assert sample.spec.seed == 7
    assert sample.spec.size == 30
    assert list(sample.df.columns) == ["a", "b"]
    assert sample.provenance["row_count"] == len(sample.df)
    assert sample.provenance["name"] == "roundtrip"


def test_regenerating_same_version_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_source: Path
) -> None:
    _write_config(tmp_path, "immutable", synthetic_source)
    samples_dir = _generated(tmp_path, monkeypatch, "immutable", synthetic_source)

    with pytest.raises(FileExistsError, match="bump `version`"):
        generate_sample("immutable", samples_dir=samples_dir)


def test_corrupted_artifact_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_source: Path
) -> None:
    _write_config(tmp_path, "integrity", synthetic_source)
    samples_dir = _generated(tmp_path, monkeypatch, "integrity", synthetic_source)

    artifact = samples_dir / "integrity@v1.parquet"
    payload = bytearray(artifact.read_bytes())
    payload[0] ^= 0xFF
    artifact.write_bytes(bytes(payload))

    with pytest.raises(ValueError, match="digest mismatch"):
        read_named_sample("integrity", samples_dir=samples_dir)


def test_version_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_source: Path
) -> None:
    _write_config(tmp_path, "verres", synthetic_source)
    samples_dir = _generated(tmp_path, monkeypatch, "verres", synthetic_source)

    # No version passed → resolves to the version declared in the config.
    sample = read_named_sample("verres", samples_dir=samples_dir)
    assert sample.provenance["version"] == "v1"

    # An explicit unknown version must fail loudly — never fall back.
    with pytest.raises(FileNotFoundError):
        read_named_sample("verres", version="v9", samples_dir=samples_dir)


def test_row_count_mismatch_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_source: Path
) -> None:
    _write_config(tmp_path, "rowcount", synthetic_source)
    samples_dir = _generated(tmp_path, monkeypatch, "rowcount", synthetic_source)

    provenance_path = samples_dir / "rowcount@v1.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["row_count"] += 1
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="row count mismatch"):
        read_named_sample("rowcount", samples_dir=samples_dir)


def test_definition_digest_mismatch_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_source: Path
) -> None:
    _write_config(tmp_path, "defdigest", synthetic_source)
    samples_dir = _generated(tmp_path, monkeypatch, "defdigest", synthetic_source)

    provenance_path = samples_dir / "defdigest@v1.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["definition_sha256"] = "stale-digest"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="definition changed"):
        read_named_sample("defdigest", samples_dir=samples_dir)


def test_schema_violation_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "src.parquet"
    pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]}).to_parquet(src, index=False)
    # Declared schema expects float64 for column "a", but the generated
    # artifact holds int64 — read must reject it.
    _write_config(tmp_path, "schema_bad", src, size=3)
    samples_dir = _generated(tmp_path, monkeypatch, "schema_bad", src)

    with pytest.raises(SchemaErrors):
        read_named_sample("schema_bad", samples_dir=samples_dir)


def test_existing_config_still_parses() -> None:
    spec = load_sample("taxi_diagnostic")
    assert spec.name == "taxi_diagnostic"
    assert spec.version == "v1"
    assert spec.source is None
    assert spec.path == "data/processed/joined_sample_live.parquet"
