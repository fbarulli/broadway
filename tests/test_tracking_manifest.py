from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest
from contract_fixture import frame_slots

import broadway.etl.module as etl_module
from broadway.config.loader import load_config
from broadway.discover.profile import build_profile
from broadway.lineage import records
from broadway.lineage.models import TransformAudit
from broadway.tracking.manifest import (
    DataManifest,
    build_manifest,
    load_profile,
    manifest_path,
    sha256_of_file,
    timestamps_from_frame,
    timestamps_from_profile,
    write_etl_manifest,
    write_manifest,
)


def _audit(rows_in: int = 4, rows_out: int = 3) -> TransformAudit:
    return TransformAudit(
        rows_in=rows_in,
        rows_out=rows_out,
        rows_dropped_total=rows_in - rows_out,
        rows_dropped_unexplained=0,
        reasons=["duplicates: -1 rows"],
        columns_before=["a", "target"],
        columns_after=["a", "target"],
        columns_added=[],
        columns_removed=[],
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_at": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-05"]),
            "value": [1, 2, 3],
        }
    )


def test_sha256_of_file_matches_hashlib(tmp_path: Path) -> None:
    target = tmp_path / "blob.bin"
    target.write_bytes(b"manifest-bytes")
    assert sha256_of_file(target) == hashlib.sha256(b"manifest-bytes").hexdigest()


def test_sha256_of_missing_file_fails_loud(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sha256_of_file(tmp_path / "absent.parquet")


def test_timestamps_from_profile_aggregates_datetime_bounds() -> None:
    profile = build_profile("demo", "demo.csv", _frame())
    low, high = timestamps_from_profile(profile)
    assert low is not None and high is not None
    assert low <= high
    assert "2024-01-01" in low
    assert "2024-01-05" in high


def test_timestamps_from_profile_without_datetimes_is_empty() -> None:
    df = pd.DataFrame({"value": [1, 2]})
    low, high = timestamps_from_profile(build_profile("demo", "demo.csv", df))
    assert (low, high) == (None, None)


def test_timestamps_from_frame_matches_profile() -> None:
    frame = _frame()
    from_frame = timestamps_from_frame(frame)
    from_profile = timestamps_from_profile(build_profile("demo", "demo.csv", frame))
    assert from_frame == from_profile
    assert timestamps_from_frame(pd.DataFrame({"v": [1]})) == (None, None)


def test_timestamps_from_frame_all_nat_returns_empty() -> None:
    frame = pd.DataFrame({"event_at": pd.Series([pd.NaT, pd.NaT], dtype="datetime64[ns]")})
    assert timestamps_from_frame(frame) == (None, None)


def test_timestamps_from_frame_skips_all_nat_column() -> None:
    frame = pd.DataFrame(
        {
            "empty_at": pd.Series([pd.NaT, pd.NaT], dtype="datetime64[ns]"),
            "event_at": pd.to_datetime(["2024-01-01", "2024-01-05"]),
        }
    )
    low, high = timestamps_from_frame(frame)
    assert low is not None and high is not None
    assert "2024-01-01" in low
    assert "2024-01-05" in high


def test_build_and_write_manifest_roundtrip(tmp_path: Path) -> None:
    cfg = load_config("etl", dataset="test", experiment="baseline")
    assert cfg.dataset is not None and cfg.experiment is not None
    parquet = tmp_path / "test_canonical.parquet"
    _frame().to_parquet(parquet, index=False)
    manifest = build_manifest(
        dataset=cfg.dataset,
        audit=_audit(),
        canonical_path=parquet,
        schema_contract=cfg.experiment.data_source.schema_contract,
        timestamp_min="2024-01-01",
        timestamp_max="2024-01-05",
    )
    out = write_manifest(manifest, tmp_path / "tracking")
    assert out == manifest_path(tmp_path / "tracking")
    reloaded = DataManifest.model_validate_json(out.read_text(encoding="utf-8"))
    assert reloaded == manifest
    assert reloaded.sha256 == sha256_of_file(parquet)
    assert (reloaded.rows_in, reloaded.rows_out) == (4, 3)


def test_write_etl_manifest_prefers_profile_over_frame(tmp_path: Path) -> None:
    cfg = load_config("etl", dataset="test", experiment="baseline")
    assert cfg.dataset is not None and cfg.experiment is not None
    parquet = tmp_path / "test_canonical.parquet"
    _frame().to_parquet(parquet, index=False)
    profile = build_profile("test", "raw.csv", _frame())
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    out = write_etl_manifest(
        dataset=cfg.dataset,
        audit=_audit(),
        canonical_path=parquet,
        schema_contract=cfg.experiment.data_source.schema_contract,
        df=pd.DataFrame({"v": [9]}),
        tracking_root=tmp_path / "tracking",
        profile_file=profile_path,
    )
    reloaded = DataManifest.model_validate_json(out.read_text(encoding="utf-8"))
    assert reloaded.timestamp_min is not None and "2024-01-01" in reloaded.timestamp_min


def test_load_profile_missing_returns_none(tmp_path: Path) -> None:
    assert load_profile(tmp_path / "absent.json") is None


def test_write_etl_manifest_missing_parquet_fails_loud(tmp_path: Path) -> None:
    cfg = load_config("etl", dataset="test", experiment="baseline")
    assert cfg.dataset is not None and cfg.experiment is not None
    with pytest.raises(FileNotFoundError):
        write_etl_manifest(
            dataset=cfg.dataset,
            audit=_audit(),
            canonical_path=tmp_path / "absent.parquet",
            schema_contract=cfg.experiment.data_source.schema_contract,
            tracking_root=tmp_path / "tracking",
        )


def test_write_etl_manifest_without_profile_or_frame_uses_none(tmp_path: Path) -> None:
    cfg = load_config("etl", dataset="test", experiment="baseline")
    assert cfg.dataset is not None and cfg.experiment is not None
    parquet = tmp_path / "test_canonical.parquet"
    _frame().to_parquet(parquet, index=False)
    out = write_etl_manifest(
        dataset=cfg.dataset,
        audit=_audit(),
        canonical_path=parquet,
        schema_contract=cfg.experiment.data_source.schema_contract,
        df=None,
        profile=None,
        tracking_root=tmp_path / "tracking",
        profile_file=tmp_path / "absent-profile.json",
    )
    reloaded = DataManifest.model_validate_json(out.read_text(encoding="utf-8"))
    assert (reloaded.timestamp_min, reloaded.timestamp_max) == (None, None)


def test_etl_run_writes_data_manifest(tmp_path: Path, monkeypatch) -> None:
    cfg = load_config("etl", dataset="test", experiment="baseline")
    assert cfg.dataset is not None and cfg.experiment is not None
    cfg = cfg.model_copy(
        update={"environment": cfg.environment.model_copy(update={"data_dir": str(tmp_path)})}
    )
    assert cfg.experiment is not None
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")
    monkeypatch.setenv("BROADWAY_TRACKING_DIR", str(tmp_path / "tracking"))
    monkeypatch.setenv("BROADWAY_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    feats, target = frame_slots(cfg.dataset)
    df = pd.DataFrame(
        {
            feats[1]: [100, 100, 150, 200],
            feats[2]: ["a", "a", "b", "c"],
            target: [10, 10, 30, 40],
            feats[0]: [2, 2, 3, 4],
        }
    )
    monkeypatch.setattr(etl_module, "load_with_audit", lambda dataset: (df.copy(), [], []))

    etl_module.run(cfg)

    manifest_file = tmp_path / "tracking" / "data_manifest.json"
    assert manifest_file.exists()
    manifest = DataManifest.model_validate_json(manifest_file.read_text(encoding="utf-8"))
    canonical = tmp_path / cfg.environment.processed_subdir / "test_canonical.parquet"
    assert manifest.dataset == "test"
    assert manifest.canonical_path == str(canonical)
    assert manifest.sha256 == sha256_of_file(canonical)
    assert (manifest.rows_in, manifest.rows_out) == (4, 3)
    assert manifest.schema_contract == cfg.experiment.data_source.schema_contract
