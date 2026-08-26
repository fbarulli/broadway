"""Slice 4b — champion manifest by logging path (scripts/check_champion_manifest.sh).

Seeds a hermetic tmp file-store registry with one champion per logging-path
class — new-path Pipeline+signature, legacy ModelPyFunc-wrapped bare model,
and ambiguous (signature-less sklearn artifact, the pre-Slice-3 logging path)
— then asserts the manifest listing structure, all three buckets, and the
--strict retirement-condition exit codes. The legacy bare-model champion is
fabricated through the real ModelPyFunc logging path (pyfunc.log_model with
the repo wrapper class), so classification runs against genuine artifacts of
both named paths plus a genuinely ambiguous one.

Hermetic: every artifact is logged to a tmp file-store tracking URI and tmp
artifact location — no server, no network.
"""

from __future__ import annotations

import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from broadway.training.mlflow_utils import (
    AMBIGUOUS,
    BARE_MODEL,
    PIPELINE_SIGNATURE,
    ChampionArtifact,
    list_champions,
    log_model,
    setup_mlflow,
)
from broadway.training.models.pyfunc_wrapper import ModelPyFunc

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_champion_manifest.sh"


def _features() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame({"cat": ["x", "y", "x", "y"], "num": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([2.0, 3.0, 4.0, 5.0])
    return X, y


def _register_champion(name: str, uri: str) -> None:
    client = mlflow.tracking.MlflowClient()
    version = mlflow.register_model(uri, name)
    client.set_registered_model_alias(name, "champion", version.version)


def _log_pipeline_champion(name: str, X: pd.DataFrame, y: pd.Series) -> None:
    """Fit + log a tiny Pipeline via the train path's helpers (new path)."""
    pipeline = Pipeline(
        [
            (
                "pre",
                ColumnTransformer(
                    [("oh", OneHotEncoder(handle_unknown="ignore"), ["cat"])],
                    remainder="passthrough",
                ),
            ),
            ("model", LinearRegression()),
        ]
    )
    pipeline.fit(X, y)
    with mlflow.start_run():
        uri = log_model(pipeline, "model", signature=infer_signature(X, y))
        _register_champion(name, uri)


def _seed_store(root: Path) -> None:
    """Log one champion per logging-path class into a tmp file-store registry."""
    X, y = _features()
    setup_mlflow(str(root / "mlruns"), "champion_manifest")

    _log_pipeline_champion("champ_pipeline", X, y)

    bare = LinearRegression().fit(X[["num"]], y)
    with tempfile.TemporaryDirectory() as tmp_dir:
        bundle = Path(tmp_dir) / "model.pkl"
        bundle.write_bytes(pickle.dumps(bare))
        with mlflow.start_run():
            info = mlflow.pyfunc.log_model(
                python_model=ModelPyFunc(),
                artifacts={"model": str(bundle)},
                name="model",
            )
            _register_champion("champ_bare", info.model_uri)

    with mlflow.start_run():
        # pre-Slice-3 style: sklearn bare model logged WITHOUT an explicit signature.
        info = mlflow.sklearn.log_model(bare, "model")
        _register_champion("champ_ambig", info.model_uri)


def _run(tracking_uri: str, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SCRIPT), "--tracking-uri", tracking_uri, *extra],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=150,  # fail fast instead of hanging on a uv probe stall (2026-08-23)
    )


def test_classify_champion_returns_buckets(tmp_path: Path) -> None:
    _seed_store(tmp_path)
    records = list_champions(str(tmp_path / "mlruns"))
    by_name = {record.model_name: record for record in records}
    assert set(by_name) == {"champ_pipeline", "champ_bare", "champ_ambig"}
    assert by_name["champ_pipeline"].bucket == PIPELINE_SIGNATURE
    assert by_name["champ_bare"].bucket == BARE_MODEL
    assert by_name["champ_ambig"].bucket == AMBIGUOUS
    assert by_name["champ_ambig"].reason
    assert all(isinstance(record, ChampionArtifact) for record in records)


def test_manifest_lists_all_three_buckets(tmp_path: Path) -> None:
    _seed_store(tmp_path)
    result = _run(str(tmp_path / "mlruns"))
    assert result.returncode == 0
    out = result.stdout
    assert "champion manifest — 3 champion(s)" in out
    assert "bare_model (ModelPyFunc-wrapped) (1):" in out
    assert "pipeline_signature (new path) (1):" in out
    assert "ambiguous (1):" in out
    assert "champ_pipeline (v1)" in out and "champ_bare (v1)" in out and "champ_ambig (v1)" in out
    assert "artifact: models:/" in out
    assert "reason: sklearn flavor without explicit signature" in out


def test_strict_exits_nonzero_when_bare_or_ambiguous(tmp_path: Path) -> None:
    _seed_store(tmp_path)
    result = _run(str(tmp_path / "mlruns"), "--strict")
    assert result.returncode == 1
    assert "STRICT: bare_model or ambiguous champions present" in result.stdout


def test_strict_exits_zero_when_only_pipeline_champions(tmp_path: Path) -> None:
    X, y = _features()
    setup_mlflow(str(tmp_path / "mlruns"), "champion_manifest")
    _log_pipeline_champion("champ_pipeline", X, y)
    result = _run(str(tmp_path / "mlruns"), "--strict")
    assert result.returncode == 0
    assert "STRICT: OK" in result.stdout


def test_unreadable_artifact_is_ambiguous_with_reason(tmp_path: Path) -> None:
    _seed_store(tmp_path)
    client = mlflow.tracking.MlflowClient()
    version = client.get_model_version_by_alias("champ_pipeline", "champion")
    shutil.rmtree(mlflow.artifacts.download_artifacts(version.source))
    result = _run(str(tmp_path / "mlruns"))
    assert result.returncode == 0
    out = result.stdout
    assert "ambiguous (2):" in out
    assert "champ_pipeline (v1)" in out
    assert "reason: metadata unreadable" in out
