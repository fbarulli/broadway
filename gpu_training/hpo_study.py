"""The HPO study definition shared by the fan-out launcher and its workers.

One file names a study: the project HPO spec (search space, direction, target
metric, budget), the project dataset/experiment configs the objective adapter
resolves, and the storage policy. Nothing here invents a dial — the HPO spec
and the project configs own every value, and both the launcher (budget +
preflight) and every worker (search space + run record) read this one file.
"""

from __future__ import annotations

import fcntl
from pathlib import Path
from typing import Any

import optuna
import yaml
from pyarrow import parquet as pq
from pydantic import BaseModel, ConfigDict
from sqlalchemy.engine import make_url

# Importing the project package selects project/config as the config overlay
# (setdefault: an explicit BROADWAY_CONFIG_OVERLAY_DIR still wins), so the study
# resolves the project dataset/experiment by name.
import project  # noqa: F401
from broadway.config.loader import config_path
from broadway.config.schema import (
    DatasetContract,
    ExperimentConfig,
    HPOConfig,
    ModelHPOSpec,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _repo_path(value: str) -> Path:
    """A config-declared path, resolved against the repo root when it is relative."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _absolute_sqlite(url: str) -> str:
    """Anchor a relative SQLite study file at the repo root, so every worker shares it."""
    parsed = make_url(url)
    if parsed.get_backend_name() != "sqlite" or not parsed.database:
        return url
    database = Path(parsed.database)
    if database == Path(":memory:") or database.is_absolute():
        return url
    return str(parsed.set(database=str(REPO_ROOT / database)))


class SharedStudyStorage:
    """Create a host-local study's schema once, even when its workers start together.

    Two workers on a fresh SQLite study construct ``RDBStorage`` in the same
    instant; SQLAlchemy's ``create_all`` checks-then-creates, so one loses with
    ``table ... already exists`` and its whole budget dies before trial one. The
    schema creation is serialized with an inter-process file lock beside the
    database; the trials themselves stay concurrent afterwards.
    """

    @classmethod
    def prepare(cls, url: str) -> None:
        """Warm a SQLite study under a lock; a shared server owns its own DDL."""
        parsed = make_url(url)
        if parsed.get_backend_name() != "sqlite" or not parsed.database:
            return
        database = Path(parsed.database)
        if database == Path(":memory:"):
            return  # per-process memory DB: no other worker can share it
        database.parent.mkdir(parents=True, exist_ok=True)
        with open(database.with_name(f"{database.name}.lock"), "a", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                optuna.storages.RDBStorage(url=url).engine.dispose()
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


class DatasetIdentity(BaseModel):
    """What the preflight proved and what every worker's run record keeps."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    path: str
    rows: int
    schema_contract: str


class HpoStudyConfig(BaseModel):
    """Boundary-validated study definition (``gpu_training/config/hpo_kernel.yaml``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hpo_spec: str
    model: str
    dataset: str
    experiment: str
    storage_env: str
    local_storage_url: str
    # The adapter's name for the fixed batch dial; None for models without one.
    micro_batch_param: str | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> HpoStudyConfig:
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"study config must be a mapping: {path}")
        study = cls(**data)
        # Resolve the declared paths once: the launcher and every worker must
        # read the same files whatever their working directory is.
        return cls(
            **{
                **study.model_dump(),
                "hpo_spec": str(_repo_path(study.hpo_spec)),
                "local_storage_url": _absolute_sqlite(study.local_storage_url),
            }
        )

    def hpo_config(self) -> HPOConfig:
        """The project HPO spec — the single source for spaces, direction and metric."""
        raw = yaml.safe_load(Path(self.hpo_spec).read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("hpo"), dict):
            raise ValueError(f"HPO spec {self.hpo_spec!r} has no 'hpo' block")
        return HPOConfig(**raw["hpo"])

    def model_spec(self) -> ModelHPOSpec:
        """This study's model entry: its name and search space, straight from the spec."""
        spec = self.hpo_config()
        for candidate in spec.models:
            if candidate.name == self.model:
                return candidate
        raise ValueError(
            f"model {self.model!r} is not in the HPO spec {self.hpo_spec!r} "
            f"(models: {[model.name for model in spec.models]})"
        )

    def budget(self) -> int:
        """The model's trial budget: the spec's phase-1 per-model budget.

        The fan-out splits this across workers, so two workers run the model's
        declared trials, not twice that — the same budget the k8s per-model
        worker runs (`project/experiments/mlflow/03_optuna_worker.py`).
        """
        return self.hpo_config().initial_trials_per_model

    def dataset_contract(self) -> DatasetContract:
        return DatasetContract(**self._section("dataset", self.dataset))

    def experiment_config(self) -> ExperimentConfig:
        return ExperimentConfig(**self._section("experiment", self.experiment))

    def identity(self) -> DatasetIdentity:
        """Preflight: parse the configs, then prove the declared data is really there.

        A missing parquet or a missing declared column is a loud failure — the
        lane never runs on absent data and never skips the check.
        """
        contract = self.dataset_contract()
        experiment = self.experiment_config()
        path = _repo_path(contract.path)
        if not path.is_file():
            raise FileNotFoundError(
                f"dataset '{contract.name}' declares {contract.path!r} (resolved {path}); "
                "materialize it or overlay the config to the mounted copy "
                "(see gpu_training/README.md -> Transport)"
            )
        parquet = pq.ParquetFile(path)
        missing = sorted(set(contract.columns) - set(parquet.schema_arrow.names))
        if missing:
            raise ValueError(f"dataset {path} is missing declared column(s): {missing}")
        rows = int(parquet.metadata.num_rows)
        if rows <= 0:
            raise ValueError(f"dataset {path} has no rows")
        return DatasetIdentity(
            name=contract.name,
            path=str(path),
            rows=rows,
            schema_contract=experiment.data_source.schema_contract,
        )

    @staticmethod
    def _section(kind: str, name: str) -> dict[str, Any]:
        path = config_path(f"{kind}/{name}.yaml")
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"{kind} config must be a mapping: {path}")
        return data
