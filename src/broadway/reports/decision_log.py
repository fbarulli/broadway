"""MLflow projection of decision records — a view of the record, never a truth.

The decision record (``<stem>_decision.json``) is the source of truth; MLflow
only *shows* it: one run per (invocation, step) carrying the record's tags,
params and metrics plus the evidence artifacts. Projection is a fail-soft
side channel — a missing tracking URI or an unreachable server records why and
never fails the step. The tracking URI is read from the environment config
(``EnvironmentConfig.mlflow_tracking_uri``, SSOT); an empty value is the
opt-out, so no second MLflow config exists here.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import mlflow
from mlflow.entities import Run

from broadway.observability.context import RUN_ID_ENV
from broadway.reports.decision import Decision
from broadway.training.mlflow_utils import log_metrics, log_params, setup_mlflow

logger = logging.getLogger(__name__)

# One experiment for every step's decision projection (the label MLflow groups
# the runs under; the record itself stays on disk next to the figure).
EXPERIMENT = "broadway-decisions"

ENVIRONMENT_ENV = "BROADWAY_ENVIRONMENT"

STEP_TAG = "step"
RUN_ID_TAG = "broadway_run_id"
RUN_ID_PROVENANCE_KEY = "mlflow_run_id"

# Provenance keys that are projection parameters when the step recorded them.
PROVENANCE_PARAM_KEYS: tuple[str, ...] = ("seed", "dataset", "config", "commit")

DISABLED_REASON = "no mlflow_tracking_uri in the environment config"


class DecisionLog:
    """Project a step's decision record into MLflow (injectable into StepOutput)."""

    def __init__(self, *, tracking_uri: str | None, experiment: str = EXPERIMENT) -> None:
        self._tracking_uri = (tracking_uri or "").strip() or None
        self._experiment = experiment
        self._reason: str | None = None

    @classmethod
    def from_config(cls, environment: str | None = None) -> DecisionLog:
        """Build from the environment config; an unreadable config disables projection."""
        # Lazy: the config loader must not ride along on every reports import.
        from broadway.config.loader import DEFAULT_ENVIRONMENT, load_environment

        name = environment or os.getenv(ENVIRONMENT_ENV) or DEFAULT_ENVIRONMENT
        try:
            config = load_environment(name)
        except (FileNotFoundError, ValueError) as error:
            log = cls(tracking_uri=None)
            log._reason = f"environment config {name!r} unreadable: {error}"
            logger.warning("mlflow projection disabled — %s", log._reason)
            return log
        return cls(tracking_uri=config.mlflow_tracking_uri)

    @property
    def reason(self) -> str | None:
        """Why projection was disabled or last failed (None while healthy)."""
        return self._reason

    def enabled(self) -> bool:
        """True when a tracking URI is configured (an empty URI is the opt-out)."""
        if self._tracking_uri is not None:
            return True
        if self._reason is None:
            self._reason = DISABLED_REASON
            logger.info("mlflow projection disabled — %s", DISABLED_REASON)
        return False

    def log(
        self,
        *,
        stem: str,
        decision: Decision,
        figure: Path,
        tables: Sequence[tuple[str, Path]],
        params: Mapping[str, str | int | float],
        record_path: Path,
    ) -> str | None:
        """Project one decision; return the run id, or None when disabled or failed."""
        if not self.enabled():
            return None
        try:
            return self._project(stem, decision, figure, tables, params, record_path)
        except Exception as error:
            self._reason = f"mlflow projection failed: {type(error).__name__}: {error}"
            logger.exception(
                "mlflow projection failed for %r at %s — the step continues",
                stem,
                self._tracking_uri,
            )
            return None

    def _project(
        self,
        stem: str,
        decision: Decision,
        figure: Path,
        tables: Sequence[tuple[str, Path]],
        params: Mapping[str, str | int | float],
        record_path: Path,
    ) -> str:
        assert self._tracking_uri is not None  # enabled() guarantees a URI
        setup_mlflow(self._tracking_uri, self._experiment)
        with self._run(stem, decision) as run:
            run_id = run.info.run_id
            log_params(self._params(decision, params))
            log_metrics(self._metrics(decision))
            self._stamp_record(record_path, decision, run_id)
            self._log_artifacts(figure, tables, record_path)
        logger.info(
            "mlflow projection — step %s -> run %s at %s",
            stem,
            run_id,
            self._tracking_uri,
        )
        return run_id

    @contextmanager
    def _run(self, stem: str, decision: Decision) -> Iterator[Run]:
        """Start a run, or reuse this invocation's run for the step when it exists."""
        existing = self._existing_run(stem)
        name = None if existing is not None else stem
        with mlflow.start_run(
            run_id=existing, run_name=name, tags=self._tags(stem, decision)
        ) as run:
            yield run

    def _existing_run(self, stem: str) -> str | None:
        """The run this invocation already projected *stem* into, when one exists."""
        invocation = os.getenv(RUN_ID_ENV)
        if not invocation:
            return None
        selector = f"tags.{STEP_TAG} = '{stem}' and tags.{RUN_ID_TAG} = '{invocation}'"
        found = mlflow.search_runs(
            experiment_names=[self._experiment],
            filter_string=selector,
            max_results=1,
            output_format="list",
        )
        return found[0].info.run_id if found else None

    @staticmethod
    def _tags(stem: str, decision: Decision) -> dict[str, str]:
        tags = {
            STEP_TAG: stem,
            "verdict": decision.verdict,
            "owner": decision.owner,
            "action": decision.action,
            "direction": decision.direction,
        }
        invocation = os.getenv(RUN_ID_ENV)
        if invocation:
            tags[RUN_ID_TAG] = invocation
        return tags

    @staticmethod
    def _params(
        decision: Decision, extra: Mapping[str, str | int | float]
    ) -> dict[str, str | int | float]:
        params: dict[str, str | int | float] = {
            "metric": decision.metric,
            "threshold": decision.threshold,
            "sample_size": decision.sample_size,
            **{
                key: decision.evidence.provenance[key]
                for key in PROVENANCE_PARAM_KEYS
                if key in decision.evidence.provenance
            },
        }
        if decision.null_threshold is not None:
            params["null_threshold"] = decision.null_threshold
        params.update(extra)
        return params

    @staticmethod
    def _metrics(decision: Decision) -> dict[str, float]:
        metrics = {
            "value": decision.value,
            "interval_low": decision.interval_low,
            "interval_high": decision.interval_high,
            "threshold": decision.threshold,
        }
        if decision.delta is not None:
            metrics["delta"] = decision.delta
        return metrics

    @staticmethod
    def _stamp_record(record_path: Path, decision: Decision, run_id: str) -> None:
        """Write the run id into the record's provenance (the record stays the truth)."""
        evidence = decision.evidence.model_copy(
            update={
                "provenance": {
                    **decision.evidence.provenance,
                    RUN_ID_PROVENANCE_KEY: run_id,
                }
            }
        )
        projected = decision.model_copy(update={"evidence": evidence})
        record_path.write_text(projected.model_dump_json(indent=2), encoding="utf-8")

    @staticmethod
    def _log_artifacts(figure: Path, tables: Sequence[tuple[str, Path]], record_path: Path) -> None:
        for path in (figure, record_path, *(path for _, path in tables)):
            mlflow.log_artifact(str(path))
