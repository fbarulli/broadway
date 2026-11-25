"""Feature engineering over four engines, with a recorded feature manifest.

Owner: platform feature engineering. ``FeatureEngineer.build`` runs an ordered
list of engine adapters and returns one frame per engine plus a manifest entry
for every produced feature (engine, params, source columns). The tracking layer
projects that manifest onto MLflow/W&B/Optuna.

Engine adapters (one class per library, same tiny interface):
``tsfresh`` (feature extraction), ``feature_engine`` (tabular transformers),
``sktime`` (series transformer), ``darts`` (series transformer — documented as
a transformer, not a feature extractor).

The libraries are optional dependencies, imported lazily inside each adapter so
the core ``broadway`` import stays light. Every knob comes from the plan
(``FeatureEngineeringConfig``), resolved by default through the platform config
tree with the project overlay winning when one is configured; this module knows
no dataset or column names.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

import pandas as pd
import yaml
from pandas.api.types import is_numeric_dtype
from pydantic import BaseModel, ConfigDict, Field, field_validator

from broadway.features.plan import ColumnPlan
from broadway.observability.profiling import phase, timed

logger = logging.getLogger(__name__)

FEATURE_ENGINEERING_CONFIG_NAME = "feature_engineering.yaml"
SERIES_KEY = "__broadway_series_key__"

__all__ = [
    "FEATURE_ENGINEERING_CONFIG_NAME",
    "DartsEngineConfig",
    "DartsTransformer",
    "EngineConfig",
    "EngineOutput",
    "FeatureEngineAdapter",
    "FeatureEngineTabularConfig",
    "FeatureEngineTransformerConfig",
    "FeatureEngineer",
    "FeatureEngineeringConfig",
    "FeatureEngineeringResult",
    "FeatureManifestEntry",
    "SktimeEngineConfig",
    "SktimeTransformer",
    "TsfreshEngineConfig",
]


# --------------------------------------------------------------------------- #
# Config (pydantic, parsed at the YAML boundary)
# --------------------------------------------------------------------------- #
class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


TsfreshFeatureSet = Literal["minimal", "efficient", "comprehensive"]
SktimeTransformer = Literal["SummaryTransformer"]
DartsTransformer = Literal["Scaler", "BoxCox", "Diff"]


class TsfreshEngineConfig(_FrozenModel):
    """tsfresh feature-extraction settings."""

    kind: Literal["tsfresh"]
    id: str = Field(min_length=1)
    feature_set: TsfreshFeatureSet
    n_jobs: int = Field(ge=1)


class FeatureEngineTransformerConfig(_FrozenModel):
    """One feature-engine transformer step: allowlisted class + its params."""

    kind: Literal["Winsorizer", "MeanMedianImputer"]
    params: dict[str, Any]


class FeatureEngineTabularConfig(_FrozenModel):
    """An ordered feature-engine pipeline applied to tabular columns."""

    kind: Literal["feature_engine"]
    id: str = Field(min_length=1)
    transformers: list[FeatureEngineTransformerConfig] = Field(min_length=1)


class SktimeEngineConfig(_FrozenModel):
    """sktime series-transformer settings (one row per instance)."""

    kind: Literal["sktime"]
    id: str = Field(min_length=1)
    transformer: SktimeTransformer
    params: dict[str, Any]


class DartsEngineConfig(_FrozenModel):
    """darts series-transformer settings; a transformer, not an extractor."""

    kind: Literal["darts"]
    id: str = Field(min_length=1)
    transformer: DartsTransformer
    freq: str | None
    params: dict[str, Any]


EngineConfig = Annotated[
    TsfreshEngineConfig | FeatureEngineTabularConfig | SktimeEngineConfig | DartsEngineConfig,
    Field(discriminator="kind"),
]


class FeatureEngineeringConfig(_FrozenModel):
    """The whole engine plan, loaded from the resolved feature-engineering YAML."""

    time_column: str = Field(min_length=1)
    id_columns: list[str]
    engines: list[EngineConfig] = Field(min_length=1)

    @field_validator("engines")
    @classmethod
    def _unique_engine_ids(cls, engines: list[EngineConfig]) -> list[EngineConfig]:
        seen: set[str] = set()
        for engine in engines:
            if engine.id in seen:
                raise ValueError(f"duplicate engine id: {engine.id}")
            seen.add(engine.id)
        return engines

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> FeatureEngineeringConfig:
        """Parse the engine plan; ``path`` wins, otherwise overlay resolution decides."""
        if path is None:
            # Lazy: loader imports the full config schema (training stack), which
            # must not ride along on every core feature-engineering import.
            from broadway.config.loader import config_path

            path = config_path(FEATURE_ENGINEERING_CONFIG_NAME)
        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"feature engineering config must be a mapping: {path}")
        return cls.model_validate(data)


# --------------------------------------------------------------------------- #
# Result: per-engine frames + the manifest the tracking layer projects
# --------------------------------------------------------------------------- #
class FeatureManifestEntry(_FrozenModel):
    """One produced feature, with the provenance the tracking layer records."""

    name: str
    engine: str
    kind: str
    params: dict[str, Any]
    source_columns: list[str]


@dataclass(frozen=True)
class EngineOutput:
    """A single adapter's frame and its manifest entries."""

    frame: pd.DataFrame
    features: list[FeatureManifestEntry]


@dataclass(frozen=True)
class FeatureEngineeringResult:
    """Per-engine frames keyed by engine id, plus the flat feature manifest."""

    frames: dict[str, pd.DataFrame]
    manifest: list[FeatureManifestEntry]


# --------------------------------------------------------------------------- #
# Adapter interface
# --------------------------------------------------------------------------- #
class FeatureEngineAdapter(ABC):
    """Tiny common interface: build one frame + manifest entries from a frame."""

    kind: ClassVar[str]

    def __init__(self, config: FeatureEngineeringConfig, engine: EngineConfig) -> None:
        self._config = config
        self._engine_id = engine.id

    @abstractmethod
    def transform(self, frame: pd.DataFrame, plan: ColumnPlan) -> EngineOutput:
        """Run this engine over the plan's projection of ``frame``."""

    def _entry(
        self, name: str, params: dict[str, Any], source_columns: list[str]
    ) -> FeatureManifestEntry:
        return FeatureManifestEntry(
            name=name,
            engine=self._engine_id,
            kind=self.kind,
            params=params,
            source_columns=source_columns,
        )

    def _require_columns(self, frame: pd.DataFrame, columns: list[str]) -> None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"{self.kind}: missing columns {missing}")

    def _value_columns(self, frame: pd.DataFrame, plan: ColumnPlan) -> list[str]:
        columns = self._value_names(frame, plan)
        if not columns:
            raise ValueError(f"{self.kind}: no numeric value columns to transform")
        projection = ColumnPlan.from_frame_columns(frame, columns)
        projected, _ = projection.apply(frame)
        return list(projected.columns)

    def _value_names(self, frame: pd.DataFrame, plan: ColumnPlan) -> list[str]:
        """The plan's numeric projection, excluding the time and id columns."""
        excluded = {self._config.time_column, *self._config.id_columns}
        return [
            column
            for column in plan.names
            if column in frame.columns
            and column not in excluded
            and is_numeric_dtype(frame[column])
        ]

    def _with_series_key(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Add one instance key column and return the key -> id-column mapping.

        tsfresh and sktime need a single instance dimension; the platform's id
        selection may be empty or composite, so a synthetic key carries it.
        """
        work = frame.copy()
        id_columns = self._config.id_columns
        if not id_columns:
            work[SERIES_KEY] = 0
        elif len(id_columns) == 1:
            work[SERIES_KEY] = work[id_columns[0]]
        else:
            work[SERIES_KEY] = work[id_columns].astype(str).agg("|".join, axis=1)
        mapping = work[[SERIES_KEY, *id_columns]].drop_duplicates(SERIES_KEY)
        mapping = mapping.sort_values(SERIES_KEY, kind="mergesort").reset_index(drop=True)
        return work, mapping

    def _restore_ids(self, output: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
        restored = mapping.merge(output, on=SERIES_KEY, how="left")
        return restored.drop(columns=SERIES_KEY)


def _ordered_unique(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


# --------------------------------------------------------------------------- #
# tsfresh — feature extraction
# --------------------------------------------------------------------------- #
class TsfreshAdapter(FeatureEngineAdapter):
    """Extract time-series features with ``tsfresh.extract_features``."""

    kind = "tsfresh"

    _FEATURE_SETS: ClassVar[dict[str, str]] = {
        "minimal": "MinimalFCParameters",
        "efficient": "EfficientFCParameters",
        "comprehensive": "ComprehensiveFCParameters",
    }

    def __init__(self, config: FeatureEngineeringConfig, engine: TsfreshEngineConfig) -> None:
        super().__init__(config, engine)
        self._spec = engine

    def transform(self, frame: pd.DataFrame, plan: ColumnPlan) -> EngineOutput:
        from tsfresh import extract_features
        from tsfresh.feature_extraction import settings as tsfresh_settings

        value_columns = self._value_columns(frame, plan)
        work, mapping = self._with_series_key(frame)
        work = work[[SERIES_KEY, self._config.time_column, *value_columns]]
        work = work.sort_values([SERIES_KEY, self._config.time_column], kind="mergesort")

        settings_class = getattr(tsfresh_settings, self._FEATURE_SETS[self._spec.feature_set])
        extracted = extract_features(
            work,
            column_id=SERIES_KEY,
            column_sort=self._config.time_column,
            default_fc_parameters=settings_class(),
            n_jobs=self._spec.n_jobs,
            disable_progressbar=True,
            show_warnings=False,
        )
        extracted = extracted.reset_index().rename(columns={"index": SERIES_KEY})
        output = self._restore_ids(extracted, mapping)

        params = self._spec.model_dump(exclude={"id", "kind"}, mode="json")
        feature_columns = [column for column in output.columns if column not in self._config.id_columns]
        return EngineOutput(
            frame=output,
            features=[self._entry(column, params, value_columns) for column in feature_columns],
        )


# --------------------------------------------------------------------------- #
# feature-engine — tabular transformers
# --------------------------------------------------------------------------- #
class FeatureEngineTabularAdapter(FeatureEngineAdapter):
    """Apply an ordered feature-engine transformer pipeline to tabular columns."""

    kind = "feature_engine"

    _TRANSFORMERS: ClassVar[dict[str, str]] = {
        "Winsorizer": "feature_engine.outliers",
        "MeanMedianImputer": "feature_engine.imputation",
    }

    def __init__(
        self, config: FeatureEngineeringConfig, engine: FeatureEngineTabularConfig
    ) -> None:
        super().__init__(config, engine)
        self._spec = engine

    def _build_transformer(
        self, step: FeatureEngineTransformerConfig
    ) -> Any:
        from importlib import import_module

        module = import_module(self._TRANSFORMERS[step.kind])
        transformer_class = getattr(module, step.kind)
        return transformer_class(**step.params)

    def transform(self, frame: pd.DataFrame, plan: ColumnPlan) -> EngineOutput:
        used_columns = self._value_columns(frame, plan)

        work = frame.copy()
        for step in self._spec.transformers:
            transformer = self._build_transformer(step)
            transformed = transformer.fit_transform(work[used_columns])
            work[used_columns] = transformed.to_numpy()

        id_columns = self._config.id_columns
        output_columns = _ordered_unique(
            [*id_columns, self._config.time_column, *used_columns]
        )
        output = work[output_columns].copy()

        steps_by_column: dict[str, list[dict[str, Any]]] = {column: [] for column in used_columns}
        for step in self._spec.transformers:
            for column in used_columns:
                steps_by_column[column].append({"transformer": step.kind, **step.params})
        return EngineOutput(
            frame=output,
            features=[
                self._entry(column, {"steps": steps_by_column[column]}, [column])
                for column in used_columns
            ],
        )


# --------------------------------------------------------------------------- #
# sktime — series transformer
# --------------------------------------------------------------------------- #
class SktimeAdapter(FeatureEngineAdapter):
    """Summarize each instance with a sktime series transformer."""

    kind = "sktime"

    def __init__(self, config: FeatureEngineeringConfig, engine: SktimeEngineConfig) -> None:
        super().__init__(config, engine)
        self._spec = engine

    def _build_transformer(self) -> Any:
        from sktime.transformations.series.summarize import SummaryTransformer

        # SummaryTransformer is the only allowlisted sktime transformer.
        return SummaryTransformer(**self._spec.params)

    def _source_columns(self, feature: str, value_columns: list[str]) -> list[str]:
        for column in value_columns:
            if feature.startswith(f"{column}__"):
                return [column]
        # Single-variable summaries drop the prefix; multi-variable ones keep it.
        return value_columns if len(value_columns) > 1 else [value_columns[0]]

    def transform(self, frame: pd.DataFrame, plan: ColumnPlan) -> EngineOutput:
        value_columns = self._value_columns(frame, plan)
        work, mapping = self._with_series_key(frame)
        work = work[[SERIES_KEY, self._config.time_column, *value_columns]]
        work = work.sort_values([SERIES_KEY, self._config.time_column], kind="mergesort")
        indexed = work.set_index([SERIES_KEY, self._config.time_column])

        transformer = self._build_transformer()
        summarized = transformer.fit_transform(indexed[value_columns])
        output = self._restore_ids(summarized.reset_index(), mapping)

        params = {"transformer": self._spec.transformer, **self._spec.params}
        feature_columns = [column for column in output.columns if column not in self._config.id_columns]
        return EngineOutput(
            frame=output,
            features=[
                self._entry(column, params, self._source_columns(column, value_columns))
                for column in feature_columns
            ],
        )


# --------------------------------------------------------------------------- #
# darts — series transformer (documented as a transformer, not an extractor)
# --------------------------------------------------------------------------- #
class DartsAdapter(FeatureEngineAdapter):
    """Transform each instance's series with darts; one output column per value."""

    kind = "darts"

    _SCALERS: ClassVar[frozenset[str]] = frozenset(
        {"MinMaxScaler", "StandardScaler", "RobustScaler"}
    )

    def __init__(self, config: FeatureEngineeringConfig, engine: DartsEngineConfig) -> None:
        super().__init__(config, engine)
        self._spec = engine

    def _build_transformer(self) -> Any:
        from darts.dataprocessing.transformers import BoxCox, Diff, Scaler

        params = dict(self._spec.params)
        if self._spec.transformer == "Scaler":
            from sklearn import preprocessing

            scaler_name = str(params.pop("scaler"))
            if scaler_name not in self._SCALERS:
                raise ValueError(f"darts: unsupported scaler {scaler_name!r}")
            scaler_class = getattr(preprocessing, scaler_name)
            return Scaler(scaler=scaler_class(), **params)
        if self._spec.transformer == "BoxCox":
            return BoxCox(**params)
        return Diff(**params)

    def transform(self, frame: pd.DataFrame, plan: ColumnPlan) -> EngineOutput:
        from darts import TimeSeries

        value_columns = self._value_columns(frame, plan)
        work, mapping = self._with_series_key(frame)
        work = work[[SERIES_KEY, self._config.time_column, *value_columns]]
        work = work.sort_values([SERIES_KEY, self._config.time_column], kind="mergesort")

        pieces: list[pd.DataFrame] = []
        for key, group in work.groupby(SERIES_KEY, sort=True):
            series = TimeSeries.from_dataframe(
                group[[self._config.time_column, *value_columns]],
                time_col=self._config.time_column,
                value_cols=value_columns,
                freq=self._spec.freq,
                fill_missing_dates=self._spec.freq is not None,
            )
            transformed = self._build_transformer().fit_transform(series).to_dataframe()
            transformed.index.name = self._config.time_column
            transformed.insert(0, SERIES_KEY, key)
            pieces.append(transformed.reset_index())
        combined = pd.concat(pieces, ignore_index=True)
        output = self._restore_ids(combined, mapping)

        params = {"transformer": self._spec.transformer, **self._spec.params}
        return EngineOutput(
            frame=output,
            features=[
                self._entry(column, params, [column])
                for column in value_columns
            ],
        )


# --------------------------------------------------------------------------- #
# Facade / factory
# --------------------------------------------------------------------------- #
class FeatureEngineer:
    """Run the configured engine adapters and record the manifest."""

    _ADAPTERS: ClassVar[dict[str, Callable[..., FeatureEngineAdapter]]] = {
        "tsfresh": TsfreshAdapter,
        "feature_engine": FeatureEngineTabularAdapter,
        "sktime": SktimeAdapter,
        "darts": DartsAdapter,
    }

    @classmethod
    def build(
        cls, frame: pd.DataFrame, config: FeatureEngineeringConfig, plan: ColumnPlan
    ) -> FeatureEngineeringResult:
        """Build every engine frame and its manifest from the plan's projection."""
        cls._require_input_columns(frame, config)
        ordered = cls._sort(frame, config)
        frames: dict[str, pd.DataFrame] = {}
        manifest: list[FeatureManifestEntry] = []
        for engine in config.engines:
            adapter = cls._ADAPTERS[engine.kind](config, engine)
            with phase(f"engine:{engine.id}"), timed() as elapsed:
                output = adapter.transform(ordered, plan)
            duration = elapsed()
            logger.info(
                "feature engine %s (%s): %d features in %.3fs",
                engine.id,
                engine.kind,
                len(output.features),
                duration,
            )
            frames[engine.id] = output.frame
            manifest.extend(output.features)
        return FeatureEngineeringResult(frames=frames, manifest=manifest)

    @staticmethod
    def _require_input_columns(
        frame: pd.DataFrame, config: FeatureEngineeringConfig
    ) -> None:
        required = _ordered_unique([config.time_column, *config.id_columns])
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"feature engineering: missing columns {missing}")

    @staticmethod
    def _sort(
        frame: pd.DataFrame, config: FeatureEngineeringConfig
    ) -> pd.DataFrame:
        keys = [*config.id_columns, config.time_column]
        return frame.sort_values(keys, kind="mergesort").reset_index(drop=True)
