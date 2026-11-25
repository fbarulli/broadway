"""The column plan: one declared add/drop path for the model surface.

A ``ColumnPlan`` is the ordered column algebra for one experiment. It records
every source column, derived feature, and encoded output in the exact order the
features step produces them, exposes ``apply`` as the single add/drop execution
point, and returns a ``ColumnManifest`` describing what was projected.

Execution stays read-side: encoders are fitted and persisted by the features
step, so ``apply`` validates encoder inputs and consumes their output columns
instead of refitting on the frame it projects (a refit would leak the
validation frame's target statistics). The experiment declares its plan as a
``module:callable`` reference that the loader resolves against the dataset
contract.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from broadway.config.schema import DatasetContract, DerivedFeature, ExperimentConfig
from broadway.features.builders import build_derived, builder_dtype, load_custom_builders

logger = logging.getLogger(__name__)

# Decision 5 pin: numeric-category matching over the OBSERVED runtime dtypes
# ({int32, int64, float64}); object and datetime64 are excluded by the guard.
_NUMERIC_CATEGORY_DTYPES = frozenset({"int32", "int64", "float64"})

_CATEGORICAL_MESSAGE = (
    "categorical column '{name}' has no preprocessing step and no "
    "numeric-selector fallback applies — add a preprocessing step claiming "
    "it, or repoint the schema contract so it is not eligible"
)


class ColumnOp(BaseModel):
    """One ordered column operation: a source, a derivation, or an encoding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    op: Literal["source", "derive", "encode"]
    name: str
    dtype: str
    sources: tuple[str, ...] = ()
    func: str | None = None
    encoder: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_shape(self) -> ColumnOp:
        if self.op == "derive" and (self.func is None or len(self.sources) != 1):
            raise ValueError(f"derived column '{self.name}' requires one func and one source")
        if self.op == "encode" and (self.encoder is None or not self.sources):
            raise ValueError(f"encoded column '{self.name}' requires one encoder and inputs")
        return self


class ColumnManifestEntry(BaseModel):
    """One projected column's provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    dtype: str
    op: Literal["source", "derive", "encode"]
    sources: tuple[str, ...]
    params: dict[str, Any]


class ColumnManifest(BaseModel):
    """The applied plan: ordered entries, the target, and dropped columns."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entries: tuple[ColumnManifestEntry, ...]
    target: str | None
    dropped: tuple[str, ...]


class ColumnPlan(BaseModel):
    """Immutable ordered column algebra for one experiment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target: str | None
    datetime_column: str | None
    columns: tuple[ColumnOp, ...]
    claimed: frozenset[str] = frozenset()
    builder_module: str | None = None
    builder_params: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_names(self) -> ColumnPlan:
        names = [op.name for op in self.columns]
        if len(set(names)) != len(names):
            raise ValueError(f"column plan contains duplicate names: {names}")
        return self

    @property
    def names(self) -> tuple[str, ...]:
        """Ordered projected column names."""
        return tuple(op.name for op in self.columns)

    @classmethod
    def from_config(cls, dataset: DatasetContract, experiment: ExperimentConfig) -> ColumnPlan:
        """Claim the experiment's resolved plan for the bound preprocessing recipe.

        The loader attached ``experiment.plan`` from its ``feature_plan`` code
        reference; this method only adds the recipe's categorical claims and
        re-runs the Decision-5 guard.
        """
        plan = experiment.plan.model_copy(
            update={
                "claimed": frozenset(
                    column for step in experiment.preprocessing for column in step.columns
                )
            }
        )
        plan._assert_claimed(
            {op.name: op.dtype for op in plan.columns if op.op == "source"}, plan.claimed
        )
        return plan

    @classmethod
    def from_declaration(
        cls,
        dataset: DatasetContract,
        *,
        include: Sequence[str] = (),
        derived: Sequence[DerivedFeature] = (),
        encodings: Sequence[ColumnOp] = (),
        builder_module: str | None = None,
        builder_params: Mapping[str, str] | None = None,
    ) -> ColumnPlan:
        """Build a plan in production order: sources, derived, then encodings."""
        ops = [
            ColumnOp(op="source", name=name, dtype=dataset.columns[name].dtype)
            for name in include
            if name != dataset.target and name in dataset.columns
        ]
        ops.extend(
            ColumnOp(
                op="derive",
                name=feature.name,
                dtype=builder_dtype(feature.func),
                sources=(feature.source,),
                func=feature.func,
            )
            for feature in derived
        )
        ops.extend(encodings)
        return cls(
            target=dataset.target,
            datetime_column=dataset.datetime_column,
            columns=tuple(ops),
            builder_module=builder_module,
            builder_params=dict(builder_params) if builder_params else {},
        )

    @classmethod
    def from_frame_columns(
        cls, frame: pd.DataFrame, names: list[str], *, target: str | None = None
    ) -> ColumnPlan:
        """Build a source-only plan for an explicit projection.

        An explicit column list is a claim by construction, so it satisfies the
        categorical guard while selecting exactly the requested columns.
        """
        ops = tuple(
            ColumnOp(op="source", name=name, dtype=str(frame[name].dtype)) for name in names
        )
        return cls(target=target, datetime_column=None, columns=ops, claimed=frozenset(names))

    def add(self, op: ColumnOp) -> ColumnPlan:
        """Return a new plan with ``op`` appended; never mutate this plan."""
        if op.name in self.names:
            raise ValueError(f"column plan already declares '{op.name}'")
        return self.model_copy(update={"columns": (*self.columns, op)})

    def drop(self, *names: str) -> ColumnPlan:
        """Return a new plan without the named columns; never mutate this plan."""
        unknown = sorted(set(names) - set(self.names))
        if unknown:
            raise ValueError(f"column plan does not declare: {unknown}")
        dropped = set(names)
        return self.model_copy(
            update={"columns": tuple(op for op in self.columns if op.name not in dropped)}
        )

    def apply(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, ColumnManifest]:
        """Compute missing derivations, verify encoder inputs, then project.

        Existing derived/encoded columns are authoritative: the features step
        produced them, so they are never recomputed. A missing encoder output
        fails loud in the projection rather than being refit here.
        """
        working = frame.copy()
        self._require_inputs(working)
        for op in self.columns:
            if op.op == "derive" and op.name not in working.columns:
                working = self._derive(working, op)
        self._assert_claimed(
            {
                op.name: str(working[op.name].dtype)
                for op in self.columns
                if op.op == "source" and op.name in working.columns
            },
            self.claimed,
        )
        missing = [name for name in self.names if name not in working.columns]
        if missing:
            raise ValueError(f"column plan: declared column(s) missing from frame: {missing}")
        dropped = tuple(
            name for name in frame.columns if name not in self.names and name != self.target
        )
        projected = working[list(self.names)]
        return projected, self._manifest(dropped)

    def _require_inputs(self, frame: pd.DataFrame) -> None:
        """Fail loud when any source or encoder input is absent."""
        available = set(frame.columns)
        for op in self.columns:
            if op.op == "source":
                missing = [] if op.name in available else [op.name]
                label = "source"
            else:
                missing = [source for source in op.sources if source not in available]
                label = f"{op.op} input"
            if missing:
                raise ValueError(f"column plan: {label} missing for '{op.name}': {missing}")
            available.add(op.name)

    def _derive(self, frame: pd.DataFrame, op: ColumnOp) -> pd.DataFrame:
        """Compute one missing derived column through the shared builder registry."""
        feature = DerivedFeature(name=op.name, func=op.func or "", source=op.sources[0])
        builder_kwargs: dict[str, Any] = dict(self.builder_params)
        built = build_derived(
            frame,
            [feature],
            self.target or "",
            extra_builders=load_custom_builders(self.builder_module),
            **builder_kwargs,
        )
        if str(built[op.name].dtype) != op.dtype:
            built[op.name] = built[op.name].astype(op.dtype)
        return built

    @staticmethod
    def _assert_claimed(dtypes: dict[str, str], claimed: frozenset[str]) -> None:
        """Reject an unclaimed categorical source column, naming it."""
        for name, dtype in dtypes.items():
            if name in claimed or dtype in _NUMERIC_CATEGORY_DTYPES:
                continue
            raise ValueError(_CATEGORICAL_MESSAGE.format(name=name))

    def _manifest(self, dropped: tuple[str, ...]) -> ColumnManifest:
        """Describe the projected columns in plan order."""
        return ColumnManifest(
            entries=tuple(
                ColumnManifestEntry(
                    name=op.name,
                    dtype=op.dtype,
                    op=op.op,
                    sources=op.sources,
                    params=op.params,
                )
                for op in self.columns
            ),
            target=self.target,
            dropped=dropped,
        )


def resolve_plan(reference: str, dataset: DatasetContract) -> ColumnPlan:
    """Resolve a ``module:callable`` plan reference against the dataset contract.

    Every failure class is loud: malformed reference, unimportable module,
    missing attribute, non-callable target, non-``ColumnPlan`` return value, or
    a plan bound to a different target than the dataset contract.
    """
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(
            f"feature_plan '{reference}' must be 'module:callable' "
            "(e.g. 'broadway.features.plans:baseline')"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        logger.exception("feature_plan module import failed: %s", module_name)
        raise ValueError(
            f"feature_plan '{reference}' module '{module_name}' could not be imported: {exc}"
        ) from exc
    factory = getattr(module, attribute, None)
    if factory is None:
        raise ValueError(
            f"feature_plan '{reference}' attribute '{attribute}' not found in '{module_name}'"
        )
    if not callable(factory):
        raise ValueError(f"feature_plan '{reference}' attribute '{attribute}' is not callable")
    plan = factory(dataset)
    if not isinstance(plan, ColumnPlan):
        raise TypeError(
            f"feature_plan '{reference}' must return a ColumnPlan, got {type(plan).__name__}"
        )
    if plan.target != dataset.target:
        raise ValueError(
            f"feature_plan '{reference}' targets '{plan.target}' but the dataset "
            f"contract targets '{dataset.target}'"
        )
    return plan
