"""FeaturePipeline dataclass — fit on train, transform on any DataFrame.

The plan is the single declaration: derived operations and encoded outputs are
read from ``ColumnPlan.columns`` in declared order. Encoders are still fitted
and persisted by the features step (Phase 3 owns any read-side execution); the
plan only supplies the declarations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from broadway.config.schema import DerivedFeature
from broadway.features.builders import build_derived, load_custom_builders
from broadway.features.plan import ColumnOp, ColumnPlan
from broadway.features.transformers import FrequencyEncoding


def _derived_feature(op: ColumnOp) -> DerivedFeature:
    return DerivedFeature(
        name=op.name, func=op.func or "", source=op.sources[0] if op.sources else ""
    )


@dataclass
class FeaturePipeline:
    _encoders: list[tuple[str, FrequencyEncoding]] = field(default_factory=list)

    def fit(self, df: pd.DataFrame, plan: ColumnPlan, target: str) -> FeaturePipeline:
        self._encoders = []
        for op in plan.columns:
            if op.op != "encode":
                continue
            if op.encoder != "frequency":
                raise ValueError(
                    f"column plan: unknown encoder {op.encoder!r} for '{op.name}' "
                    "(supported: frequency)"
                )
            encoder = FrequencyEncoding(
                columns=list(op.sources),
                feature_name=op.name,
                normalize=bool(op.params.get("normalize", True)),
            )
            self._encoders.append((op.name, encoder.fit(df)))
        return self

    def transform(
        self, df: pd.DataFrame, plan: ColumnPlan, target: str, freq_fill: float
    ) -> pd.DataFrame:
        result = df.copy()
        plan_kwargs: dict[str, Any] = dict(plan.builder_params)
        for op in plan.columns:
            if op.op != "derive":
                continue
            builder_kwargs: dict[str, Any] = {
                **plan_kwargs,
                "columns": dict(op.params.get("columns") or {}),
            }
            result = build_derived(
                result,
                [_derived_feature(op)],
                target,
                extra_builders=load_custom_builders(plan.builder_module),
                **builder_kwargs,
            )
            result[op.name] = result[op.name].astype(op.dtype)
        for name, encoder in self._encoders:
            result = encoder.transform(result, fill=freq_fill)
            declared = next(column for column in plan.columns if column.name == name)
            if str(result[name].dtype) != declared.dtype:
                result[name] = result[name].astype(declared.dtype)
        return result
