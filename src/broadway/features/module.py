"""Fit FeaturePipeline on train → transform train + test → save pipeline."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pandas as pd

from broadway.config.schema import PipelineConfig
from broadway.data.loader import ProcessedStore
from broadway.features.generic import build_generic_feature_specs
from broadway.features.pipeline import FeaturePipeline
from broadway.features.plan import ColumnPlan
from broadway.features.schema import build_engineered_schema
from broadway.lineage.ids import node_id
from broadway.lineage.models import TransformAudit
from broadway.lineage.records import enforce_drop_fraction, write_record
from broadway.reports.decision import Decision, DecisionRule, Evidence, SegmentEvidence
from broadway.reports.step_output import StepOutput

logger = logging.getLogger(__name__)


def _load_split(cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    assert cfg.etl is not None
    out_dir = ProcessedStore.from_config(cfg).directory
    train = pd.read_parquet(out_dir / cfg.etl.train_file)
    val_path = out_dir / cfg.etl.val_file
    val = pd.read_parquet(val_path) if val_path.exists() else None
    return train, val


def _plan_ordered(frame: pd.DataFrame, plan: ColumnPlan) -> pd.DataFrame:
    """Put the plan's columns first, in plan order; keep extra columns after.

    The read contract (``engineered_schema_for``) is ordered: the persisted
    feature file must carry the model surface in plan order or a real
    train/evaluate read fails loud. ``FeaturePipeline.transform`` appends
    derived/encoded columns to the joined frame, so the write reorders here —
    extras (the target, joined columns) are tolerated and follow.
    """
    declared = list(plan.names)
    extras = [name for name in frame.columns if name not in declared]
    return frame.loc[:, declared + extras]


class FeaturesDecision:
    """Publish the feature-surface decision: keep, drop, transform.

    The verdict is about row alignment (the feature step must never change the
    row count); the keep/drop/transform counts are the evidence behind it, so a
    reader sees which engineered columns entered the model.
    """

    def __init__(
        self,
        cfg: PipelineConfig,
        audit: TransformAudit,
        plan_names: list[str],
        val_rows_in: int,
        val_rows_out: int | None,
        half_sample_loss: float,
        half_sample_rows: int,
    ) -> None:
        assert cfg.dataset is not None and cfg.features is not None
        self._cfg = cfg
        self._audit = audit
        self._plan_names = plan_names
        self._val_rows_in = val_rows_in
        self._val_rows_out = val_rows_out
        self._half_sample_loss = half_sample_loss
        self._half_sample_rows = half_sample_rows

    def rule(self) -> DecisionRule:
        assert self._cfg.dataset is not None and self._cfg.features is not None
        return DecisionRule(
            metric="row_loss_share",
            direction="max",
            min_effect=max(self._cfg.features.max_drop_fraction, 1e-9),
            min_sample_size=1,
            action="fix the feature transform before training",
            owner=self._cfg.dataset.name,
        )

    def decision(self) -> Decision:
        assert self._cfg.features is not None and self._cfg.dataset is not None
        audit = self._audit
        rows_in = max(audit.rows_in, 1)
        value = audit.rows_dropped_total / rows_in
        val_loss = (
            0.0
            if self._val_rows_out is None or self._val_rows_in == 0
            else (self._val_rows_in - self._val_rows_out) / self._val_rows_in
        )
        evidence = Evidence(
            baseline="row-preserving transform (every input row survives)",
            segments=[
                SegmentEvidence(
                    segment="train_split", value=value, sample_size=rows_in
                ),
                SegmentEvidence(
                    segment="val_split",
                    value=val_loss,
                    sample_size=max(self._val_rows_in, 1),
                ),
            ],
            sensitivity={
                f"features_added_n{rows_in}": float(len(audit.columns_added)),
                f"features_removed_n{rows_in}": float(len(audit.columns_removed)),
                f"plan_columns_n{rows_in}": float(len(self._plan_names)),
                f"first_half_row_loss_n{self._half_sample_rows}": self._half_sample_loss,
            },
            falsifier="the row-loss interval clears the configured drop ceiling",
            provenance={
                "artifact": self._cfg.features.pipeline_file,
                "dataset": self._cfg.dataset.name,
                "plan_columns": ", ".join(self._plan_names),
                "columns_added": ", ".join(audit.columns_added) or "none",
                "columns_removed": ", ".join(audit.columns_removed) or "none",
            },
        )
        return Decision.evaluate(
            rule=self.rule(),
            value=value,
            interval=(value, value),
            sample_size=rows_in,
            question="Did the feature step keep every row and the declared feature surface?",
            evidence=evidence,
            unit="share",
        )

    def publish(self) -> Decision:
        """Write the record plus the keep/drop/transform figure."""
        assert self._cfg.features is not None
        decision = self.decision()
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        audit = self._audit
        figure, axis = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
        labels = ["kept", "added", "removed"]
        values = [
            len(self._plan_names),
            len(audit.columns_added),
            len(audit.columns_removed),
        ]
        bars = axis.bar(labels, values, color=["#08519c", "#009E73", "#D55E00"])
        for bar in bars:
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{int(bar.get_height())}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        axis.set_ylabel("columns")
        axis.set_title("feature surface: keep / drop / transform")
        output = StepOutput(
            results_dir=Path(self._cfg.features.output_dir), stem="features", decision=decision
        )
        output.save_figure(figure)
        plt.close("all")
        output.write()
        return decision


def run(cfg: PipelineConfig) -> None:
    if not cfg.dataset or not cfg.experiment or not cfg.features or not cfg.etl:
        raise ValueError("features step requires dataset, experiment, features, and etl config")
    train, val = _load_split(cfg)
    plan = ColumnPlan.from_config(cfg.dataset, cfg.experiment)
    pipeline = FeaturePipeline()
    pipeline.fit(train, plan, cfg.dataset.target)
    rows_in = len(train)
    columns_before = list(train.columns)
    train_out = _plan_ordered(
        pipeline.transform(train, plan, cfg.dataset.target, cfg.features.frequency_fill), plan
    )
    specs = build_generic_feature_specs(cfg.dataset, plan)
    build_engineered_schema(specs).validate(train_out)
    rows_out = len(train_out)
    columns_after = list(train_out.columns)
    out_dir = ProcessedStore.from_config(cfg).directory
    train_out.to_parquet(out_dir / cfg.etl.train_features_file, index=False)
    logger.info(f"train features written ({len(train_out)} rows)")
    if val is not None:
        val_out = _plan_ordered(
            pipeline.transform(val, plan, cfg.dataset.target, cfg.features.frequency_fill), plan
        )
        build_engineered_schema(specs).validate(val_out)
        val_out.to_parquet(out_dir / cfg.etl.val_features_file, index=False)
        logger.info(f"val features written ({len(val_out)} rows)")
    pipeline_path = out_dir / cfg.features.pipeline_file
    with open(pipeline_path, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info(f"pipeline saved to {pipeline_path}")
    dropped_total = rows_in - rows_out
    audit = TransformAudit(
        rows_in=rows_in,
        rows_out=rows_out,
        rows_dropped_total=dropped_total,
        rows_dropped_unexplained=max(0, dropped_total),
        reasons=[] if dropped_total == 0 else [f"unexpected row loss: {dropped_total} rows"],
        columns_before=columns_before,
        columns_after=columns_after,
        columns_added=sorted(set(columns_after) - set(columns_before)),
        columns_removed=sorted(set(columns_before) - set(columns_after)),
    )
    enforce_drop_fraction(audit, cfg.features.max_drop_fraction)
    half_rows = len(train) // 2
    half_out = pipeline.transform(
        train.iloc[:half_rows], plan, cfg.dataset.target, cfg.features.frequency_fill
    )
    decision = FeaturesDecision(
        cfg,
        audit,
        plan_names=list(plan.names),
        val_rows_in=0 if val is None else len(val),
        val_rows_out=None if val is None else len(val_out),
        half_sample_loss=(half_rows - len(half_out)) / max(half_rows, 1),
        half_sample_rows=max(half_rows, 1),
    ).publish()
    logger.info(
        "features decision: %s — row loss %.4f (ceiling %.4f), +%d/-%d columns",
        decision.verdict,
        decision.value,
        decision.threshold,
        len(audit.columns_added),
        len(audit.columns_removed),
    )
    write_record(node_id("features", cfg.dataset.name), "features", str(pipeline_path), [node_id("etl", cfg.dataset.name)], audit=audit)
