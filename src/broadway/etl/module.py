"""Orchestrates data layer — load → sample → canonicalize → validate → save → split."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

from broadway.cleaning.models import StructuralCleanResult
from broadway.cleaning.structural import CoercionAuditReport, CoercionRecord
from broadway.config.schema import PipelineConfig, TaskType
from broadway.contracts.pandera import build_raw_schema
from broadway.contracts.selectors import datetime_columns, numeric_columns
from broadway.data.cleaner import canonicalize
from broadway.data.join_audit import JoinAuditReport
from broadway.data.loader import ArtifactStore, ProcessedStore, canonical_path, load_with_audit
from broadway.data.lookup_value_audit import LookupValueAuditReport
from broadway.data.splitter import split
from broadway.lineage.ids import node_id
from broadway.lineage.models import TransformAudit
from broadway.lineage.records import enforce_drop_fraction, records_dir, write_record
from broadway.reports.decision import Decision, DecisionRule, Evidence, SegmentEvidence
from broadway.reports.decision_log import DecisionLog
from broadway.reports.step_output import StepOutput
from broadway.tracking.manifest import tracking_dir, write_etl_manifest

logger = logging.getLogger(__name__)

# The ETL's class-balance evidence file (named in the decision's provenance).
CLASS_BALANCE_FILE = "class_balance.csv"

# Loaders whose data etl can ingest from the raw source. Experiments bound to
# pre-built loaders (named_sample, pinned) consume immutable artifacts and
# must never re-run etl — the declared loader drives that selection.
_RAW_INGEST_LOADERS = frozenset({"canonical", "joined"})


def _explained_rows(reasons: list[str]) -> int:
    total = 0
    for reason in reasons:
        match = re.search(r"-(\d+) rows$", reason)
        if match:
            total += int(match.group(1))
    return total


def _assert_data_source_supported(cfg: PipelineConfig) -> None:
    """The experiment's declared data source selects whether etl may run.

    ``canonical`` and ``joined`` resolve through the raw ingest; any other
    loader means the experiment consumes a pre-built artifact, so running
    etl for it fails loud instead of silently re-ingesting.
    """
    if cfg.experiment is None:
        raise ValueError(
            "etl requires an experiment binding: without one its output "
            "would overwrite the raw input path declared by dataset.path"
        )
    loader = cfg.experiment.data_source.loader
    if loader not in _RAW_INGEST_LOADERS:
        raise ValueError(
            f"etl cannot run for data_source.loader '{loader}' "
            f"(supported: {', '.join(sorted(_RAW_INGEST_LOADERS))})"
        )


class ClassBalance(BaseModel):
    """The binary target's class balance at the ETL boundary.

    ``bad``/``good`` are measured on the PRODUCED canonical frame (positive is
    target == 1, negative target == 0); ``missing`` is the target-null count in
    the input frame that the ETL's accounted drop removed. ``bad_share`` is
    bad / kept, so the 768 pre-ETL nulls cannot deflate the kept-frame rate.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    bad: int
    good: int
    missing: int
    total: int
    kept: int
    bad_share: float

    @classmethod
    def binary(
        cls, raw: pd.DataFrame, produced: pd.DataFrame, target: str
    ) -> ClassBalance | None:
        """Build the balance for a 0/1 target; ``None`` for any other shape."""
        observed = set(raw[target].dropna().unique())
        if not observed <= {0, 1}:
            logger.info(
                "etl: target %r is not binary (values=%s); no class balance table",
                target,
                sorted(str(value) for value in observed),
            )
            return None
        bad = int((produced[target] == 1).sum())
        good = int((produced[target] == 0).sum())
        kept = len(produced)
        return cls(
            bad=bad,
            good=good,
            missing=int(raw[target].isna().sum()),
            total=len(raw),
            kept=kept,
            bad_share=bad / kept if kept else 0.0,
        )

    def frame(self) -> pd.DataFrame:
        """The tall evidence table the story reads (``class`` + ``count``).

        ``share_of_total`` sums to 1 over all input rows; ``share_of_kept`` is
        defined for the modelled bad/good classes only (missing is a pre-ETL
        bucket, so its kept share is null).
        """
        total = self.total or 1
        kept = self.kept or 1
        return pd.DataFrame(
            {
                "class": ["bad", "good", "missing"],
                "count": [self.bad, self.good, self.missing],
                "share_of_total": [
                    self.bad / total,
                    self.good / total,
                    self.missing / total,
                ],
                "share_of_kept": [self.bad / kept, self.good / kept, None],
                "total": [self.total] * 3,
                "kept": [self.kept] * 3,
            }
        )


class EtlDecision:
    """Publish the row-drop decision: did the ingest lose more than the contract allows?

    The metric is the UNEXPLAINED drop share (a drop no recorded reason
    accounts for) — the signal that the source changed under the contract.
    The explained drops are the evidence slices, not the headline.
    """

    def __init__(
        self,
        cfg: PipelineConfig,
        audit: TransformAudit,
        balance: ClassBalance | None = None,
        balance_file: str | None = None,
    ) -> None:
        assert cfg.dataset is not None and cfg.etl is not None
        self._cfg = cfg
        self._audit = audit
        self._balance = balance
        self._balance_file = balance_file

    def rule(self) -> DecisionRule:
        assert self._cfg.dataset is not None and self._cfg.etl is not None
        # A configured zero drop tolerance means "no loss": the floor is a
        # positive epsilon only because the rule model forbids zero.
        return DecisionRule(
            metric="rows_dropped_unexplained_share",
            direction="max",
            min_effect=max(self._cfg.etl.max_drop_fraction, 1e-9),
            min_sample_size=1,
            action="fix the source before training",
            owner=self._cfg.dataset.name,
        )

    def decision(self) -> Decision:
        audit = self._audit
        rows_in = max(audit.rows_in, 1)
        unexplained = audit.rows_dropped_unexplained / rows_in
        explained = max(audit.rows_dropped_total - audit.rows_dropped_unexplained, 0) / rows_in
        provenance = {
            "artifact": "canonical.parquet",
            "dataset": self._cfg.dataset.name if self._cfg.dataset else "",
            "rows_in": str(audit.rows_in),
            "rows_out": str(audit.rows_out),
            "reasons": "; ".join(audit.reasons) or "none",
        }
        if self._balance is not None and self._balance_file is not None:
            provenance["class_balance"] = self._balance_file
        evidence = Evidence(
            baseline="contract-clean ingest: every dropped row carries a recorded reason",
            segments=[
                SegmentEvidence(
                    segment="explained_drops", value=explained, sample_size=rows_in
                ),
                SegmentEvidence(
                    segment="unexplained_drops", value=unexplained, sample_size=rows_in
                ),
            ],
            sensitivity={
                f"total_drop_share_n{rows_in}": audit.rows_dropped_total / rows_in,
                f"rows_out_n{audit.rows_out}": float(audit.rows_out),
            },
            falsifier="the unexplained-share interval clears the configured drop ceiling",
            provenance=provenance,
        )
        return Decision.evaluate(
            rule=self.rule(),
            value=unexplained,
            interval=(unexplained, unexplained),
            sample_size=rows_in,
            question="Did the ingest lose more rows than the contract explains?",
            evidence=evidence,
            unit="share",
        )

    def publish(self, out_dir: Path) -> Decision:
        """Write the decision record + row-funnel figure next to the processed data."""
        target = Path(self._cfg.etl.output_dir) if self._cfg.etl and self._cfg.etl.output_dir else out_dir
        if self._balance is not None and self._balance_file is not None:
            target.mkdir(parents=True, exist_ok=True)
            balance_path = target / self._balance_file
            self._balance.frame().to_csv(balance_path, index=False)
            logger.info(
                "etl: class balance %d bad / %d good / %d missing (kept %d) -> %s",
                self._balance.bad,
                self._balance.good,
                self._balance.missing,
                self._balance.kept,
                balance_path,
            )
        decision = self.decision()
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        audit = self._audit
        figure, axis = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
        stages = ["rows in", "rows out", "unexplained"]
        values = [audit.rows_in, audit.rows_out, audit.rows_dropped_unexplained]
        bars = axis.bar(stages, values, color=["#9ecae1", "#08519c", "#d62728"])
        for bar in bars:
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{int(bar.get_height()):,}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        axis.set_ylabel("rows")
        axis.set_title("ETL row funnel")
        output = StepOutput(
            results_dir=target,
            stem="etl",
            decision=decision,
            log=DecisionLog(tracking_uri=self._cfg.environment.mlflow_tracking_uri),
        )
        output.save_figure(figure)
        plt.close("all")
        output.write()
        return decision


def run(cfg: PipelineConfig) -> None:
    if not cfg.dataset:
        raise ValueError("etl step requires a dataset config")
    if not cfg.etl:
        raise ValueError("etl step requires an etl config")
    dataset = cfg.dataset
    _assert_data_source_supported(cfg)
    df, join_audits, value_audits = load_with_audit(dataset)
    rows_in = len(df)
    columns_before = list(df.columns)

    reasons: list[str] = []
    rs = cfg.experiment.random_state if cfg.experiment else cfg.etl.random_state
    if cfg.etl.ci_sample_size > 0 and os.getenv("CI") == "true":
        n_before = len(df)
        df = df.sample(n=min(cfg.etl.ci_sample_size, len(df)), random_state=rs)
        if len(df) < n_before:
            reasons.append(f"CI sampling: -{n_before - len(df)} rows")

    numeric_map = {col: dataset.columns[col].dtype for col in numeric_columns(dataset)}
    coercions: list[CoercionRecord] = []
    raw = df
    df, clean_reasons, parse_failures, observed_missing = canonicalize(
        df,
        target=dataset.target,
        datetime_columns=datetime_columns(dataset),
        numeric_columns=numeric_map,
        missing_encodings=cfg.etl.missing_encodings,
        coercions=coercions,
    )
    reasons.extend(clean_reasons)
    balance = (
        ClassBalance.binary(raw, df, dataset.target)
        if dataset.task is TaskType.CLASSIFICATION
        else None
    )

    out_dir = ProcessedStore.from_config(cfg).directory
    out_dir.mkdir(parents=True, exist_ok=True)

    build_raw_schema(dataset).validate(df)

    canonical_path_ = canonical_path(dataset, cfg.environment)
    df.to_parquet(canonical_path_, index=False)
    logger.info(f"saved canonical ({len(df)} rows) to {canonical_path_}")

    split_cfg = cfg.experiment.split if cfg.experiment else None
    if split_cfg:
        train, val = split(df, dataset, split_cfg, random_state=rs)
        train.to_parquet(out_dir / cfg.etl.train_file, index=False)
        val.to_parquet(out_dir / cfg.etl.val_file, index=False)
        logger.info(f"saved train ({len(train)} rows) and val ({len(val)} rows)")
        rows_out = len(train) + len(val)
    else:
        df.to_parquet(out_dir / cfg.etl.training_data_file, index=False)
        logger.info(f"saved training_data ({len(df)} rows)")
        rows_out = len(df)

    columns_after = list(df.columns)
    dropped_total = rows_in - rows_out
    unexplained = max(0, dropped_total - _explained_rows(reasons))
    audit = TransformAudit(
        rows_in=rows_in,
        rows_out=rows_out,
        rows_dropped_total=dropped_total,
        rows_dropped_unexplained=unexplained,
        reasons=reasons,
        columns_before=columns_before,
        columns_after=columns_after,
        columns_added=sorted(set(columns_after) - set(columns_before)),
        columns_removed=sorted(set(columns_before) - set(columns_after)),
    )
    enforce_drop_fraction(audit, cfg.etl.max_drop_fraction)
    decision = EtlDecision(
        cfg,
        audit,
        balance=balance,
        balance_file=CLASS_BALANCE_FILE if balance is not None else None,
    ).publish(out_dir)
    logger.info(
        "etl decision: %s — unexplained drop share %.4f (ceiling %.4f)",
        decision.verdict,
        decision.value,
        decision.threshold,
    )

    result = StructuralCleanResult(
        audit=audit,
        parse_failures=parse_failures,
        missing_encodings=observed_missing,
        canonical_path=str(canonical_path_),
    )
    result_path = out_dir / f"{dataset.name}_clean.json"
    result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    logger.info(f"saved structural clean result to {result_path}")

    experiment = cfg.experiment
    if experiment is None:
        raise ValueError("etl manifest requires an experiment binding")
    write_etl_manifest(
        dataset=dataset,
        audit=audit,
        canonical_path=canonical_path_,
        schema_contract=experiment.data_source.schema_contract,
        df=df,
        tracking_root=ArtifactStore.from_contract(dataset, tracking_dir()).directory,
    )

    ingest_id = node_id("ingest", dataset.name)
    upstream = (
        [ingest_id]
        if (records_dir() / f"{ingest_id.replace(':', '_')}.json").exists()
        else [node_id("dataset", dataset.name)]
    )
    if coercions:
        coercion_audit_path = out_dir / f"{dataset.name}_coercion_audit.json"
        coercion_audit_path.write_text(
            CoercionAuditReport(coercions=coercions).model_dump_json(indent=2),
            encoding="utf-8",
        )
        write_record(
            node_id("coercion", dataset.name),
            "coercion",
            str(coercion_audit_path),
            upstream,
        )
    if join_audits:
        join_audit_path = out_dir / f"{dataset.name}_join_audit.json"
        join_audit_path.write_text(
            JoinAuditReport(joins=join_audits).model_dump_json(indent=2), encoding="utf-8"
        )
        write_record(node_id("join", dataset.name), "join", str(join_audit_path), upstream)
        value_audit_path = out_dir / f"{dataset.name}_lookup_value_audit.json"
        value_audit_path.write_text(
            LookupValueAuditReport(lookups=value_audits).model_dump_json(indent=2), encoding="utf-8"
        )
        write_record(node_id("lookup_value", dataset.name), "lookup_value", str(value_audit_path), [node_id("join", dataset.name)])
        etl_parent = [node_id("join", dataset.name)]
    else:
        etl_parent = upstream
    write_record(
        node_id("etl", dataset.name),
        "etl",
        str(canonical_path_),
        etl_parent,
        audit=audit,
    )
