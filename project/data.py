"""Olist canonical order-grain builder — the single owner of the raw pre-join.

The nine downloaded CSVs are joined and aggregated into ONE row per ``order_id``
(the parquet ``DatasetContract.path`` names). Every source label, join key,
aggregate and output column is declared in ``project/config/project/olist.yaml``;
this module contains no column literal. The build is deterministic: a stable
sort on two review timestamps resolves the 551 duplicate review rows, every
groupby is sorted, and the output frame is ordered by ``order_id``.

Row policy (reported in ``<name>_build_report.json`` next to the parquet):

* no row is dropped here — orders without a review keep a null ``bad_review``
  and the ETL's accounted "null target" rule removes them (768);
* orders without item rows are KEPT with null item aggregates (775 overall,
  756 with a review); their count columns are 0 because the order has zero
  item lines, which is a fact, not an imputation.

Target: ``bad_review = review_score <= 2`` (config threshold), a binary
classification target whose decision moment is at purchase. Delivery, review
and post-purchase timestamp columns are carried as EDA/context only; the
feature plan (``project/features/olist.py``) never selects them.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from broadway.paths import repo_root
from project.paths import load_project_paths

logger = logging.getLogger(__name__)

_SECONDS_PER_DAY = 86_400.0


class _Spec(BaseModel):
    """Base for the frozen, exhaustively validated join spec."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class FileSet(_Spec):
    orders: str
    customers: str
    items: str
    payments: str
    reviews: str
    products: str
    sellers: str
    category_translation: str


class OrdersSpec(_Spec):
    key: str
    customer_key: str
    status: str
    purchase: str
    approved: str
    carrier: str
    delivered: str
    estimated: str


class CustomersSpec(_Spec):
    key: str
    columns: dict[str, str] = Field(min_length=1)
    zero_pad: dict[str, int] = Field(default_factory=dict)


class EnrichSpec(_Spec):
    file: str
    key: str
    columns: dict[str, str] = Field(min_length=1)


class ItemsSpec(_Spec):
    key: str
    count: str
    distinct: dict[str, str] = Field(default_factory=dict)
    sum: dict[str, str] = Field(default_factory=dict)
    mean: dict[str, str] = Field(default_factory=dict)
    mode: dict[str, str] = Field(default_factory=dict)
    enrich: list[EnrichSpec] = Field(default_factory=list)


class PaymentsSpec(_Spec):
    key: str
    sum: dict[str, str] = Field(default_factory=dict)
    max: dict[str, str] = Field(default_factory=dict)
    distinct: dict[str, str] = Field(default_factory=dict)
    mode: dict[str, str] = Field(default_factory=dict)


class ReviewsSpec(_Spec):
    key: str
    score: str
    target: str
    bad_threshold: int = Field(ge=1)
    dedupe_sort: list[str] = Field(min_length=1)


class DeliveriesSpec(_Spec):
    delivery_days: str
    estimated_days: str
    late: str


class CategoryTranslationSpec(_Spec):
    file: str
    key: str
    value: str
    source: str
    output: str


class OlistBuildSpec(_Spec):
    """The declarative join/aggregation spec read from the project config."""

    name: str
    source_dir: str
    output: str
    grain: str
    files: FileSet
    orders: OrdersSpec
    customers: CustomersSpec
    items: ItemsSpec
    payments: PaymentsSpec
    reviews: ReviewsSpec
    deliveries: DeliveriesSpec
    category_translation: CategoryTranslationSpec

    @model_validator(mode="after")
    def _grain_matches_order_key(self) -> OlistBuildSpec:
        if self.grain != self.orders.key:
            raise ValueError(
                f"grain '{self.grain}' must equal orders.key '{self.orders.key}'"
            )
        return self


class ColumnStats(BaseModel):
    """One produced column's observed dtype and null count."""

    model_config = ConfigDict(extra="forbid")

    dtype: str
    null_count: int


class BuildReport(BaseModel):
    """The pre-join funnel a reader needs to audit the grain and the drops."""

    model_config = ConfigDict(extra="forbid")

    name: str
    grain: str
    output: str
    source_rows: dict[str, int]
    review_rows: int
    review_orders: int
    duplicate_review_rows: int
    orders_without_items: int
    orders_without_reviews: int
    rows: int
    class_balance: dict[str, int]
    columns: dict[str, ColumnStats]


def _dominant(series: pd.Series) -> object:
    """The most frequent non-null value, ties broken by the smallest label."""
    counts = series.dropna().value_counts()
    if counts.empty:
        return None
    tied = counts[counts == counts.max()].index
    return min(tied)


class OlistCanonicalBuilder:
    """Factory + executor for the Olist order-grain canonical table."""

    def __init__(self, spec: OlistBuildSpec, root: Path) -> None:
        self._spec = spec
        self._source_dir = root / spec.source_dir
        self._output = root / spec.output

    @classmethod
    def from_config(cls, path: Path | None = None) -> OlistCanonicalBuilder:
        """Load the join spec; a missing file fails loud at the boundary."""
        config_path = path or load_project_paths().config / "project" / "olist.yaml"
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{config_path}: join spec must be a mapping")
        return cls(OlistBuildSpec(**raw), repo_root())

    @property
    def output(self) -> Path:
        return self._output

    def build(self) -> Path:
        """Read, join, aggregate and write the parquet + its funnel report."""
        started = time.perf_counter()
        frames = self._read_sources()
        source_rows = {name: len(source) for name, source in frames.items()}
        frame, counts = self._assemble(frames)
        frame = frame.sort_values(self._spec.grain, kind="mergesort").reset_index(drop=True)
        self._output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(self._output, index=False)
        self._write_report(frame, counts, source_rows)
        logger.info(
            "olist: wrote %s rows to %s in %.2fs",
            len(frame),
            self._output,
            time.perf_counter() - started,
        )
        return self._output

    def _read_sources(self) -> dict[str, pd.DataFrame]:
        missing = [
            name
            for name in self._spec.files.model_dump()
            if not (self._source_dir / getattr(self._spec.files, name)).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"olist source_dir {self._source_dir} is missing: {missing}"
            )
        return {
            name: pd.read_csv(self._source_dir / getattr(self._spec.files, name))
            for name in self._spec.files.model_dump()
        }

    def _assemble(
        self, frames: dict[str, pd.DataFrame]
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        orders = frames["orders"]
        reviews = self._dedupe_reviews(frames["reviews"])
        counts = {
            "review_rows": len(frames["reviews"]),
            "review_orders": int(reviews[self._spec.reviews.key].nunique()),
        }
        counts["duplicate_review_rows"] = counts["review_rows"] - counts["review_orders"]
        counts["orders_without_items"] = int(
            (~orders[self._spec.orders.key].isin(frames["items"][self._spec.items.key])).sum()
        )
        counts["orders_without_reviews"] = int(
            (~orders[self._spec.orders.key].isin(reviews[self._spec.reviews.key])).sum()
        )

        frame = self._base_orders(orders)
        frame = frame.merge(
            self._customers(frames["customers"]),
            left_on=self._spec.orders.customer_key,
            right_on=self._spec.customers.key,
            how="left",
            validate="many_to_one",
        )
        frame = frame.merge(
            self._items(frames), on=self._spec.items.key, how="left", validate="one_to_one"
        )
        for name in (self._spec.items.count, *self._spec.items.distinct.values()):
            frame[name] = frame[name].fillna(0.0)
        frame = frame.merge(
            self._payments(frames["payments"]),
            on=self._spec.payments.key,
            how="left",
            validate="one_to_one",
        )
        frame = self._with_reviews(frame, reviews)
        frame = self._with_deliveries(frame)
        frame = self._with_category(frame, frames[self._spec.category_translation.file])
        frame = frame[self._output_columns()]
        return frame, counts

    def _base_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        o = self._spec.orders
        timestamps = [o.purchase, o.approved, o.carrier, o.delivered, o.estimated]
        frame = orders[
            [o.key, o.customer_key, o.status, *timestamps]
        ].copy()
        for name in timestamps:
            frame[name] = pd.to_datetime(frame[name], errors="coerce")
        return frame

    def _customers(self, customers: pd.DataFrame) -> pd.DataFrame:
        c = self._spec.customers
        frame = customers[[c.key, *c.columns]].drop_duplicates(subset=[c.key]).copy()
        for column, width in c.zero_pad.items():
            if column not in frame.columns:
                raise ValueError(f"customers.zero_pad names absent column '{column}'")
            # CEP prefixes can start with 0; an int read silently loses that digit.
            frame[column] = frame[column].astype("int64").astype(str).str.zfill(width)
        return frame.rename(columns=c.columns)

    def _items(self, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        i = self._spec.items
        enriched = frames["items"].copy()
        for enrich in i.enrich:
            right = frames[enrich.file][[enrich.key, *enrich.columns]]
            right = right.drop_duplicates(subset=[enrich.key])
            enriched = enriched.merge(
                right.rename(columns=enrich.columns),
                on=enrich.key,
                how="left",
                validate="many_to_one",
            )
        grouped = enriched.groupby(i.key, sort=True)
        aggregate = grouped.size().rename(i.count).to_frame()
        for column, name in i.distinct.items():
            aggregate[name] = grouped[column].nunique()
        for column, name in i.sum.items():
            aggregate[name] = grouped[column].sum(min_count=0)
        for column, name in i.mean.items():
            aggregate[name] = grouped[column].mean()
        for column, name in i.mode.items():
            aggregate[name] = grouped[column].agg(_dominant)
        result = aggregate.reset_index()
        for name in (i.count, *i.distinct.values()):
            result[name] = result[name].astype("float64")
        return result

    def _payments(self, payments: pd.DataFrame) -> pd.DataFrame:
        p = self._spec.payments
        grouped = payments.groupby(p.key, sort=True)
        aggregate = pd.DataFrame(index=grouped.size().index)
        for column, name in p.sum.items():
            aggregate[name] = grouped[column].sum(min_count=0)
        for column, name in p.max.items():
            aggregate[name] = grouped[column].max()
        for column, name in p.distinct.items():
            aggregate[name] = grouped[column].nunique()
        for column, name in p.mode.items():
            aggregate[name] = grouped[column].agg(_dominant)
        return aggregate.reset_index()

    def _dedupe_reviews(self, reviews: pd.DataFrame) -> pd.DataFrame:
        r = self._spec.reviews
        ordered = reviews.sort_values([r.key, *r.dedupe_sort], kind="mergesort")
        return ordered.drop_duplicates(subset=[r.key], keep="last")

    def _with_reviews(self, frame: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
        r = self._spec.reviews
        scores = reviews.set_index(r.key)[r.score]
        frame[r.score] = frame[self._spec.orders.key].map(scores)
        frame[r.target] = np.where(
            frame[r.score].notna(),
            (frame[r.score] <= r.bad_threshold).astype("float64"),
            np.nan,
        )
        return frame

    def _with_deliveries(self, frame: pd.DataFrame) -> pd.DataFrame:
        d = self._spec.deliveries
        o = self._spec.orders
        frame[d.delivery_days] = (
            frame[o.delivered] - frame[o.purchase]
        ).dt.total_seconds() / _SECONDS_PER_DAY
        frame[d.estimated_days] = (
            frame[o.estimated] - frame[o.purchase]
        ).dt.total_seconds() / _SECONDS_PER_DAY
        late = pd.Series(np.nan, index=frame.index, dtype="float64")
        known = frame[o.delivered].notna() & frame[o.estimated].notna()
        late.loc[known] = (
            frame.loc[known, o.delivered] > frame.loc[known, o.estimated]
        ).astype("float64")
        frame[d.late] = late
        return frame

    def _with_category(self, frame: pd.DataFrame, translation: pd.DataFrame) -> pd.DataFrame:
        t = self._spec.category_translation
        lookup = (
            translation[[t.key, t.value]]
            .dropna()
            .drop_duplicates(subset=[t.key])
            .set_index(t.key)[t.value]
        )
        frame[t.output] = frame[t.source].map(lookup)
        return frame.drop(columns=[t.source])

    def _output_columns(self) -> list[str]:
        """The deterministic output order, generated from the spec (no literals)."""
        o = self._spec.orders
        i = self._spec.items
        p = self._spec.payments
        r = self._spec.reviews
        d = self._spec.deliveries
        t = self._spec.category_translation
        names = [
            o.key,
            o.customer_key,
            *self._spec.customers.columns,
            o.status,
            o.purchase,
            o.approved,
            o.carrier,
            o.delivered,
            o.estimated,
            d.delivery_days,
            d.estimated_days,
            d.late,
            i.count,
            *i.distinct.values(),
            *i.sum.values(),
            *i.mean.values(),
            *p.sum.values(),
            *p.max.values(),
            *p.distinct.values(),
            *p.mode.values(),
            t.output,
            *(name for name in i.mode.values() if name != t.source),
            r.score,
            r.target,
        ]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"olist join spec produces duplicate output column(s): {duplicates}")
        return names

    def _write_report(
        self, frame: pd.DataFrame, counts: dict[str, Any], source_rows: dict[str, int]
    ) -> None:
        r = self._spec.reviews
        report = BuildReport(
            name=self._spec.name,
            grain=self._spec.grain,
            output=str(self._output),
            source_rows=source_rows,
            review_rows=counts["review_rows"],
            review_orders=counts["review_orders"],
            duplicate_review_rows=counts["duplicate_review_rows"],
            orders_without_items=counts["orders_without_items"],
            orders_without_reviews=counts["orders_without_reviews"],
            rows=len(frame),
            class_balance={
                "bad_review": int((frame[r.target] == 1.0).sum()),
                "good_review": int((frame[r.target] == 0.0).sum()),
                "missing_target": int(frame[r.target].isna().sum()),
            },
            columns={
                name: ColumnStats(
                    dtype=str(frame[name].dtype),
                    null_count=int(frame[name].isna().sum()),
                )
                for name in frame.columns
            },
        )
        path = self._output.parent / f"{self._spec.name}_build_report.json"
        path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        logger.info(
            "olist: %d rows, bad=%d good=%d missing=%d; report %s",
            report.rows,
            report.class_balance["bad_review"],
            report.class_balance["good_review"],
            report.class_balance["missing_target"],
            path,
        )


def main() -> None:
    """Build the canonical table; the printed path is the dataset contract's path."""
    print(OlistCanonicalBuilder.from_config().build())


if __name__ == "__main__":
    main()
