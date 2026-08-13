Broadway — LLM Handoff

Purpose of this handoff

You are assisting on Broadway, an opinionated framework for reproducible, traceable tabular data science.

The user wants help in the same style as the prior assistant:

reason about architecture before adding code,

keep scope small and phased,

avoid speculative abstractions,

protect single sources of truth,

prioritize traceability and loud failure,

distinguish authored intent from observed evidence and runtime results,

prefer mature external tools over reimplementing them,

and use real dataset work to drive the next abstractions.

Do not treat Broadway as a generic AutoML system or model-serving framework.

The current product thesis is:

Broadway is a traceable workflow framework for tabular statistical analysis and predictive modeling that preserves the path from raw data and analytical intent through evidence, decisions, features, models, and results.

Operational principle:

Automate the mechanics; facilitate the judgment; record the decision.

Maintainability litmus test:

If the dataset YAML were swapped tomorrow, files under src/broadway/** should generally not need editing.

That invariant is now demonstrated by a non-taxi end-to-end onboarding test.

Repository / branch state

main

main is now the proven generic platform.

Current state:

pushed to origin/main

194 tests green

contains generic src/broadway/**

contains generic configs, tests, packaging, and public-facing platform docs

contains no taxi residue in src/, configs/, or tests/

the only remaining taxi string on main is descriptive text in a non-taxi onboarding test docstring

Important split commits:

49046f9 — removed taxi-owned files, genericized platform test fixtures, removed dead taxi feature code

7231a6e — removed dead ProjectConfig

b4758df — rewrote README / dataflow / HANDOFF around the platform boundary

a618695 — fixed one stale project/data.py reference in stats docs

main should remain the clean abstraction boundary.

taxi

taxi is the demo + active development branch.

It remains the real NYC taxi application and the place where new abstractions are dogfooded before promotion to main.

The branch was intentionally left untouched by the ownership split at:

7be0dd3

The stale broadway branch still exists and is intentionally untouched for now.

Uncommitted/untracked user notes such as DATA_VALIDATION.md, BROADWAY.md, and EXTERNAL_HELP.md are outside branch history unless explicitly committed later.

Core architectural principles

1. One authoritative owner per concept

Concept

Owner

Accepted raw/canonical dataset schema

DatasetContract

Observed dataset facts

DatasetProfile / ColumnProfile

Analytical intent

AnalysisContract

Naive reference result

BaselineResult

Engineered feature definition

FeatureSpec / experiment feature config

Runtime provenance

ArtifactTrace / lineage sidecars

Dataset slices

authored config

Actual analysis decisions

runtime DecisionRecord artifacts

Deterministic structural-cleaning policy

step config + typed evidence + TransformAudit

Never create parallel hand-maintained definitions if something can be derived from an authoritative contract.

2. Distinguish authored intent, observed evidence, and execution outputs

Authored / committed

Examples:

configs/dataset/*.yaml

configs/analysis/*.yaml

configs/experiment/*.yaml

configs/slice/*.yaml

step configs / policies

optional project feature-builder declarations

Observed / regenerated

Examples:

artifacts/discover/profile.json

diagnostics

statistics

model evaluation

lineage graph

structural-cleaning evidence

Runtime decisions

Actual analytical decisions are events, not config:

actor / source

evidence

reason

outcome

timestamp

parents

Store them under:

artifacts/lineage/decisions/

Do not use DecisionRecord for deterministic mechanics such as exact duplicate removal or configured missing-value normalization. Those belong to policy + typed evidence + audit.

3. No silent branches

Broadway distinguishes:

Fact — e.g. variance ratio = 2.71

Policy — e.g. configured missing encodings or a recommendation threshold

Decision — e.g. analyst chooses Welch's t-test

No automatic branch should exist without a clear reason/evidence trail.

When automation is ambiguous, Broadway should recommend rather than pretend there is one universally correct answer.

4. Loud failure

Current policy includes:

missing artifact → None only when absence is valid

malformed artifact → raise

missing required feature/config → raise

non-finite metrics/objectives → raise

unexplained row loss above configured threshold → fail

missing expected output after a step ran → integrity error

important warnings should be typed results, not log-only

custom builder import / registry errors → fail where the builder is actually needed

custom feature builders extend the generic registry; they do not silently override generic builders

General step invariant:

input validated → operation → output validated → artifact persisted → lineage sidecar persisted → invariants checked

High-level lifecycle

Broadway has three analysis modes:

prediction

hypothesis

causal

AnalysisContract.mode selects the workflow.

Conceptually:

AnalysisContract
    ↓
full dispatcher
    ↓
mode-specific flow

Mode flows:

prediction:
discover → etl → contracts → eda → baseline → features → train → evaluate

hypothesis:
discover → etl → contracts → eda → baseline → stats

causal:
discover → etl → contracts → eda → baseline → causal

Per-step require_mode(...) remains the final guardrail.

Major shipped abstractions

AnalysisContract

Represents authored analytical intent.

Key fields include:

name

mode

goal

row_definition

decision_moment

available_info

leakage_notes

success_criterion

These remain author-owned. Discovery may generate hints, but must not mutate analytical intent.

DatasetProfile / ColumnProfile

Observed facts live outside DatasetContract.

Includes things such as:

dtype

null count

cardinality

min/max

datetime min/max

identifier score

Observed facts are evidence, not accepted truth.

BaselineResult

Mode-specific baselines:

prediction → mean/median/majority-class

hypothesis → descriptive stats + naive effect

causal → estimand + power/MDE

Persisted under artifacts/baseline/.

TransformAudit

Tracks:

rows in/out

total rows dropped

unexplained rows dropped

reasons

columns before/after

columns added/removed

ETL/features emit audits.

max_drop_fraction guards unexplained row loss.

Lineage / decision system

Core lineage concepts:

DatasetRef

DatasetSlice

DecisionRecord

LineageRecord

LineageNode

LineageEdge

LineageGraph

RunState

Sidecars:

artifacts/lineage/records/

Decisions:

artifacts/lineage/decisions/

CLI:

ds-pipeline lineage --analysis <name> --dataset <name>

Graph generation uses only real persisted nodes and must never fabricate missing execution outputs.

Onboarding abstraction — shipped P0 → P3

The onboarding work was completed on taxi and proved before the main split.

P0 — semantic inference hints

Commit:

7a2a112

Package:

src/broadway/onboard/

Inference produces ephemeral hints such as:

dtype

null rate

cardinality

identifier score

datetime candidate

numeric vs categorical

suggested role

These are evidence only.

They pre-fill onboarding questions but never become contract truth until confirmed.

P1 — ds-pipeline init

Commit:

1041b3e

init is the full onboarding/scaffolding command.

It supports interactive questions plus flag overrides for CI/non-interactive use.

It writes:

configs/dataset/*.yaml

configs/analysis/*.yaml

configs/experiment/*.yaml

artifacts/discover/profile.json

profile lineage sidecar

discover remains the raw contract proposer and is not replaced.

Important UX rule:

A datetime column does not automatically imply time-based split semantics. The relevant split datetime must be explicitly confirmed.

P2a — ownership correction

Commit:

98dea62

The generic step schemas were made genuinely dataset-agnostic.

Taxi-specific ETL/features/stats config fields were removed from core step models and relocated to taxi-owned config.

group_column / group_values moved to the appropriate analysis-level hypothesis structure rather than remaining generic step fields.

P2b — generic feature default + extension hook

Commit:

badeab1

Shipped:

generic datetime builders

FeatureConfig.builder_module

lazy builder-module import at feature execution

loud failure on import / malformed registry / collisions

no silent overriding of generic builders

build_generic_feature_specs(contract, experiment)

contract-derived engineered output validation

Extension model:

generic builder registry
    +
explicit experiment builder_module
    ↓
merged registry

Custom builders extend; they do not silently replace core behavior.

P3 — non-taxi end-to-end proof

Commit:

7be0dd3

tests/test_onboarding_e2e.py proves that a synthetic non-taxi CSV can:

init
→ etl
→ contracts
→ baseline
→ features
→ train
→ evaluate

with:

local MLflow file store

no taxi names

no if dataset != taxi

zero edits to src/broadway/**

This is the acceptance proof behind the main ownership split.

Taxi dogfood findings already completed

The taxi dataset was used to prove evidence → decision → cleaning → re-profile → lineage.

Datetime anomaly

Observed:

18 pre-2024 rows

Decision:

drop_invalid_datetime

Outcome:

filter before training

re-profile confirms zero pre-2024 rows

Passenger count anomaly

Observed:

101,802 rows with passenger count 0

20 rows above 6

raw accepted dtype remains float64

Decision:

cast_and_bound_passenger_count

Outcome:

filter to integer-valued 1–6 in ETL

keep float64 at processed boundary

cast to int64 in feature layer

A real-data ETL rerun exposed a NaN casting bug; the NaN-safe filter fix proved the loud-failure hardening useful.

Distance-duration anomaly

Observed:

18 rows with:

trip_distance roughly 0.01–0.04 mi

duration 63–171 min

Raw inspection showed they are mostly legitimate flat-rate / negotiated trips:

ratecode 2 — JFK flat-rate

ratecode 3 — Newark flat-rate

ratecode 5 — negotiated fare

Decision:

distance_duration_is_flat_rate

Outcome:

keep

no cleaning change

trip_distance is unreliable for flat-rate regimes

any ratecode/flat-rate handling is a future feature-design choice, not an ETL rule

The lineage now contains three useful semantic cases:

invalid → drop
invalid domain values → filter/bound
suspicious but legitimate → keep

This is important: Broadway records analytical judgment, not merely cleaning actions.

Current active work — structural cleaning boundary

Before resuming the manual statistical walkthrough, the next concrete implementation is an explicit raw → validated canonical dataset stage inside ETL.

The purpose is to make representation cleanup explicit and traceable before statistical/domain cleaning.

Goal

ETL's first responsibility becomes:

raw
→ structural validation
→ deterministic structural cleaning
→ strict canonical validation
→ canonical artifact
→ split

No domain/outlier cleaning belongs here.

No DecisionRecord is created for deterministic mechanics.

Locked design decisions

Step structure

Enhance the existing etl step.

Do not add a separate canonicalize pipeline step yet.

A separate step is only warranted later if canonicalization needs independent execution/reuse.

Raw preservation

Record-only preservation.

Do not add _raw shadow columns to the canonical dataset.

The immutable source file preserves the original values; typed evidence + lineage carry what changed.

Missing encodings

Generic configured default:

missing_encodings:
  - ""
  - "NA"
  - "N/A"
  - "null"
  - "None"
  - "?"

Do not include "-" globally because it can be legitimate data.

Dataset-specific additions require actual evidence.

Duplicate definition

Only exact all-column duplicates:

df.duplicated()

Do not use key-based deduplication at the structural layer; that requires domain identity semantics and belongs to authored judgment/policy.

Validation boundaries

There are two distinct validation points.

Raw/input boundary

Validate structural expectations compatible with source representation:

expected columns

allowed extra/missing columns according to contract policy

null constraints that make sense before representation parsing

Do not apply the strict canonical dtype schema here because CSV date strings may legitimately represent a canonical datetime column.

Canonical/output boundary

Apply the authoritative full contract/Pandera validation after representation cleanup.

Canonical dtype is authoritative here.

Parse failures

Parse failures:

become NaT / NaN

are recorded as typed evidence

are not automatically dropped

Only explicit null policy removes rows.

Feature-column nulls, including failed datetime parses, remain unless another authored policy later addresses them.

Target-null rows

Target-null is currently the sole deterministic null-driven row drop.

It must be explicit and reflected in the audit.

Structural-cleaning evidence models

Planned package:

src/broadway/cleaning/

ParseFailure

Fields:

column

count

examples: list[str]

target_dtype

Represents non-missing raw values that failed the requested representation parse.

StructuralCleanResult

Fields:

audit: TransformAudit

parse_failures: list[ParseFailure]

missing_encodings: dict[str, list[str]]

canonical_path: str

Important:

missing_encodings records only encodings actually observed in each column.

The configured policy set remains in EtlStep.

The result artifact records what happened, not the policy definition.

Structural-cleaning execution order

The exact order matters.

Use:

load raw
→ raw structural validation
→ drop exact raw duplicates
→ standardize configured missing encodings
→ parse datetime columns
→ drop target-null rows
→ strict canonical validation
→ save canonical artifact
→ persist StructuralCleanResult
→ persist ETL lineage sidecar / TransformAudit
→ split canonical into train / validation

Why this order:

duplicate means genuinely identical source rows

missing representations are normalized before parse-failure detection

"NA" / "null" should not be falsely reported as datetime parse failures

parse failure is a genuine non-missing parse problem

target-null policy runs after representation normalization

canonical contract validates the final accepted representation

Planned structural functions

src/broadway/cleaning/structural.py

standardize_missing(...)

Maps only configured encodings to missing values.

Returns:

cleaned series

encodings actually observed

No inferred missing semantics.

parse_datetime(...)

Parses with coercion, but explicitly detects:

raw value was non-null
AND
value was not a configured missing representation
AND
parse produced NaT

Returns parsed data + typed ParseFailure evidence.

No row drop.

remove_duplicates(...)

Exact all-column duplicate removal.

Adds an explicit audit reason.

remove_target_null(...)

Drops rows with null target only.

Adds an explicit audit reason.

Existing ETL / contract pieces to reuse

Do not rebuild these:

data/cleaner.py::clean() already accounts for target-null + duplicate drops

etl/module.py already emits TransformAudit + lineage

contracts/pandera.py::build_raw_schema(contract)

contracts/checks.py

lineage/models.py::TransformAudit

enforce_drop_fraction

The structural-cleaning work should refactor/reuse these pieces rather than establish parallel implementations.

Tests required for structural cleaning

At minimum:

datetime parsing records failures and produces NaT without dropping rows

missing standardization maps only configured encodings and reports only observed encodings

exact duplicate + target-null removal are visible in TransformAudit.reasons

raw boundary accepts a date represented as a string when the canonical contract expects datetime

canonical boundary strictly validates the parsed dtype

ETL persists StructuralCleanResult

canonical dataset is saved and validated before splitting

ETL lineage sidecar references the canonical transformation/audit

parse failures do not become implicit row drops

No DecisionRecord should be emitted by these mechanics.

Statistical work — next after structural cleaning

The user explicitly wants to perform statistical tests one by one manually first in order to understand the process before Broadway automates recommendations.

Do not jump directly to the group-comparison router.

The intended manual group-comparison walkthrough is:

define groups and estimand

inspect sample sizes + descriptive statistics

run normality diagnostics

run Levene

run classical ANOVA

run Welch ANOVA

run Kruskal-Wallis

compute effect size

if warranted, run Games-Howell

compare conclusions

record why a method would be chosen

Broadway should calculate and expose evidence first.

The recommendation layer comes after real manual use.

Relevant statistical philosophy:

AnalysisSpec
    ↓
Diagnostics
    ↓
Recommendation / method options
    ↓
Analyst decision
    ↓
Statistical execution
    ↓
typed result
    ↓
plots / reports

Rule:

Statistical methods calculate. Recommendation logic advises. Decisions record judgment. Renderers present.

Do not encode:

if normal → method A
else → method B

Normality tests, especially at large N, should be evidence rather than hard gates.

Consider:

estimand

sample size

dependence

variance

outliers

effect size

confidence intervals

robustness

Statistical capabilities already implemented

src/broadway/stats/ already contains:

classical ANOVA

Welch ANOVA

Kruskal-Wallis

Levene

normality diagnostics

Games-Howell

effect sizes

guards

OLS

HC3 robust SE

diagnostics

time-series helpers

AnalysisPlan

The current stats pipeline step still effectively runs a single ANOVA.

Missing recommendation-layer work includes:

Group-comparison router

Future behavior should roughly:

inspect evidence

default toward Welch where appropriate

treat non-normality as a warning/evidence signal, not a hard gate

combine omnibus result + effect size + diagnostics

recommend Games-Howell when post-hoc comparison is warranted

record the eventual analyst choice

Do not implement this until the manual walkthrough has been dogfooded.

OLS remediation ladder

Future exploratory concept:

base OLS
→ inspect diagnostics
→ transformation candidate
→ robust SE
→ interaction candidate
→ escalate to nonlinear model only with logged evidence

This is also deferred until the manual statistics flow is better understood.

Feature / extension architecture

The generic feature path is the default platform path.

Custom project extensions are explicit.

FeatureConfig.builder_module declares a custom module.

Import occurs lazily at feature execution.

Requirements:

import failures are loud

BUILDERS must exist and be valid

custom names may not collide with generic names

custom builders extend only

no if dataset == ...

no convention-only magic imports

Generic feature scope remains conservative:

numeric passthrough

datetime decomposition

categorical encoding

Do not automatically introduce:

log transforms

interactions

clipping

imputation

feature selection

semantic filters

Those require evidence and/or authored intent.

Cleaning boundary

The intended boundary is now:

raw source
    ↓
structural canonicalization
    ↓
validated canonical dataset
    ↓
profiling / EDA / slices
    ↓
analytical decisions
    ↓
domain cleaning / feature decisions
    ↓
modeling / inference

Initial structural cleaning should correct representation problems, not hide analytical judgments.

Hold off automatically on:

outlier dropping

imputation

winsorization

scaling

rare-category merging

feature selection

leakage-sensitive removal

semantic filters

transformations that should be fit only on training data

Training / external tools philosophy

Broadway orchestrates mature tools rather than replacing them.

Current stack:

Pandera → dataframe validation

Pydantic → configuration/contracts

Optuna → HPO

MLflow → experiment/model/artifact tracking

Kubernetes → scalable execution

sklearn / statsmodels / SciPy → modeling/statistics

Broadway should own semantics and traceability:

data contract

feature contract

analytical intent

HPO configuration ownership

evaluation/promotion policy

decision lineage

Avoid becoming “our wrapper around every library.”

Testing philosophy

Prefer:

typed contracts

scenario tests for decision/recommendation logic

regression tests for silent-failure classes

full-suite gates after each architectural phase

real-data smoke checks after infrastructure changes

non-taxi E2E tests for abstraction claims

Fixture rule:

Use contract-driven valid fixtures where useful, but also explicit malformed fixtures so tests do not become tautological.

Git / execution style

The user prefers:

one coherent commit per phase

tightly scoped agent contracts

full suite green before moving on

no scope creep

docs updated with architecture changes

checkpoints after meaningful real-data runs

Current development pattern:

build / dogfood on taxi

prove the abstraction against non-taxi data

promote only genuinely generic work to main

Do not merge taxi-specific judgment into main.

Immediate next task

Implement the structural-cleaning boundary inside ETL exactly as locked above.

Recommended phase shape:

S0 — typed structural-cleaning evidence models
S1 — structural functions + tests
S2 — ETL wiring + raw/canonical validation split
S3 — lineage/artifact integration + full-suite gate
S4 — real taxi smoke run + inspect artifacts
STOP

Keep this pass narrow:

representation cleanup only

deterministic mechanics only

typed evidence

TransformAudit

canonical output validation

no outlier/domain cleaning

no statistical recommendation router

no DecisionRecords for deterministic operations

After that passes on real taxi data, resume the manual statistical walkthrough one test at a time.

One-sentence project description

Broadway is a traceable workflow framework for tabular statistical analysis and predictive modeling that preserves the path from raw data and analytical intent through evidence, decisions, features, models, and results.

Alternative framing:

A golden-path framework for reproducible tabular data science where analytical judgment is facilitated rather than hidden, and every consequential decision remains traceable.