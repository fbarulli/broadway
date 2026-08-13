Broadway — Current Handoff

Repository state

Active development branch: taxi.

Latest verified Step 00.2 commit: 3aac481.

Latest verified suite at that point: 281 tests green.

Later doc inventory reported 293 effective tests after subsequent work; verify the current count with uv run pytest -q before relying on a number.

Keep scratch/untracked user files untouched unless explicitly asked.

What has been completed

1. Platform / ownership split

main is the generic platform boundary.

taxi is the demo + active dogfood branch.

Generic code must not require taxi-specific edits when dataset config changes.

Authored intent, observed evidence, runtime decisions, and deterministic mechanics remain separate concepts.

2. Onboarding / generic platform proof

Shipped before the current walkthrough:

semantic inference hints,

ds-pipeline init,

generic step schemas,

generic feature builders + explicit custom builder modules,

non-taxi end-to-end onboarding proof,

generic platform split to main.

3. Structural canonicalization

The generic ETL structural-cleaning boundary is implemented.

Order:

load input
→ raw structural validation
→ exact duplicate removal
→ configured missing-value normalization
→ datetime parsing
→ numeric coercion
→ target-null removal
→ strict canonical validation
→ canonical parquet
→ typed cleaning evidence / audit
→ lineage

Shipped behavior:

exact all-column duplicate removal,

configured missing encodings,

datetime parse failures recorded as typed ParseFailure,

numeric strings coerced according to contract dtype,

numeric parse failures recorded as ParseFailure,

parse failures become NaT / NaN and are not silently dropped,

target-null rows are the deterministic null-driven row drop,

strict canonical validation remains authoritative,

no nullable-integer schema weakening unless explicitly supported,

no column renaming in structural cleaning,

no domain/outlier cleaning in the structural layer.

4. Raw ingest is now a real CLI step

Step 00 was extended so the walkthrough can start before training_data.parquet.

Shipped:

Polars dependency added for the ingestion boundary only.

ds-pipeline ingest calls taxi process_data().

raw multi-file parquet loading uses a Polars lazy scan and materializes to pandas at the existing processing boundary.

All raw columns are read for now. Column projection is an authored-policy decision, not a hidden optimization.

process_config.py honors Broadway's config root instead of hardcoded configs/... paths.

ETL's ci_sample_size is applied only when CI=true.

normal/local ETL no longer silently truncates the canonical dataset to 50K rows.

ingest CLI logging was fixed so the user sees progress.

Observed full ingest run:

raw rows: 9,554,778

after taxi domain filters: 8,545,833

row removals reported:

distance/time: 217,392

duration: 40,195

passenger count: 751,358

training_data.parquet contains the six downstream taxi columns:

pickup_datetime

passenger_count

trip_distance

pickup_location_id

dropoff_location_id

trip_duration_minutes

5. Ingest / ETL lineage

Ingest now persists a real lineage record:

{
  "node_id": "ingest:taxi",
  "kind": "ingest",
  "artifact": "data/processed/training_data.parquet",
  "parents": ["dataset:taxi"]
}

ETL chooses its parent from persisted evidence:

if an ingest record exists: ingest:<dataset> is upstream,

otherwise standalone ETL retains dataset:<dataset> as upstream.

This avoids fabricated lineage.

6. Contract ownership cleanup — Step 00.1

Shipped in commit 415c7e0 with 272 tests green at that checkpoint.

row_count

Removed from authoritative DatasetContract.

Actual row counts remain observed evidence in DatasetProfile / TransformAudit / runtime artifacts.

Datetime dtype

Contract datetime representation is normalized to the semantic token datetime64.

aliases such as datetime64[us] / datetime64[ns] normalize at the model boundary.

Pandera validation already treats datetime units semantically.

Join completeness

Added typed JoinAudit evidence.

The actual execution lineage when lookups exist is:

dataset → ingest → join → etl

The taxi full-data join audit showed:

lookup keys matched 100%,

unmatched=0,

null_keys=0,

unmatched_rate=0.0.

7. Lookup-value quality — Step 00.2

Shipped in commit 3aac481 with 281 tests green at that checkpoint.

Added typed LookupValueAudit evidence, deliberately separate from JoinAudit.

Lineage:

dataset → ingest → join → etl
                    └→ lookup_value

Important distinction:

JoinAudit asks whether keys matched.

LookupValueAudit asks whether matched enrichment values are usable / deficient.

Full taxi evidence before canonical deduplication (8,545,833 rows):

Enriched column

Null

Sentinel

Affected

Affected lookup keys

Borough

1,093

Unknown: 27,053

28,146

264, 265

Zone

27,053

—

27,053

264

service_zone

28,146

—

28,146

264, 265

Ground truth in taxi_zone_lookup.csv:

LocationID 264 → Borough="Unknown", Zone="N/A", service_zone="N/A"

LocationID 265 → Borough=NaN, Zone="Outside of NYC", service_zone=NaN

The join is therefore structurally complete while the lookup values themselves contain deficient source values.

8. Lookup NA parsing gap identified — Step 00.3 is next in this area

Current lookup CSV loading uses pandas default NA recognition, so tokens such as "N/A" can become NaN before Broadway's authored missing-value policy applies.

Locked direction:

add LookupSpec.na_values: list[str],

read lookup CSVs with keep_default_na=False,

only explicitly authored na_values become null,

taxi lookup configs should explicitly list the desired missing tokens rather than inheriting pandas defaults,

record the applied na_values policy in lookup-value evidence.

This is an ingestion/parsing ownership issue, not a domain-cleaning decision.

9. Sample provenance / statistical sample roles

The sampling discussion was resolved conceptually as:

estimation sample → population/proportional interpretation,

diagnostic sample → enough small-group coverage for diagnostics,

do not silently treat these as interchangeable.

A first-class SampleSpec was designed around:

name,

role: diagnostic | estimation,

path,

optional description,

sample-specific column mapping when external materialized samples use different physical names.

Important principle:

the role belongs to the actual sample used by a result, not to the analysis contract itself.

Result/lineage provenance should carry both sample_name and sample_role.

Longer term, Broadway-produced samples should use the canonical schema; sample column mapping remains a bridge for external materialized samples.

10. Group-description walkthrough step

stats describe is the first formal statistical walkthrough step.

It persists typed group evidence including:

total N,

per-group N / mean / std,

all configured groups including n=0,

absent groups,

proportions,

imbalance_ratio as evidence only,

sample/source provenance.

Dogfood on the earlier 50K sampled canonical surfaced severe imbalance and missing Staten Island. This was evidence that led to the later sample-role / sampling-boundary work.

The rule remains:

imbalance is surfaced now as evidence; reweighting / resampling / method choice is later analytical judgment.

11. Stats library already implemented

Existing statistical engine includes:

classical ANOVA,

Welch ANOVA,

Kruskal-Wallis,

Levene,

normality diagnostics,

Games-Howell,

effect sizes,

group guards,

OLS,

HC3 robust SE,

residual diagnostics,

time-series helpers,

AnalysisPlan save/load.

Most of these are library functions only; they are not yet first-class walkthrough steps.

12. OLS diagnostic direction

The OLS diagnostic foundation is framed as:

Question → Evidence → Ramification

Diagnostic questions to eventually cover:

Is the mean relationship correctly specified?

Are observations independent?

Is error variance constant?

Is the result driven by influential observations?

Is residual non-normality problematic for inference?

Is multicollinearity problematic?

Is the sample sufficient for the intended inference?

Is the estimated effect plausibly confounded / subject to omitted-variable bias?

The first diagnostic capability designed is mean specification:

residuals-vs-fitted evidence,

later residuals-vs-predictor evidence,

no automated U/S-shape classifier yet,

do not treat a p-value as the sole functional-form diagnostic.

DiagnosticResult is intended to remain separate from generic AnalysisPlan.

13. Results-first product surface

The product-surface decision was changed explicitly:

Results are front and center. Machine evidence supports them.

Target layout:

reports/
  index.md
  results/
    describe.md
    normality.md
    levene.md
    anova.md
    welch.md
    kruskal.md
    effect_size.md
    games_howell.md
    ols_diagnostics.md
  figures/
  lineage/
    graph.md
    graph.json
    areas/
      data.md
      inference.md
      modeling.md

artifacts/          # machine-readable evidence / provenance

data/processed/     # materialized datasets
results/            # legacy caches / scratch outputs; phase out as things migrate

Human-facing reports and figures should be tracked in git so a checkout contains inspectable results without rerunning the analysis.

Machine JSON/parquet remain regenerable and out of the main human surface.

14. Results navigation / workflow direction

The earlier fixed sequence idea (describe → normality → Levene → ANOVA → Welch → ...) is not meant to become a universal router.

The more flexible target is a guided workflow with explicit gates:

automatic preparation
→ automatic evidence gathering
→ decision gate
→ analyst-chosen principal method
→ optional remediation / robustness / post-hoc branches
→ conclusion

Rule remains:

automate mechanics; facilitate judgment; record the decision.

For a group-comparison question, describe / normality / Levene can gather evidence before a method decision.

For an OLS-centered analysis, the principal path should instead be:

specify estimand / model question
→ fit base OLS
→ diagnostic bundle
→ decision gate
→ remediate / refit if warranted
→ primary result
→ supporting analyses only when useful

Do not force ANOVA/Welch/Kruskal to be the spine of an OLS analysis.

15. Documentation cleanup identified

A docs-only cleanup was planned after inventorying drift.

Known documentation fixes:

README: add missing CLI commands such as init / profile; name shipped audit / diagnostic types; correct SampleSpec; update config tree; note row-count ownership + datetime normalization.

dataflow.md: fix package ownership / directory tree and AnalysisContract location.

src/broadway/stats/API.md: document actual Pydantic AnalysisPlan, describe/guards, and truthful stats module behavior.

tests/README.md: rewrite around capability groups rather than a stale exhaustive flat list.

FIX.md: reconcile completed vs superseded checklist items.

No behavioral changes should be mixed into that docs pass.

Important unresolved architecture / design work

A. First-class canonical multi-source assembly

A naming mismatch (Borough vs pickup_borough) exposed a deeper future need.

Long-term direction:

one canonical dataset schema can be assembled from multiple physical sources,

canonical columns own semantic names regardless of origin,

joins are an explicit authored join plan,

multiple providers for one canonical field require explicit precedence/resolution,

no silent provider choice,

future taxi canonical naming should likely prefer semantic names such as pickup_borough so pickup/dropoff concepts are unambiguous.

This remains design-first, not an immediate refactor.

A future CANONICAL_SCHEMA.md should cover:

sources,

join plan,

canonical columns,

provider precedence,

validation boundaries,

migration compatibility,

at least one non-taxi example.

B. Data execution / performance abstraction

Current direction:

optimize physical access without changing analytical semantics,

Polars at raw ingestion is already acceptable,

pandas/NumPy remain the stats boundary,

PyArrow/Polars optimizations such as projection/predicate pushdown/batching are allowed when semantics are preserved,

column selection must be authored, not silently inferred,

canonicalization remains normal global DataFrame semantics for now,

streaming/chunked canonicalization is a separate future initiative if memory pressure requires redesigned global invariants.

Possible future execution interface should be backend-neutral enough that Spark could be an optional distributed backend later, but Spark is not needed now.

C. Lookup-value analytical decisions

The lookup audit has exposed real values requiring analyst/domain decisions, especially for borough analysis:

LocationID 264 → unknown borough,

LocationID 265 → outside-NYC zone with no borough.

These should not be silently repaired in structural cleaning.

A later borough-specific analysis may explicitly define an analytical population such as “resolved pickup borough is one of the five NYC boroughs” and record excluded counts/reasons in a DecisionRecord.

Immediate next work

1. Finish Step 00.3 — explicit lookup NA parsing

Implement explicit lookup na_values ownership so pandas defaults no longer silently determine lookup semantics.

Then rerun the data-foundation path and inspect the resulting lookup-value evidence.

2. Finish the results surface infrastructure

The human-facing product surface should be populated from persisted typed results, not terminal scraping.

Immediate first result should remain describe:

JSON machine evidence → artifacts/stats/,

human Markdown → reports/results/describe.md,

plots → reports/figures/,

reports/index.md should summarize the current question, latest result, strongest evidence, ramification/decision, and next available action,

lineage remains provenance, not the sole navigation UI.

The report renderer must not become a second source of truth; it renders persisted evidence.

3. Build the next analytical walkthrough step one at a time

Do not implement the full recommendation router yet.

For group-comparison dogfooding, likely next formal step:

normality evidence + Q-Q plots,

then Levene,

then reach a decision gate before deciding which omnibus method is the principal analysis.

Do not automatically run every available test just because it exists.

4. OLS principal-analysis walkthrough

If the intended principal analysis is OLS, prioritize this path rather than treating the group-comparison sequence as universal:

define estimand / outcome / predictor / unit / population / sample,

fit base OLS,

run Question → Evidence → Ramification diagnostics,

decision gate,

remediate/refit only where warranted,

persist the primary result + limitations + decision trail.

The first OLS diagnostic implementation should remain mean-specification evidence, not automated model selection.

5. Documentation-only reconciliation

Run the planned docs pass after the current behavior is stable. Keep it separate from code/refactors.

Current working principles to preserve

One authoritative owner per concept.

Authored intent ≠ observed evidence ≠ runtime decision.

No silent branches.

Loud failure for malformed/missing required state.

Deterministic mechanics produce policy + evidence + audit, not DecisionRecord.

Analytical judgment is recorded as a decision.

Reports render evidence; they do not become truth themselves.

Results are the primary human-facing product surface.

Lineage is provenance / execution history, not a substitute for analytical interpretation.

Build/dogfood narrowly, inspect real artifacts, then promote abstractions.