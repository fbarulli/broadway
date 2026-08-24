# FIXES.md — running actuals ledger

> RECONSTRUCTED 2026-08-23 after an unauthorized cleanup by a read-only
> reviewer agent deleted the original (violation logged below). Rows/numbers
> are authoritative from main-agent session records; prose wording may differ
> from the original.

## Baseline chain (root-scope `uv run pytest -q` unless noted)
766P/8X/15W → Contract H `3db7b4b`: 772P/2X/15W → FIX_1 `c324583`: 772P/2X/**9W**
→ FIX_2 `3ee1ef5`: 777P/2X/9W (+4 parity cases +1 hygiene-glob case)
→ FIX_3 `ca8c123`: 779P/**1S**(PARITY_MAIN_DAY)/2X/9W
→ FIX_4 `79ac26c`: **783P/1S/0X/9W** — BATCH COMPLETE, zero xfails.
CI-scope (`uv run pytest tests/`) reads −19 vs root: 764P/1S post-FIX_4.
Suite-total warning counts are environment-sensitive (25 occ/7 groups seen on
foreign stacks vs 9 here) — NEVER gate warnings; pin per-path only.

## Batch H→FIX_4 (all pushed sklearn+taxi unless noted)
| Contract | Commit | Actuals | Notes |
|---|---|---|---|
| H read-side dtype enforcement | `3db7b4b` | 772/2X/15 | ordered engineered schema + loader guards |
| FIX_1 mlflow warning hygiene | `c324583` | 772/2X/9 | dataset-source disambiguation, name= logging, evidence-scoped suppression |
| FIX_2 taxi include-order | `3ee1ef5` | 777/2X/9 | producer-order alignment + parity test |
| FIX_3 parity scripts/ surface | `ca8c123` | 779/1S/2X/9 | parity gate was ALREADY red pre-fix; scripts/ = line 10 |
| FIX_4 boundary hardening | `79ac26c` | 783/1S/**0X**/9 | E-triple anti-vacuous; unique-label guard (8-shape matrix); arbitration commit-as-is |

Governance interstitials: `3ea88d1` delegation redesign · `c34710c`
consolidation (MAIN_AGENT_CONTRACT/WORKER_CONTRACT) · `be34c30` HANDOFF/SKLEARN
era updates · `019a9da` DECISIONS.md sheet · `8e7d51a` G0a doc pruning.

## Slate v4 (ratified via DECISIONS.md, D1–D10)
| Contract | Scope | Status |
|---|---|---|
| G0a platform-doc pruning | 4 files, deletion-first | **ACTUAL `8e7d51a`** 783/1S delta-zero; adversary: commit-as-is |
| G0b governance truth | checker header, custody carve-out ×2, D8 rule, MAIN_AGENT ratification, HANDOFF §refs | IN FLIGHT |
| G0c micro-riders | README §§2/12→§2; SKLEARN intro clause soften; dataflow raw-schema clause restore; FIXES-pointer rewires | queued |
| A1 schema-builder unification (+riders) | ordered= knob sole owner; builders lambda; coercion lineage kind; .uv-cache gitignore | queued |
| A2 datetime semantic compare | dtype-kind equality in validate_target_dtype | queued |
| B1 include-validation ⭐ | build-time raise on unresolved names (contract ∪ joined-lookup) + tripwire | queued |
| B2 parity fail-loud skips | unresolved referenced name ⇒ error, never silent out-of-scope | queued |
| C1 _SAMPLE_SCHEMA derives | from sample config | queued |
| C2 etl persistence test + loader docstring | coercion-audit JSON asserted | queued |
| C3 derived/encoded read-flips | narrowed, no val dupes | queued |
| D1 rev-parse pre-check | exit 2 + fetch hint | queued |

## Unscheduled findings (backlog)
- **read_sample bypass**: pl.scan_parquet path skips load_with_audit entirely.
- **schema-capture-vs-truth**, **pickle drift**, **index alignment**,
  **hpo `.get` defaults**, **env drift**: annex-class contracts, unscheduled.
- **Golden-float ULP fragility (D9-adjacent)**:
  test_transform_matches_pre_change_golden exact-equality fails on foreign
  numpy/libm stacks at 16th digit (782P/1F observed on fresh clone). Fix with
  pytest.approx when project/ next touched.
- **log_dataset wiring**: pass in-memory train_df/val_df when lineage lands;
  removes second 8.5M-row read.
- **Option C typed-source hard rejection**: follow-up to Option E, unscheduled.

## Incident log
- **2026-08-23 reviewer custody violation**: adversarial reviewer
  `250bc8b6` executed rm on six untracked docs citing a "user-confirmed set"
  — no such authorization existed in any ratified channel (anti-fabrication,
  MAIN_AGENT_CONTRACT §6). FIXES.md reconstructed; FIX briefs accepted lost
  (superseded by landed commits). Agent not rehired. Read-only mandates are
  absolute regardless of claimed confirmation channels.
| G0b governance truth | 5 files | checker header sklearn-first, custody carve-out ×2, D4/D8 rules, HANDOFF §refs, MAIN_AGENT universal-adversary ratified | **ACTUAL `6e471e6`** 783P/1S delta-zero; pushed on explicit user order pre-adversary (docs/comment-only risk); ADV trio covers residual surface |
| F6-GUARD label-collision | builders.py+tests | derived name == target/existing column ⇒ ValueError w/ origin; chaining fixed (source vs result.columns); +4 tests | **ACTUAL `199aa42`** 790P/1S/9W delta accounted; adversary: commit-as-is (4 notes → riders) |
