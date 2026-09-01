# STATE.md — current operational control record

`STATE.md` holds only active custody and retryable operational intent. Git is
the authority for landed history; GitHub Project #4 is a mirror, never evidence.

## CURRENT

| id | kind | status | owner | custody | updated | source | github_item | mirror_state | summary |
|---|---|---|---|---|---|---|---|---|---|
| STATE-20260830-001 | checkpoint | open | main agent | main agent | 2026-08-30 | state-foundation seed | pending | pending | Seed record for the private STATE-to-Project mirror; no worktree claim is implied. |
| STATE-20260901-008 | decision | open | human owner (deferred 2026-09-01) | main agent | 2026-09-01 | owner chat decision; risk-sweep inventory; git status census | PVTI_lAHOAZFnCc4Bhhjqzg46WMo | synced | TAXI INFORMATION DISPOSITION — DEFERRED by owner, tracked not acted on. Trigger: euromonitor is the active mission; taxi surfaces are superseded legacy. Inventory of what still exists (data already deleted in 1071dbf): (1) taxi experiment code — 5 series under project/experiments (fare_prediction, univariate/fare_amount_trip_distance, multivariate, more_modeling, mlflow, polynomial_regression_et_all), all hard-wired to taxi schema, cannot run on euromonitor; (2) taxi configs — project/config/{dataset,analysis,experiment,sample,project}/*taxi*.yaml + fare_prediction*.yaml; (3) shared loader project/working.py + working.yaml still bound to taxi columns (pickup_datetime/dropoff_datetime/fare_amount); (4) taxi-bound project tests under project/tests referencing taxi config paths. KEY FACT: deletion is not data loss — every file recoverable from git history. RECOMMENDED WHEN TRIGGER FIRES: delete the 5 taxi series + taxi configs, replace project/working.py+yaml with euromonitor binding, fix or delete taxi-bound project tests. ALTERNATIVE (if archival wanted): move taxi series under scratch/ and build a parallel euromonitor loader. Deferred decision recorded so the inventory is not re-derived; revisit when taxi surfaces block the euromonitor pipeline or the owner calls it. |
| STATE-20260901-009 | decision | open | human owner | main agent | 2026-09-01 | gate-registry euromonitor surface verification | PVTI_lAHOAZFnCc4Bhhjqzg479oQ | synced | meta.head re-stamp: brief said 7bec391 but that was stale (HEAD had already advanced past it); the parent-stamp law stamps the current HEAD for a dirty registry, so meta.head auto-tracks HEAD (now 9dcfebe). No human ruling required - mechanical. |
| STATE-20260901-010 | checkpoint | open | main agent | main agent | 2026-09-01 | gate-registry euromonitor surface verification | PVTI_lAHOAZFnCc4Bhhjqzg479pk | synced | euromonitor surface gate verification complete (221 tests green). Resolved: GATE-SURF-103 owner gtin_coverage() corrected to plot_barcode_coverage() after the gtin-to-barcode refactor; the taxi-disposition decision row date token was reworded for the 8-hex probe. The stale duplicate counts block was already resolved in the working tree (total 144, by_band verified). |
| STATE-20260901-011 | decision | open | human owner | main agent | 2026-09-01 | gitbook docs-as-code integration investigation | PVTI_lAHOAZFnCc4Bhhjqzg48ink | synced | GitBook docs-as-code integration investigated: GitBook syncs repo markdown on push via SUMMARY.md plus optional gitbook.yaml; repo has ~80 md files (contracts, ledger, reports, notes) and no SUMMARY.md yet. Open decisions: scope (reference docs only vs include internal STATE/FIXES/DECISIONS), hosting (GitBook SaaS vs mdBook/MkDocs), visibility (public/private), branch (main vs euromonitor). Recommendation: GitBook SaaS with a curated SUMMARY.md covering contracts and platform docs. |
| STATE-20260901-012 | decision | open | main agent | landed 1af7bd6/9dcfebe | 2026-09-01 | 04c_blocking_ab.py decisive experiment 2026-09-01 | PVTI_lAHOAZFnCc4Bhhjqzg48miE | synced | Blocking A/B on 29,320 ground-truth pairs: only 36.9% of true pairs have a detected volume on BOTH members, so ANY blocking key containing volume caps recall at ~0.63 (A brand x strict x vol 0.586, B brand x macro x vol 0.615, C brand x vol 0.628). Dropping volume: brand x macro recall 0.961 at 2.5M candidates; brand-only 0.983 at 3.0M. DECISION for step 03: block on brand x macro (recall-first), volume becomes a SCORING feature (99.0% within-barcode agreement when both sides have it). Highest-leverage next improvement = volume COVERAGE (25,594 rows Missing, 02d), not a better category key. |
| STATE-20260901-013 | checkpoint | open | main agent | landed 1af7bd6/9dcfebe | 2026-09-01 | 04b_category_utility.py 2026-09-01 | PVTI_lAHOAZFnCc4Bhhjqzg48mk8 | synced | Category utility audit: strict category matches 93.7% of true pairs vs macro 97.9% (chance 7.1%/20.0%) — strict blocking would reject 6.3% of real matches. MI with match label: brand 0.6364 > strict 0.4317 > macro 0.3734. Cramer's V vs retailer: strict 0.184 (low leakage), macro 0.242. Verdict: block coarse (macro), score fine (strict), weight brand highest; greedy selection under 5M budget confirmed brand x macro. |
| STATE-20260901-014 | checkpoint | open | main agent | landed 1af7bd6/9dcfebe | 2026-09-01 | 05b_exact_duplicates.py + 06_dedupe.py 2026-09-01 | PVTI_lAHOAZFnCc4Bhhjqzg48mnw | synced | 62.7% of the catalog is repeat (retailer,barcode) marketplace listings. 13,519 rows in retailer+title dup groups, only 4,439 also match price => ~9,080 are DISTINCT offers (Gittigidiyor listed one title 143x), so dedupe is price-AGGREGATION not noise-removal. Tiered dedupe T1 retailer+barcode / T2 retailer+title+price / T3 retailer+title with deliberate representative (barcode > completeness > lowest price); 690 price-varying offer groups flagged, never silently merged. 71,623 -> 26,998 rows; dataset_deduped.csv is the matching input. |
| STATE-20260901-015 | checkpoint | open | main agent | landed 1af7bd6/9dcfebe | 2026-09-01 | 05b + 06b_mislabeled_barcode_report.py 2026-09-01 | PVTI_lAHOAZFnCc4Bhhjqzg48mqc | synced | 339 cross-retailer exact-title groups carry >1 UNIQUE real barcode (5.9% of 5,782 exact-title cross-retailer groups). Corrected from an earlier 887 (old probe counted empty strings as a 'second barcode' after fillna('')). Flagged for human review, never merged — barcode is ground truth. Explains the 04 negative-pair max similarity 0.704. Artifact: 06b_conflicting_barcode_groups.csv; 5,443 clean groups = calibration positives. |
| STATE-20260901-016 | checkpoint | open | main agent | landed 1af7bd6/9dcfebe | 2026-09-01 | 04_tfidf_matching.py + 06c_validation_sets.py 2026-09-01 | PVTI_lAHOAZFnCc4Bhhjqzg48mtM | synced | TF-IDF cosine validated on ground truth: positive cosine mean 0.603 / median 0.626; negative mean 0.017 / median 0.000. Separation P(pos>neg)=0.968; largest threshold keeping 95% recall = 0.12 (fpr 3.4%). Demo pair is real ground truth (barcode 5021554989646): literal strings 0.476, actual rows 0.379. Validation sets carved: 5,443 calibration positives + hard band 0.3-0.8 (15,674 pos / 356 neg). |
| STATE-20260901-017 | checkpoint | open | main agent | landed 1af7bd6/9dcfebe | 2026-09-01 | 01c_sparsity_noise.py + 01d_description_missingness.py 2026-09-01 | PVTI_lAHOAZFnCc4Bhhjqzg48mv4 | synced | Data ceilings (data-side, not code-side): price is LOCAL CURRENCY across 19 countries (median 35.00 TRY vs 2.39 EUR; CV 5.0, 6% beyond 1.5xIQR, max 9,998) — unusable without FX normalization (pair-agreement 0.36). Barcode coverage 42% (1,747 anomalous-length values). Volume coverage 36% on true pairs (the blocking killer). Description missing 16.8%, concentrated in Gittigidiyor (72%) + amazon (1,534 rows); 7,760 rows miss both description and barcode. |
| STATE-20260901-018 | checkpoint | open | main agent | landed 1af7bd6/9dcfebe | 2026-09-01 | 02 + 02d_measurement_validation.py + _text.py 2026-09-01 | PVTI_lAHOAZFnCc4Bhhjqzg48myM | synced | Extractor v3 + validation layer: EU-decimal-with-space fix ('0, 33l' = 0.33 L, 103 rows corrected; count-list guards keep 'case of 24, 500ml' = 500), dot-decimals never take a space ('pH 9.0 bottle 600 ml' = 600), mg/zero-claim weight hints excluded (1,932 -> 1,644). Category taxonomy (24 categories, default volume) replaced the substring gate; multipack totals flagged (20/163 = 12.3%). Validation: Missing 25,594 / Flagged 163 / Valid 45,866; Valid+Flagged = canonical_volume_detected exactly; 6 sanity checks PASS; 68 tests pin the layer. |
| STATE-20260901-019 | hazard | blocked | human owner | main agent | 2026-09-01 | euromonitor fix push attempt | PVTI_lAHOAZFnCc4Bhhjqzg48oBA | synced | Push of euromonitor fixes (46e2135) blocked by pre-push CI gate: test_probe_f_meta_head_parent_stamp (gates.yaml meta.head 9dcfebe vs required 46e2135, parent-stamp law) and test_active_state_has_no_legacy_tail_before_events (STATE.md structure). Both stem from uncommitted governance state; needs human ruling before re-push. |
| STATE-20260901-020 | hazard | open | main agent | main agent | 2026-09-01 | euromonitor dedupe review | PVTI_lAHOAZFnCc4Bhhjqzg48oC0 | synced | 06_dedupe.py T1 collapsed all missing-barcode products per retailer (drop_duplicates NaN equals NaN), silently destroying 41,307 rows (71,623 to 26,998). FIXED in 46e2135: T1 gates on _has_bc; dataset_deduped.csv regenerated (71,623 to 61,404). Downstream results built on the buggy deduped data need re-derivation. |
| STATE-20260901-021 | checkpoint | open | main agent | main agent | 2026-09-01 | euromonitor review | PVTI_lAHOAZFnCc4Bhhjqzg48oEs | synced | euromonitor code-quality fixes landed in 46e2135: 06b dead n_barcodes lambda; 07 unused SEED; _text dead fl.oz fl-oz keys and norm_unit; 02b 03 03b import coupling (import_module, parents[3] path, sys.path.insert); 04c NO_CATEGORY sentinel; SEED SSOT in 01b 04 04b 06c. Gate: test_euromonitor_volume.py 68 passed. |
| STATE-20260901-022 | decision | open | human owner | main agent | 2026-09-01 | euromonitor review | PVTI_lAHOAZFnCc4Bhhjqzg48oGQ | synced | Deferred euromonitor findings (behavior-changing, need owner decision): 01b unigram TF-IDF diverges from _matching build_vectorizer; 07 _build_pairs triplicated across 04 04b 07; 06c per-pair score_pairs inside sampling loop. No fix applied pending decision. |
| STATE-20260901-023 | hazard | open | main agent | main agent | 2026-09-01 | full pytest suite | PVTI_lAHOAZFnCc4Bhhjqzg49wMs | synced | Full pytest suite reports 35 errors, all in project/tests/test_euromonitor_volume.py: ImportError cannot import name load_euromonitor from _common resolved to project/experiments/more_modeling/_common.py. Root cause: the exec-based loader execs 02_volume_normalize.py whose bare from _common import load_euromonitor collides with the more_modeling same-named module cached in sys.modules when the full suite runs. Pre-existing isolation bug, unrelated to this session dedupe-wiring and encode_corpus commits; volume tests pass in isolation and paired with test_nlp. Fix: load extract_volume_ml from _text directly instead of exec-ing the whole script. |

## Access protocol

- Main alone uses `state_records.py record add|update|sync`; workers and
  reviewers only read and report.
- Local CURRENT intent is written before its mirror. A failed mirror leaves
  `mirror_state=pending`, and `record sync <id>` retries the same record.
- The helper changes only CURRENT; `## EVENTS` and the historical archive are
  immutable through this interface.

## Retention

The pre-foundation record is preserved verbatim in
`agents/ledger/archive/2026-08.md` from source commit `7dcb34f`. Current
operational history stays in git and Project #4 mirrors.

## EVENTS

Resolution registry for EVENT-line event-ids (third namespace — see the
citation rule in MAIN_AGENT_CONTRACT.md). A FULL-LINE `EVENT: issues/<n>#
issuecomment-<m> event-id <8hex>` line is valid iff a UNIQUE row exists here;
role vocabulary is no escape in this namespace, and duplicate event-id rows
are a probe violation. Superseded rows may outlive their EVENT lines.
Created_at values verified via gh api at pilot close; board-row created_at
values verified via api at seeding (store-then-hash). Provenance: the first
six rows are pilot backfills entered via owner gh-api writes (two genesis
events, D19-D21 backfills, custody-baseline); the five 2026-08-24T20:43Z
rows are CHANGE BOARD (#5) seeds R1-R5 posted this session. Each row's
`type` keeps its per-event ruling class, one unique row per legacy
narrative id. One further class exists: reviewer-authority rows registering
HARNESS-ERA AGENT AUTHORITIES so the ratified TIER-GATE grammar can resolve
Reviewer:-trailer verdicts; such a row carries NO GitHub comment provenance
(out-of-band verification only, grandfathered per the agent-id namespace
disposition lean), and its id pins the canonical posted form: sha256
first-8 over the row line with the id cell itself blanked — byte form:
single-space cells (`| |`), no trailing newline; any other blanking
reproduces a DIFFERENT hash and must be re-pinned.

| event-id | issue | comment-id | created_at | type | supersedes |
|---|---|---|---|---|---|
| b16fb9ca | issues/3#issuecomment-5398091966 | 5398091966 | 2026-08-24T16:19:07Z | authorization | - |
| e1f7cc62 | issues/3#issuecomment-5398093447 | 5398093447 | 2026-08-24T16:19:14Z | authorization | - |
| 493e21ce | issues/4#issuecomment-5398092241 | 5398092241 | 2026-08-24T16:19:08Z | amendment | - |
| 3afcd9b1 | issues/4#issuecomment-5398092508 | 5398092508 | 2026-08-24T16:19:09Z | ratification | - |
| 7595cb13 | issues/4#issuecomment-5398092820 | 5398092820 | 2026-08-24T16:19:11Z | verdict | - |
| 555b6fb8 | issues/4#issuecomment-5398093134 | 5398093134 | 2026-08-24T16:19:13Z | verdict | - |
| bb8c548b | issues/5#issuecomment-5401138010 | 5401138010 | 2026-08-24T20:43:14Z | board-row | - |
| e277f63a | issues/5#issuecomment-5401138608 | 5401138608 | 2026-08-24T20:43:16Z | board-row | - |
| ebf1c913 | issues/5#issuecomment-5401139225 | 5401139225 | 2026-08-24T20:43:19Z | board-row | - |
| 392fa146 | issues/5#issuecomment-5401139768 | 5401139768 | 2026-08-24T20:43:22Z | board-row | - |
| ae44dbfd | issues/5#issuecomment-5401140343 | 5401140343 | 2026-08-24T20:43:24Z | anomaly | - |
| 8ee2eb96 | issues/5#issuecomment-5401490304 | 5401490304 | 2026-08-24T21:17:00Z | board-row | - |
| de82c84b | issues/5#issuecomment-5401490717 | 5401490717 | 2026-08-24T21:17:02Z | board-row | - |
| 64864f79 | issues/5#issuecomment-5401491180 | 5401491180 | 2026-08-24T21:17:05Z | board-row | - |
| 434a8be2 | issues/5#issuecomment-5401491636 | 5401491636 | 2026-08-24T21:17:08Z | board-row | - |
| 357eb775 | issues/5#issuecomment-5401492192 | 5401492192 | 2026-08-24T21:17:11Z | board-row | - |
| a682f9f5 | issues/5#issuecomment-5401492630 | 5401492630 | 2026-08-24T21:17:14Z | board-row | - |
| 392683bf | issues/5#issuecomment-5401517725 | 5401517725 | 2026-08-24T21:19:49Z | board-row | - |
| 162938c2 | issues/5#issuecomment-5401518149 | 5401518149 | 2026-08-24T21:19:52Z | board-row | - |
| facb63f1 | issues/5#issuecomment-5401518526 | 5401518526 | 2026-08-24T21:19:55Z | board-row | - |
| db84cc51 | issues/5#issuecomment-5401518902 | 5401518902 | 2026-08-24T21:19:57Z | board-row | - |
| 5e83df74 | issues/5#issuecomment-5401519275 | 5401519275 | 2026-08-24T21:20:00Z | board-row | - |
| 53c5b09a | issues/5#issuecomment-5401519805 | 5401519805 | 2026-08-24T21:20:03Z | board-row | - |
| 137464a0 | issues/5#issuecomment-5401524262 | 5401524262 | 2026-08-24T21:20:31Z | board-row | - |
| f0ef9baf | issues/5#issuecomment-5401532216 | 5401532216 | 2026-08-24T21:21:22Z | board-row | - |
| 5ca166eb | issues/5#issuecomment-5401532706 | 5401532706 | 2026-08-24T21:21:25Z | board-row | - |
| c9b9345b | issues/5#issuecomment-5402315704 | 5402315704 | 2026-08-24T22:26:22Z | board-row | - |
| a4d6f7c2 | issues/5#issuecomment-5402316139 | 5402316139 | 2026-08-24T22:26:25Z | board-row | - |
| 70859c98 | issues/5#issuecomment-5402316504 | 5402316504 | 2026-08-24T22:26:27Z | board-row | - |
| 007c632c | issues/5#issuecomment-5402316944 | 5402316944 | 2026-08-24T22:26:30Z | board-row | - |
| 384bc23c | issues/5#issuecomment-5402317352 | 5402317352 | 2026-08-24T22:26:32Z | board-row | - |
| 6094ce2b | issues/5#issuecomment-5402317727 | 5402317727 | 2026-08-24T22:26:35Z | board-row | de82c84b |
| 674adc2e | issues/5#issuecomment-5402318084 | 5402318084 | 2026-08-24T22:26:37Z | board-row | - |
| b625b5e0 | issues/5#issuecomment-5402318525 | 5402318525 | 2026-08-24T22:26:40Z | board-row | ae44dbfd |
| 3191279b | issues/5#issuecomment-5402421646 | 5402421646 | 2026-08-24T22:36:11Z | board-row | b625b5e0 |
| ed2f1fdf | issues/5#issuecomment-5402422093 | 5402422093 | 2026-08-24T22:36:14Z | board-row | ae44dbfd |
| 0aef1131 | issues/5#issuecomment-5407855643 | 5407855643 | 2026-08-25T08:50:59Z | board-row | bb8c548b |
| b0df65f1 | issues/5#issuecomment-5407856241 | 5407856241 | 2026-08-25T08:51:03Z | board-row | e277f63a |
| a886ae68 | issues/5#issuecomment-5407856844 | 5407856844 | 2026-08-25T08:51:06Z | board-row | 392fa146 |
| 3ca96ffd | issues/5#issuecomment-5408050227 | 5408050227 | 2026-08-25T09:05:32Z | board-row | ebf1c913 |
| 39de4245 | reviewer-authority:2d9ab1a1 — HARNESS-ERA AGENT AUTHORITY, grandfathered per the agent-id namespace disposition lean; verified out-of-band via its two delivered read-only review reports this session (ten-ruling batch + eight-packet batch); scope: valid Reviewer:-trailer resolution target for TIER-GATE | 0 | 2026-08-25T14:48:29Z | reviewer-authority | - |
