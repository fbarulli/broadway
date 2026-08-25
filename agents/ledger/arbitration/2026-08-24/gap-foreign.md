# GAP-FOREIGN — declared-foreign plane inventory (GAP-HUNTER GH-4) — 2026-08-24

Contract GH-4 · provenance: repo `/home/opc/ONE/broad-way`, branch `sklearn`, HEAD
`5016e93` · read-only sweep; **this file is the session's single created artifact**
(zero git ops performed; custody claims below cite ledger structure and
zero-reference greps, never `git status`). Two territories deliberately excluded
from all nine mapping lanes are inventoried so future ratification starts from a
complete map:

1. `experiments/more_modeling/**` — per-script register + RATIFICATION-GATE-CHECKLIST.
2. Top-level `k8s/` non-optuna siblings — PROPOSE-GH4-NN deployment-gate candidates,
   cross-referenced to packet-D defers BY NAME for later merge without duplication.

House-format rules honored: gate-row schema as in `agents/ledger/gates/*.md`;
`validated_by: []` marks an unexercised guard and **every FINDING carries `root:`**;
kubeconform-scope hole is CITED (GATE-INFRA-99, deploy-diff CI verdict), not re-found.

---

## PART 1 — Territory 1: `experiments/more_modeling/**`

### Territory facts

- 25 Python files: **24 runnable scripts** (`01`–`22` including subscripts `03_1`,
  `03_2`) + `_common.py` shared helper; plus `README.md` (11 lines, describes only
  questions 1–3 — stale relative to scripts 04–22).
- **Zero external references**: no hit for `more_modeling` in `tests/`, `scripts/`,
  `.github/`, `pyproject.toml` (incl. ruff targets), or `agents/ledger/gates.yaml`.
  Consequence: outside EVERY tier of `scripts/run_local_ci.sh` (ruff input list is
  closed — GATE-INFRA-92 `inputs` :41 names only `experiments/mlflow` +
  `experiments/fare_prediction`) and outside parity custody (GATE-INFRA-93 SHARED[]
  lists only those same two experiment siblings).
- **Single platform touchpoint**: `_common.py:12` imports
  `broadway.samples.read_named_sample`; every step consumes named sample
  `fare_prediction_1m` (registry owns seed/size/columns/filters per
  `configs/sample/fare_prediction_1m.yaml`). No other `src/broadway` import exists
  in the batch. Direct libs: pandas everywhere; statsmodels OLS family in 14
  scripts; sklearn in exactly four (`05`, `06` ColumnTransformer/Pipeline/
  OneHotEncoder; `08`, `11`, `22` train_test_split; `22` additionally
  RandomForestRegressor+metrics); scipy in `02`, `04`.
- **MLflow/optuna usage: ZERO** across all 25 files (grep-clean).
- Seeds/literals: literal `42` dominates (`np.random.default_rng(42)` in `03_1:83`,
  `09:81`, `10:97`, `11:95`; `random_state=42` in `08:36`, `11:47`, `16:133`,
  `22:86`; derived `42+i` in `07:83`; RF `random_state=42` in `22:91`). **Exactly
  one UNSEEDED draw in the whole batch**: `15_complexity_funnel.py:65`
  `np.random.choice(len(eval_df), …)` (matches DET-C1 "hardcoded literals incl.
  more_modeling batch" and DET-f "unseeded plot scripts" — packet D #20/#28).
- Entry shape: ALL 24 scripts end in `if __name__ == "__main__":` guards; NO
  argparse/sys.argv anywhere. Only `22_demand_forecasting.py` mutates its own
  runtime (`os.environ["DISABLE_PANDERA_IMPORT_WARNING"]="True"` :23 and
  `sys.path.insert(0, str(Path(__file__).parent))` :37 before importing `_common`)
  — every other script relies on CWD-equals-script-dir for the bare
  `from _common import …`.
- Outputs: `experiments/results/more_modeling/**` per house `.gitignore:19`
  convention; ~60 artifacts present (CSV/PNG/MD), including orphan
  `demand_forecast_model_metrics.md` (un-prefixed leftover of an earlier naming
  era beside `22_demand_forecast_model_metrics.md`).
- Archaeology: `__pycache__/03_2_heteroscedasticity_test.cpython-312-pytest-8.4.2.pyc`
  proves a pytest process imported `03_2` at least once, yet no test references the
  batch today — coverage was touched and lost.

### Per-script register

Format: purpose · entry · platform (imports / reads / writes) · seeds/literals ·
MLflow-optuna · gated-surface overlap.

- **`_common.py`** (helper, no main guard) — Shared paths/constants/loader.
  Platform: pandas + `broadway.samples.read_named_sample`; `RESULTS =
  experiments/results/more_modeling` (:15), `SAMPLE_NAME="fare_prediction_1m"`
  (:16), `TARGET="fare_amount"` (:17); `load_sample()` prints provenance incl.
  `artifact_sha256` (:20-26); `INDEPENDENT_NUMERIC_FEATURES=[trip_distance,
  trip_duration_minutes]` (:37-40, excludes speed_mph as deterministic ratio and
  zone ids as categorical). Seeds: none (delegates to sample registry). Overlap:
  IS the sanctioned read path — mirrors `experiments/mlflow/_common.py:51`
  convention (cited at GATE-SURFACES row 07-surfaces:135).
- **`01_feature_mean_ci.py`** — Normal-approx confidence interval for the mean of
  each numeric feature. Entry: main guard. Reads named sample; writes
  `01_feature_mean_ci.csv`. Seeds: none needed (deterministic). MLflow/optuna:
  none. Overlap: none — descriptive stats outside `stats/describe.py` scope but
  non-rival.
- **`02_feature_target_correlation_ci.py`** — Pearson feature↔`fare_amount`
  correlation CIs (scipy). Entry: main guard. Writes `02_…ci.csv/.png/.md`.
  Seeds: none. Overlap: mild thematic overlap with `stats/regression.py`
  correlations; no machinery reuse conflict.
- **`03_feature_effect_ci.py`** — OLS coefficient effect CIs on
  INDEPENDENT_NUMERIC_FEATURES (statsmodels). Writes `03_feature_effect_ci.csv/
  .png/.md`. Seeds: none. Overlap: `src/broadway/stats/regression.py` +
  `stats/assumptions.py` cover the same question inside governed lanes (band
  05-stats) — adjudicate before ratification.
- **`03_1_raw_ols_and_residuals.py`** — Raw OLS fit + residual diagnostic plots.
  Writes `03_1_raw_ols_coefs.csv`, `03_1_residual_diagnostics.png`,
  `03_1_raw_ols_summary.md`. Seeds: `default_rng(42)` :83 (residual resampling).
  Overlap: same as 03 (diagnostics ≈ `stats/diagnostics.py`).
- **`03_2_heteroscedasticity_test.py`** — Breusch-Pagan-family heteroscedasticity
  tests (`statsmodels.stats.api`). Writes `03_2_heteroscedasticity_test.csv`.
  Seeds: none. Overlap: direct sibling of `stats/assumptions.py`; the pytest-pyc
  fossil suggests it once fed a test.
- **`04_categorical_anova_eta2.py`** — One-way ANOVA + η² of fare across zone
  groups (scipy). Writes `04_categorical_anova_eta2.csv/.png/.md`. Seeds: none.
  Overlap: `src/broadway/stats/anova.py` + `stats/effect_size.py` are the gated
  equivalents (band 05-stats) — clearest stats-module duplication in the batch.
- **`05_joint_model_top10_zones.py`** — Joint OLS with one-hot top-10 pickup
  zones (sklearn ColumnTransformer/Pipeline + statsmodels); zone-premium
  extraction. Writes `05_joint_model_coefs.csv`, `05_zone_premiums.png`,
  `05_joint_model_summary.md`. Seeds: none. Overlap: dummy-coded regression lives
  in `stats/regression.py`; premium tables rival `experiments/fare_prediction`
  estimation outputs.
- **`06_joint_model_time_of_day.py`** — Same joint design against time-of-day
  buckets; overnight/rush premium coefficients. Writes
  `06_time_of_day_coefs.csv`, `06_time_of_day_premiums.png`. Seeds: none.
  Overlap: as 05; temporal surcharge facts feed 12's narrative.
- **`07_airport_flat_rate_audit.py`** — Airport flat-rate policy audit (JFK/LGA/
  EWR flat trips vs meter prediction). Writes `07_airport_flat_rate_audit.csv`,
  `07_airport_residuals.png`. Seeds: derived literal `default_rng(42 + i)` :83.
  Overlap: ratecode logic adjacent to cleaning/lookup surfaces; facts reused by
  16's flat-rate overrides.
- **`08_model_validation.py`** — Hold-out validation: coefficient stability
  train-vs-full + fare-bucket residual audit (`FARE_BINS` literal :24-25).
  Writes `08_model_validation_summary.csv`, 2 PNG, `.md`. Seeds:
  `train_test_split(random_state=42)` :36. Overlap: conceptual rival of
  `evaluate/validation.py` (band 30–39) though no MLflow involvement.
- **`09_total_amount_fees.py`** — Meter-vs-total decomposition (fees/tolls/tips
  "funnel of risk"). Writes `09_total_amount_fees.csv`, 2 PNG, `.md`. Seeds:
  `default_rng(42)` :81. Overlap: none direct; feeds 12/16 narratives.
- **`10_subpopulation_audit.py`** — Residual KDE/scatter subpopulation audit.
  Writes `10_subpopulation_audit.csv`, 2 PNG, `.md`. Seeds: `default_rng(42)`
  :97. Overlap: `trust/drift.py`-adjacent thematically; no machinery clash.
- **`11_pure_meter_audit.py`** — Pure-meter subset OLS + bucket stats.
  Writes `11_pure_meter_validation.csv`, 2 PNG, `.md`. Seeds:
  `random_state=42` :47 + `default_rng(42)` :95. Overlap: as 03/08.
- **`12_executive_summary.py`** — ⚠ writes a FULLY HARDCODED narrative essay:
  `SUMMARY_OUT.write_text(md)` :78 where `md` is a literal string :13-76 quoting
  frozen numbers ("93% of variance", "$3.25/mile", "$70 JFK") — READS NOTHING
  (imports only `RESULTS`). Entry: main guard. Seeds: none. Overlap: bypasses
  `reports/markdown.py` registry convention entirely. Drift hazard: prose can
  silently diverge from sibling CSVs forever.
- **`13_spatiotemporal_demand.py`** — Weekly demand pattern + zone×hour heatmap
  matrix. Writes `13_…matrix.csv` + 2 PNG. Seeds: none. Overlap: none in gated
  planes (viz-only; `viz.py` is styling, not content).
- **`14_demand_revenue_signatures.py`** — Top revenue cells signature scatter +
  table. Writes `14_top_revenue_cells.csv`, PNG, `.md`. Seeds: none. Overlap: none.
- **`15_complexity_funnel.py`** — Model-class complexity ladder on a subsample.
  Writes `15_complexity_funnel.csv/.png/.md`. Seeds: **UNSEEDED**
  `np.random.choice(...)` :65 — the batch's only nondeterminism.
  Overlap: selection/learning-curve themes (`selection/learning_curves.py`).
- **`16_production_pricing_engine.py`** — ⚠ HARDCODED RIVAL PRICING ENGINE as
  literal constants :21-35 (`CORE_INTERCEPT=4.13`, `$3.03/mile`, `$0.29/min`,
  `MINIMUM_FARE=5.00`, `AIRPORT_ZONES={132,138,237}`, `MANHATTAN_ZONES=
  set(range(1,101))`, `FLAT_RATE_JFK=70/LGA=52/EWR=70`, risk premium `$1/mile
  over 15mi`) scored against actual fares via `production_pricing_engine()` :51-71;
  eval sample `random_state=42` :133. Writes `16_production_engine_comparison.csv/
  .png/.md`. Overlap: THE duplication-hazard row — competes with governed pricing
  surfaces that are still aspirational: `inference/api.py` docstring stub
  (DEP-F1a), `baseline/prediction.py`, `experiments/fare_prediction` estimations,
  config-law pricing (`configs/experiment/*.yaml`).
- **`17_spatial_velocity_mapping.py`** — Per-zone speed impact on fare (OLS).
  Writes `17_velocity_impact.csv/.png/.md`. Seeds: none. Overlap: features/speed
  handling touches the `speed_mph` exclusion rule `_common.py:34-35` documents.
- **`18_fleet_revenue_simulator.py`** — Hourly rides/revenue simulation baseline
  vs scenario (pure pandas/numpy). Writes `18_fleet_revenue_simulator.csv/.png/
  .md`. Seeds: none. Overlap: none (scenario math exists in no gated plane).
- **`19_surge_sensitivity.py`** — Surge multiplier sensitivity sweep. Writes
  `19_surge_sensitivity.csv/.png/.md`. Seeds: none. Overlap: none.
- **`20_heatmap_surge_matrix.py`** — Surge matrix pivot heatmap. Writes
  `20_surge_matrix.csv/.png/.md`. Seeds: none. Overlap: none.
- **`21_surge_waterfall.py`** — Surge revenue waterfall decomposition. Writes
  `21_surge_waterfall.csv/.png/.md`. Seeds: none. Overlap: none.
- **`22_demand_forecasting.py`** — Zone×hour ride-VOLUME forecasting with a bare
  `RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=10,
  min_samples_leaf=5, random_state=42, n_jobs=-1)` :91 + custom MAPE :41-48;
  split `random_state=42` :86. ONLY script with runtime mutation preamble
  (:23 os.environ, :37 sys.path.insert). Writes `22_demand_volume_by_zone_hour.csv`,
  `22_demand_forecast_actual_vs_predicted.png`,
  `22_demand_forecast_error_by_zone.png`, `22_demand_forecast_model_metrics.md`.
  Overlap: duplicates the GOVERNED RandomForest surface
  (`training/models/random_forest.py`, band 30–39) minus every governed property:
  no config, no MLflow logging, no champion comparison, no lineage record.

### RATIFICATION-GATE-CHECKLIST (proposed — minimal gates before the batch could
leave foreign context; NO code changes proposed now, standing ruling keeps it untracked)

- **RGC-1 custody flip precedes everything** — promotion begins with a parity
  SHARED[] amendment + ledger row (GATE-INFRA-93 `check()` list :42-67 governs
  which experiment dirs are tracked); until amended, directory stays foreign by
  construction. Gate = parity checker passes WITH the new entry.
- **RGC-2 seed-policy conformance** — every RNG draw YAML-sourced or a declared,
  tested constant; kills the literal-42 class AND the one unseeded draw
  (`15:65`). Merge point: board lane `EXPERIMENTS-SEED-POLICY` (packet D #28);
  index-pinning test pattern per `DET-A-SUBSAMPLE-PINS` (#23).
- **RGC-3 lint/type tier inclusion** — ruff/mypy input lists (GATE-INFRA-92 :41)
  currently close the door; promotion requires adding the directory to a tier or
  recording an explicit exclusion rationale in the gate script SSOT (D16a/D26 law).
- **RGC-4 import hygiene** — uniform main-guard exists ✔; gate = importable
  WITHOUT CWD dependence (generalize or ban the `sys.path.insert` hack of `22:37`)
  + Agg-backend assertion so no tier ever opens a display.
- **RGC-5 output determinism contract** — one writer convention for
  `experiments/results/more_modeling/**` (filenames, dtypes, index=False) and
  `EXECUTIVE_SUMMARY.md` either GENERATED from sibling CSVs or relabeled
  editorial-opinion (hardcoded stats drift silently today, `12:13-76`).
- **RGC-6 tracking-boundary declaration** — batch is MLflow/optuna-free (verified);
  ratified lane must DECLARE that boundary (assert-no-tracking-import probe) or
  adopt `setup_mlflow` — no silent third state.
- **RGC-7 duplication adjudication** — before ratification: (a) `22`'s bare RF vs
  governed `training/models/random_forest.py`; (b) `16`'s hardcoded pricing
  constants vs config law + the F-1 API contract; (c) `03*/04/05/06` vs
  `stats/{regression,assumptions,anova,effect_size}.py`. Each pair resolves to
  route-through-governed-code OR documented non-production status.
- **RGC-8 test minimum** — ≥1 real pytest node per promoted script (tiny-fixture
  smoke); today coverage is ZERO (the `__pycache__` pytest-pyc of `03_2` is a
  fossil, not coverage).

---

## PART 2 — Territory 2: top-level `k8s/` non-optuna siblings

The DEPLOY-DIFF lane found the mismatches (F-1 stub, F-2 env, M1–M9) but emitted
NO GATE rows for these surfaces as such. Below: **PROPOSE-GH4-NN candidates** in
house format — NOT ratified `GATE-*` ids; `validated_by: []` throughout; every
FINDING carries `root:`; the kubeconform-scope hole is CITED once per row from
GATE-INFRA-99 (`ci.yml:80` scans `k8s/optuna/` ONLY, minus kind-config ignore) and
the deploy-diff CI verdict ("four top-level manifests structurally unvalidated",
factsheet :43-48) — not re-derived.

- id: PROPOSE-GH4-01
  phase: ops-deploy-proposed
  owner: k8s/api-deployment.yaml:2 Deployment broadway-api (image :17, uvicorn cmd :18-24) + :28 Service broadway-api (80→8000 :35-37)
  inputs: ["image broadway:latest — root Dockerfile build product, never built by CI (DEP-M9)", "uvicorn target literal `inference.api:app` :20", "no env/volume/probe/resource keys"]
  outputs: ["2 replica pods serving :8000", "cluster-internal Service on :80"]
  transforms: ["PYTHONPATH=/app/src inside image exposes package `broadway` only → bare target cannot resolve (Dockerfile:17)", "api needs MLflow registry per stub docstring; manifest sets no tracking URI, mounts nothing"]
  touched_by: ["DEPLOY-F1-API-LANE (packet D #8+#9 — primary)", "CI-BUILD-BROADWAY-LATEST (#19) for the image half"]
  validated_by: []
  if_changed: [k8s/api-deployment.yaml, Dockerfile, src/broadway/inference/api.py]
  # FINDING: manifest written against an ASPIRATIONAL contract nothing validates — stub has no app object, uvicorn target wrong for exposed package, tracking URI missing (deploy-diff F-1 verdict, both breaks).
  # root: no render-and-validate stage exists for THIS directory — kubeconform scope hardcoded to k8s/optuna/ (ci.yml:80; GATE-INFRA-99 FINDING) so syntax AND semantics both invisible; a sibling-directory scan would still not catch the wrong uvicorn target — only a boot probe would.
- id: PROPOSE-GH4-02
  phase: ops-deploy-proposed
  owner: k8s/api-deployment.yaml:39 HorizontalPodAutoscaler broadway-api (min2/max10 :48-49, cpu avgUtilization 70 :56)
  inputs: ["scaleTargetRef → same-file Deployment :44-47", "cpu utilization metric :50-56"]
  outputs: ["replica count 2..10 driven (or not) by CPU"]
  transforms: ["threshold 70 hardcoded; configs/environment/production.yaml:16 api_hpa_cpu_threshold: 80 never wired (DEP-M4)"]
  touched_by: ["senior D-verdict killed DEP-M4 wiring ('correct end-state is deletion of one side') — candidate records BOTH halves pending that ruling"]
  validated_by: []
  if_changed: [k8s/api-deployment.yaml, configs/environment/production.yaml]
  # FINDING: threshold exists twice (manifest literal 70, config key 80) with no wiring owner; config key independently ruled dead (C#25).
  # root: dual-source constant with no SSOT AND no consumer — plus container declares NO resources.requests (whole file), so CPU-utilization HPA has no denominator and cannot compute utilization even if thresholds agreed.
- id: PROPOSE-GH4-03
  phase: ops-deploy-proposed
  owner: k8s/train-job.yaml:2 Job train-{{ .Values.runId }} (cmd uv run ds-pipeline train :11-21, restartPolicy Never :22)
  inputs: ["image broadway:latest :10 (same unbuilt image, DEP-M9)", "Helm values .Values.runId/.dataset/.experiment :4,:17,:19", "--environment production :21"]
  outputs: ["one-shot training pod; artifacts written cwd-relative (discover/module.py:22-23) — NO PVC mounted"]
  transforms: ["kind is Job, NOT CronJob — no schedule exists anywhere despite 'CronJob' folklore; Go-template braces make the file invalid static YAML until rendered"]
  touched_by: ["DEPLOY-F2-TRAINJOB-LANE (packet D #10+#18 — primary; resolver root owned by C#21)", "render precedent to copy: optuna orchestrator dry-run render+kubeconform (ci.yml:85-104)"]
  validated_by: []
  if_changed: [k8s/train-job.yaml, configs/environment/production.yaml, src/broadway/config/resolver.py]
  # FINDING: runs --environment production against 7 ${VARS}; unset vars pass through literally → database_port int coercion crashes at config load (config/schema.py:73); local gates blind via concrete dev defaults.
  # root: deployment assumes an AMBIENT env contract (env block/secretRef/PVC) that no manifest or lifecycle step fulfills — and the file cannot even enter kubeconform as-is because template syntax precedes validation (cite scope hole, ci.yml:80 / GATE-INFRA-99).
- id: PROPOSE-GH4-04
  phase: ops-deploy-proposed
  owner: k8s/postgres-deployment.yaml:2 StatefulSet postgres (image :18, env :21-36, vct pg-data 10Gi :40-47) + :49 Service postgres (5432)
  inputs: ["image broadway-postgres:latest :18 — no repo build produces it (DEP-M6)", "ConfigMap `environment` keys database_user/database_password/database_name via configMapKeyRef :24-36", "PVC template 10Gi"]
  outputs: ["single-postgres cluster state with persistent volume; Service :5432"]
  transforms: ["credential material sourced from configMapKeyRef, not secretKeyRef (:27-31)", "contrast: optuna stack mounts a proper secret volume (k8s/optuna/postgres.yaml:50,:55) — in-house correct pattern exists"]
  touched_by: ["K8S-CONFIGMAP-ENV (packet D #11 — primary)", "IMAGE-TAG-COHERENCE (#15/#16) for the tag half", "context-only citations: SEC-S6 dev literals, SEC-S1 upstream rotation (HUMAN-CALL, other lane's custody)"]
  validated_by: []
  if_changed: [k8s/postgres-deployment.yaml, k8s/optuna/postgres.yaml, configs/environment/*.yaml]
  # FINDING: POSTGRES_PASSWORD flows through a plain ConfigMap — secret-tier enforcement exists for the optuna sibling stack but not here.
  # root: no policy or schema distinguishes secret-class keys in TOP-LEVEL manifests; the optuna stack's discipline is convention inside one directory, not a rule any gate applies to siblings (scope hole again: ci.yml:80, GATE-INFRA-99).
- id: PROPOSE-GH4-05
  phase: ops-deploy-proposed
  owner: ABSENT-OBJECT — no manifest in the repo declares ConfigMap `environment`; consumption sites k8s/postgres-deployment.yaml:24,:30,:35 (recorded owner-of-record = the consuming file:symbol triple)
  inputs: ["expected keys database_user, database_password, database_name", "repo-wide ConfigMap roster is exactly {optuna-config (k8s/optuna/configmap.yaml:11), postgres-init (k8s/optuna/postgres.yaml:9)} — neither matches"]
  outputs: ["none satisfiable — pod creation fails CreateContainerConfigError (DEP-M1 verdict)"]
  transforms: ["k8s/optuna/lifecycle.sh:104 applies only optuna-dir files — no step generates the top-level ConfigMap from configs/environment/*.yaml either"]
  touched_by: ["K8S-CONFIGMAP-ENV (packet D #11) + its kubeconform-scope rider + .dockerignore rider"]
  validated_by: []
  if_changed: [k8s/postgres-deployment.yaml, any future ConfigMap manifest]
  # FINDING: three required-key references resolve to an object declared NOWHERE (manifests, lifecycle scripts, Helm values — checked repo-wide).
  # root: consumers written against an environment-injection convention that was never given a declaring artifact; widening kubeconform scope WOULD catch this class (dangling configMapKeyRef) — making the ci.yml:80 scope hole the single cheapest structural fix.
- id: PROPOSE-GH4-06
  phase: ops-deploy-proposed
  owner: k8s/mlflow-deployment.yaml:2 Deployment mlflow (image :17, cmd mlflow server :20-26) + :27 Service mlflow (5000)
  inputs: ["image broadway-mlflow:latest — CI tags mlflow-server:<sha>, so no build produces this tag (DEP-M5)", "no env, no backend-store-uri, no PVC"]
  outputs: ["stateless-looking MLflow UI/API on :5000 with ephemeral in-container storage"]
  transforms: ["second MLflow-server surface coexists with k8s/optuna/mlflow.yaml stack — two truths with different tag vocabularies (<sha> vs :latest) and different persistence stories"]
  touched_by: ["IMAGE-TAG-COHERENCE (packet D #15 — primary)"]
  validated_by: []
  if_changed: [k8s/mlflow-deployment.yaml, k8s/optuna/mlflow.yaml, .github/workflows/ci.yml CD job]
  # FINDING: tracking-server state evaporates on pod reschedule (sqlite/file store inside container filesystem, nothing mounted).
  # root: top-level manifest sketches a SECOND mlflow deployment instead of referencing the governed optuna-stack one — duplication born where no gate compares sibling directories (kubeconform scope, ci.yml:80 / GATE-INFRA-99, cited).
- id: PROPOSE-GH4-07
  phase: ops-deploy-proposed
  owner: Dockerfile:1 multi-stage build (uv sync --frozen :7, PYTHONPATH=/app/src :17, CMD ds-pipeline :19) — the `broadway:latest` source for BOTH api and train-job manifests
  inputs: ["pyproject.toml + uv.lock :6", "COPY src/ scripts/ configs/ :12-14 (experiments/, project/, data/ deliberately absent)"]
  outputs: ["local image broadway:latest — consumed by k8s/api-deployment.yaml:17 and k8s/train-job.yaml:10"]
  transforms: ["PYTHONPATH=/app/src is precisely why bare `inference.api:app` cannot resolve (feeds DEP-F1a)", "uv:latest floating pin :3"]
  touched_by: ["CI-BUILD-BROADWAY-LATEST (packet D #19 — primary; DEP-M9 'never built by CI')", "build-and-boot builds OPTUNA images only (ci.yml:163-187)"]
  validated_by: []
  if_changed: [Dockerfile, pyproject.toml, uv.lock, src/, k8s/api-deployment.yaml, k8s/train-job.yaml]
  # FINDING: image sits at the center of TWO deployment manifests yet ZERO CI jobs or local tiers build it — the entire top-level k8s story is unverifiable by construction.
  # root: build-and-boot lane (ci.yml:132,:163-187) scoped to optuna images only, mirroring the same sibling-blindness as the kubeconform scope (:80); no gate binds 'image referenced by a manifest' to 'image built by some job'.
  # GH-1-OVERLAP: contract named `docker/base + docker/postgres` Dockerfiles — NEITHER EXISTS (docker/base/ absent; docker/postgres/ holds init.sql only). This ROOT Dockerfile is the build surface GH-4 can detail best (consumed by both manifests above), so it is CLAIMED HERE; GH-1 plausibly wants it for compose-context reasons — see Part 3 notes.
- id: PROPOSE-GH4-08
  phase: ops-deploy-proposed
  owner: docker-compose.yml:16 build context ./docker/postgres (holds ONLY init.sql — no Dockerfile) + docker/mlflow/Dockerfile:1 (7-line) + docker/postgres/init.sql:1
  inputs: ["compose service build paths", "init.sql seed script (uncited by any k8s top-level manifest — optuna stack uses its own postgres-init ConfigMap instead)"]
  outputs: ["compose-local images; BROKEN for the postgres service (DEP-M7: build context contains no Dockerfile)"]
  transforms: ["docker/mlflow/Dockerfile is a THIRD mlflow image recipe beside root-Dockerfile-consumer tag and optuna/Dockerfile.mlflow — tag vocabulary divergence again"]
  touched_by: ["IMAGE-TAG-COHERENCE (packet D #17 — primary)", "GH-1 (probable primary owner for compose/build-context gates — overlap declared, see Part 3)"]
  validated_by: []
  if_changed: [docker-compose.yml, docker/postgres/, docker/mlflow/]
  # FINDING: compose builds ./docker/postgres — a directory that never contained a Dockerfile in this tree; `docker/base/` likewise does not exist although the contract's mental model expects it.
  # root: build-context layout was planned as Dockerfile homes and never materialized; nothing validates that a compose build path contains a Dockerfile before ship (and GH-4 declines to deep-detail these, deferring detail-depth to GH-1 to avoid silent duplication).

### Merge map — PROPOSE-GH4-NN ↔ packet-D register ↔ board lanes (numbering merges later)

| PROPOSE | surface | packet-D rows | board lane (BY NAME) |
|---|---|---|---|
| 01 | api Deployment+Service | DEP-F1a, DEP-F1b, (DEP-M3 rides here) | DEPLOY-F1-API-LANE |
| 02 | HPA | DEP-M4 (senior kill noted) | fold into DEPLOY-F1-API-LANE cleanup |
| 03 | train-job Job | DEP-F2, DEP-M8 | DEPLOY-F2-TRAINJOB-LANE (resolver root C#21) |
| 04 | postgres StatefulSet+Service | DEP-M6 (+secret-tier aspect) | K8S-CONFIGMAP-ENV + IMAGE-TAG-COHERENCE |
| 05 | ConfigMap `environment` (absent) | DEP-M1 | K8S-CONFIGMAP-ENV (+kubeconform-scope rider, .dockerignore rider) |
| 06 | mlflow Deployment+Service | DEP-M5 | IMAGE-TAG-COHERENCE |
| 07 | root Dockerfile build gate | DEP-M9 | CI-BUILD-BROADWAY-LATEST |
| 08 | compose + docker/ contexts | DEP-M7 | IMAGE-TAG-COHERENCE |

## IDS DECLARED

PROPOSE-GH4-01 … PROPOSE-GH4-08 (8 candidates, phase ops-deploy-proposed);
RATIFICATION-GATE-CHECKLIST RGC-1 … RGC-8; script register 25 entries
(24 runnable + `_common.py`).

---

## PART 3 — Docker sweep & overlap declarations

- `docker/base/` DOES NOT EXIST; `docker/postgres/` contains ONLY `init.sql`
  (no Dockerfile). The contract's presumed sweep targets are therefore VACUOUS
  at their named paths — recorded as evidence inside PROPOSE-GH4-07/-08 FINDINGs
  rather than invented around.
- **Claimed by GH-4 (best detail)**: root `Dockerfile` (PROPOSE-GH4-07) — because
  both top-level manifests consume its tag and its `PYTHONPATH=/app/src` fact is
  load-bearing for the F-1a wrong-target analysis.
- **Flagged for GH-1 (overlap declared, not duplicated)**: `docker-compose.yml`
  build contexts, `docker/mlflow/Dockerfile`, `docker/postgres/init.sql`
  (PROPOSE-GH4-08 carries only the M7 broken-context fact that the postgres
  manifest row needs). If GH-1 also emits a root-Dockerfile row, keep GH-4's as
  the deployment-consumer view and GH-1's as the build-mechanics view — do not
  merge silently.
- Other overlaps declared inline: mlflow-server ×2 and postgres ×2 (top-level vs
  optuna stack) inside PROPOSE-GH4-06/-04; batch-vs-stats-module duplication
  inside RGC-7; packet-D lane names used verbatim in the merge map above.

## Most likely to surprise a future main-day reconciliation

`16_production_pricing_engine.py`: a COMPLETE, fully-specified production pricing
formula — intercept, per-mile/per-minute rates, minimum fare, JFK/LGA/EWR flat
rates, Manhattan zone enumeration, long-haul risk premium — sits as literal
Python constants in FOREIGN territory, while the GOVERNED pricing surface
(`inference/api.py`) is still a one-line docstring stub (DEP-F1a). The real spec
lives outside the law and the lawful home is empty — the exact configuration in
which someone quietly ships the foreign engine. (Runner-up: `12_executive_summary.py`
freezes its statistics into hand-written prose, so the batch's public-facing
conclusions can drift from its own CSVs with nothing ever noticing.)
