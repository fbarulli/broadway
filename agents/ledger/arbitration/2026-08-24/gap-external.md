# GAP-HUNTER GH-3 — EXTERNAL RESOURCES & ENVIRONMENT (2026-08-24)

- Contract: GH-3 · repo `/home/opc/ONE/broad-way` · branch `sklearn` · HEAD stamp `5016e93`
  · read-only except this file · zero git ops performed.
- Scope: everything the code touches OUTSIDE its own source tree with no registry
  presence today — env vars, filesystem contracts, network endpoints, wall-clock/RNG.
- Dedupe discipline: items already owned by arbitration rows are REFERENCED, never
  re-proposed. Known-covered honored: `BROADWAY_CONFIGS_DIR` repoint (C#20→S14),
  `${VAR}` unset-passthrough / `database_*` interpolation probe (C#21+#22→S1,
  deploy consequences D#10→DEPLOY-F2), `.env` drift (D#5→SLATE A), lookup-symlink
  documentation (D#7→SLATE B), dev literal creds (D#6), wall-clock freeze design
  (DET-d→REPRO-TIME-FREEZE), RNG families (DET-C1/C2/C3/a/b/c/f), numpy dual pin
  (DET-C2→ENV-NUMPY-SINGLE-PIN).

## 1 · ENV VAR CENSUS

Python-plane readers/writers (14 distinct names):

| # | var | reader(s) | breaks when unset/wrong | documented | gated |
|---|-----|-----------|--------------------------|------------|-------|
| 1 | `BROADWAY_CONFIGS_DIR` | config/loader.py:34 (`or "configs"`) | foreign tree loaded silently | ledger only | **COVERED** C-S14 (warn) — reference only |
| 2 | `BROADWAY_MLFLOW_CONFIG` | experiments/mlflow/_common.py:60 (**import-time eager default**); experiments/mlflow/03_optuna_worker.py:46-47 (lazy twin); produced by k8s/optuna/render_worker_jobs.py:48 | wrong path → FileNotFoundError at import, no var named in error | code comments + k8s generator; NOT in README/.env.example | no — two divergent resolution styles (see GH3-07) |
| 3 | `DATA_MODE` | project/data.py:77 via `_resolve_mode`, frozen at import (:102 `MODE = _resolve_mode()`) | invalid → ValueError naming var (good); BUT value frozen for whole process at import; keys cache filenames `joined_sample_{mode}.parquet`/`sample_meta_{mode}.json` | README §3 (:304-317) | partially — mode param can override call sites, import constant cannot |
| 4 | `CI` | src/broadway/etl/module.py:74 (`== "true"` gates `ci_sample_size` reduction); project/etl/process.py:44 legacy twin `sample_for_ci` | unset locally = full data (fine); SET-anywhere = silent population change; legacy twin signals by one `logger.info` line only, platform twin records reason string in audit trail | contract declared gates.yaml:126-127; README:109 says "CI-gated"; mechanics undocumented as env surface | **NO local-provenance teeth** — GH3-04 |
| 5 | `BROADWAY_ARTIFACTS_DIR` | onboard/module.py:33; discover/module.py:23; reports/audit.py:672-673 (duplicated inline re-reads) | cwd-relative default `"artifacts"` scatters writes when CLI run off-root | tests + ledger only | no — GH3-06 |
| 6 | `BROADWAY_LINEAGE_DIR` | lineage/records.py:8 (default `artifacts/lineage`) | same cwd-scatter class | ledger (gates.yaml:1939) | no — GH3-06 |
| 7 | `BROADWAY_TIMELINE_DIR` | timeline/module.py:8 (default `artifacts/timeline`) | same class | gates.yaml:1773 | no — GH3-06 |
| 8 | `BROADWAY_REPORTS_DIR` | reports/paths.py:6 (default `"reports"`) | same class | tests only | no — GH3-06 |
| 9 | `BROADWAY_DATASET_DIR` | discover/module.py:22 (plain str default `"dataset"`) | same class | tests only | no — GH3-06 |
| 10 | `BROADWAY_IDENTIFIER_THRESHOLD` | onboard/infer.py:10 (**float() at import**); discover/module.py:24 (**float() at import**); reports/audit.py:30 (call-time float()) | garbage → bare `ValueError: could not convert string to float: …` AT IMPORT, var unnamed; three duplicated defaults free to drift | nowhere user-facing | no — GH3-03 |
| 11 | `BROADWAY_QQ_MIN_UNIQUE` | discover/qq.py:111 (int() at call) | garbage → int() ValueError, var unnamed | test_qq.py:589 pins override | tested but same parse-teeth class — folded into GH3-03 |
| 12 | `MLFLOW_ALLOW_FILE_STORE` | WRITTEN by our code training/mlflow_utils.py:56,203; read by mlflow lib | n/a (we produce it) | docstring/comment context | deliberate shim — NOT-GATE |
| 13 | `MLFLOW_TRACKING_URI` | ambient third-party consumer: optuna-integration tracker per hpo.py:74 comment "(caller-set)"; plus staging/prod yaml interpolation | ambient-unset during HPO → mlflow lib defaults/file-store silently | .env.example (example-only key) | interpolation half COVERED S1/D-F2; ambient-HPO half rides DEP-F2 env block — reference |
| 14 | `DISABLE_PANDERA_IMPORT_WARNING` | WRITTEN at import by WIP experiment experiments/more_modeling/22_demand_forecasting.py:23; read by pandera | n/a | none | NOT-GATE (untracked WIP, lib-facing suppression) |

Shell/deployment-plane (11 names): `DB_USER`/`DB_PASSWORD`/`DB_NAME` (k8s/optuna/lifecycle.sh:61-78, password auto-generated via openssl fallback), `OPTUNA_BACKUP_DIR` (:19), `OPTUNA_WORKER_IMAGE` (:137), `VIEW_PG_NAME`/`MLFLOW_VIEW_PORT`/`VIEW_PG_PORT`/`VIEW_PG_USER`/`VIEW_PG_PASSWORD` (:177-181), `PGPASSWORD` (:124). All self-defaulted inside lifecycle.sh; manifest-side env validation belongs to K8S-CONFIGMAP-ENV / DEP-F2 lanes — NOT-GATES here. Compose adds `MLFLOW_PORT`/`DATABASE_PORT` defaults (docker-compose.yml:5,18) — deployment-plane rot owned by IMAGE-TAG-COHERENCE/#17 row. `TMPDIR` consumed by scripts/run_local_ci.sh:18. Deleted dialect noted: `PARITY_MAIN_DAY` (tests/test_branch_parity_scripts.py:8 asserts its absence).

Documentation verdict: `.env.example` covers exactly the 6 infra keys; README documents `DATA_MODE` + MLflow start commands; the entire `BROADWAY_*` family (9 of 14 Python names) is documented ONLY in ledger/tests — no single env-var table exists anywhere.

## 2 · FILESYSTEM CONTRACTS

Surveyed contracts (7 areas): data/raw contents · configs roots · artifacts/ tree · MLflow local stores · /tmp usage · tool caches · absolute-path assumptions.

1. **data/raw is out-of-band on TWO levels.** `taxi_zone_lookup.csv` is a symlink to `/home/opc/ONE/learning/data/raw/taxi_zone_lookup.csv` (verified live today; confirmed by factsheet 2026-08-24-secret-audit:33 and D#7 ADOPT). Additionally ALL three `yellow_tripdata_2024-{01,02,03}.parquet` (~160MB) are gitignored real files fetched out-of-band; fresh clones get neither. Parquet absence fails LOUD (`project/etl/process.py:31-32` FileNotFoundError naming the dir; README documents TLC fetch); the lookup symlink fails equally loud but names only a relative path, and its fix slate (README prose) documents rather than GATES. → **GH3-01** proposes the existence-check tooth (distinct mechanism from D-Slate-B's doc edit).
2. **Dual configs-root convention (split-brain risk).** src-plane resolves configs through loader.py CONFIGS_DIR (env-overridable); project-plane hardcodes CWD-relative `Path("configs/…")` SEVEN times at module import (project/data.py:31-47: dataset/analysis/stats/train/project/features yamls) bypassing that gate entirely. Two consequences: (a) importing project.data from any non-root CWD dies with a raw FileNotFoundError listing a relative path; (b) if BROADWAY_CONFIGS_DIR IS set (the exact scenario C-S14 warns about), src validates the redirected tree while project-plane keeps reading the original repo tree — a silent split-brain where the two planes disagree about which config universe they are in. Also cwd-relative: RESULTS_DIR = `data/processed` (project/data.py:126). → **GH3-02**.
3. **artifacts/ layout convention is enforced only by scattered literals.** Writers: discover/{profile.json,qq_overview.json}, lineage/records/<sanitized>.json, timeline/<analysis>/steps/<step_id>.json, training/training_result.json. Readers rebuild sibling paths independently — audit.py:672-673 re-derives ARTIFACTS_DIR from env twice inline instead of sharing the writer's constant; root overrides mean writer and reader CAN resolve different trees in one process if env changes between phases or modules disagree. Layout itself is pinned by tmp_path tests; the ROOT SHARING is not. → **GH3-06**.
4. **MLflow local store has two divergent documented locations.** README:15 uses `sqlite:///$(pwd)/.mlflow.db` + `file://$(pwd)/mlruns`; docker-compose.yml:12 uses `sqlite:///mlflow/mlflow.db`; development.yaml points at `http://localhost:5000`. Stale-registry skew between identical runs is acknowledged with a manual reset command (README:16) but nothing detects it. Folded into GH3-05's finding surface (connection/store hygiene).
5. **/tmp**: k8s entrypoint-mlflow.sh:19-20 `/tmp/backend_uri`+`/tmp/allowed_hosts` handshake is container-local (fine); run_local_ci.sh:18 puts MPLCONFIGDIR at `${TMPDIR:-/tmp}/broadway-mpl` — predictable shared path, stale-font-cache class only.
6. **Tool caches**: `.uv-cache`/`.mplconfig` are script-defaulted (scripts/ship.sh:10, run_local_ci.sh:18, G0B.md recipe) and documented (TODO.md:66-67) — NOT-GATE.
7. **Absolute-path assumptions**: symlink target (GH3-01); `_common.py:54` depth-fallback REPO heuristic (documented, image-layout-coupled); all other ROOTs are `Path(__file__)`-anchored (render_worker_jobs.py:23 etc.) — fine.

## 3 · NETWORK / ENDPOINT SURFACES

Four surfaces; src/*.py makes ZERO direct HTTP calls — every outbound connection is library-mediated:

1. **MLflow tracking HTTP :5000** — dev literal `http://localhost:5000` (development.yaml:6), compose `${MLFLOW_PORT:-5000}`, k8s `http://mlflow:5000` (configmap.yaml:27), e2e harness spins ephemeral server + curl health loop (check_e2e_determinism.sh:148,168-175). Teeth exist but are NARROW: setup_mlflow converts failure to a helpful RuntimeError ONLY when the wrapped message contains refusal markers `(Connection refused | Failed to establish a new connection | Max retries exceeded)` (mlflow_utils.py:47-51,66-72 — comment admits it is text-matching because MLflow swallows `__cause__`). Timeout/DNS/reset classes (`Read timed out`, `Name or service not known`, `Remote end closed…`) miss the markers → raw MlflowException, hint lost. And the guard covers SETUP ONLY: every later log_* call during training re-hits the network unguarded. Positive precedent worth cloning: worker preflight() probes optuna+mlflow reachability loudly before starting work (03_optuna_worker.py:118-138, prints `[worker] FATAL: cannot reach MLflow`). → **GH3-05**.
2. **Postgres / optuna storage** — DSNs assembled by compose_db_url (tested test_optuna_extended.py:80-81); k8s `postgresql+psycopg2://postgres:5432/{optuna,mlflow}` (configmap.yaml:24-25); compose exposes :5432 w/ DATABASE_* defaults; lifecycle view-server restores dumps into a local postgres (:177-209). Local dev carries literal localhost creds — ratified accepted deviation (SEC-S6 MODIFY). Failure modes: worker preflight loud (good); pipeline plane never touches Postgres (training/evaluate are file+mlflow only) — reference DEP rows, no new gate.
3. **Package indexes** — PyPI only: uv.lock pins files.pythonhosted.org URLs WITH hashes; NO uv.toml exists (glob verified empty); no index-url/tool.uv overrides in pyproject or scripts. Gate = lock hashes themselves; the ungated moment is any `uv lock` refresh (adjacent board row ENV-NUMPY-SINGLE-PIN already owns pin hygiene). NOT-GATE beyond that reference.
4. **Provider API endpoints (DEEPSEEK/NVIDIA/OPENROUTER/ROUTER/ORCA)** — five live keys in `.env`, ZERO in-repo consumers (repo-wide grep: only the secret-audit factsheet names them; no endpoint URL for any provider exists anywhere in-tree; python-dotenv is a dependency but load_dotenv is called NOWHERE). These are credentials for tooling OUTSIDE this repo. Endpoint/connection-gate analysis is therefore vacuous in-repo; parity/ownership is SEC-S5→SLATE A. Reference only.

## 4 · TIME & RNG EXTERNALITY

Wall-clock reads outside evidence fields — full census (10 sites): timeline/runners.py:58 (`_now()` step timestamps = evidence fields), timeline/decide.py:44 (`decided_at` = evidence), samples/generate.py:146 (`created_at` = evidence), baseline/module.py:32 (`created_at` = evidence), training/trainer.py:62-64 (`time.time()` elapsed = metric, persisted via TrainingResult per D#29-g CLOSED-POSITIVE), experiments multivariate 01/02/03 `created_at` ×3 (evidence), experiments_ui.py:882 (`updated_at` UI metadata), agents/tools/render_gates.py:402 (ledger rendering date). **Consumer-ordering/naming dependency on clock found: NONE** — artifact filenames derive from step_id/node_id/model alias, never timestamps; champion alias is an explicit tag, not latest-by-time. Per mission rule ⇒ zero new candidates; DET-d (DEFER→REPRO-TIME-FREEZE, incl. BROADWAY_ARTIFACT_CLOCK idea, D#26) remains sole owner of the class.

RNG: all production surfaces seeded via config-required `random_state` (schema.py:192,228,278) threaded through splitter/cv/hpo/explain; registry seeding policy test-pinned (test_training_contracts.py:163-172); qq demo global seed sits behind `__main__` guard (qq.py:820 — DET-c rejected); remaining literal seeds in experiments are ruled (DET-C1/C3/a/b/f). No new candidates.

---

## CANDIDATES (PROPOSE-GH3 blocks)

### PROPOSE-GH3-01 — bootstrap-existence gate for out-of-repo lookup symlink
- **FINDING:** `data/raw/taxi_zone_lookup.csv` → absolute target OUTSIDE the repo (live symlink, verified). Consumed as a plain path from configs/dataset/taxi.yaml:33,47 and configs/project/taxi.yaml:17 via project/data.py LOOKUP_PATH/_load_zones and the src ingest join. On any machine but this one (or after the learning checkout moves) the failure is a bare pandas FileNotFoundError naming only the RELATIVE path — the out-of-repo nature, the symlink, and the remedy are invisible at the crash site. D#7/SLATE-B lands README prose; prose does not fail loudly.
- **root:** environment-resident resource treated as in-tree contract; consumption boundary has no precondition check.
- **GATE (proposed):** at both lookup consumption boundaries (src data-loader join site and project/data.py `_load_zones`), raise ValueError naming (a) configured path, (b) resolved symlink target, (c) the bootstrap command (README anchor). ~6 lines/site + tripwire test asserting the message mentions "symlink"/"bootstrap". Test-first; complements (does not duplicate) SLATE-B docs edit.
- refs: factsheet 2026-08-24-secret-audit:33 · D-verdicts #7 · MAIN_AGENT_CONTRACT.md:314.

### PROPOSE-GH3-02 — single configs-root owner; kill the project-plane split-brain
- **FINDING:** project/data.py:31-47 loads SEVEN yaml files via CWD-relative hardcoded `Path("configs/…")` at IMPORT time, bypassing loader.py CONFIGS_DIR/BROADWAY_CONFIGS_DIR. Two failure classes: import-from-wrong-CWD crashes with relative-path-only error; and when BROADWAY_CONFIGS_DIR redirects the src-plane (C-S14's warned scenario), project-plane silently stays on the original tree — the two planes validate different config universes in one process, with no signal.
- **root:** two independent declarations of "where configs live", neither cross-checked.
- **GATE (proposed):** route project/data.py loads through `broadway.config.loader` (or at minimum a shared repo-root constant); tripwire test asserting project-plane honors BROADWAY_CONFIGS_DIR identically to loader (same-dir fixture). Fixes import-outside-root class as a side effect. Rides the spirit of S14 but distinct mechanism (root unification vs warn).
- refs: loader.py:34 · C-verdicts #20/S14 · project/data.py:126 (RESULTS_DIR same class).

### PROPOSE-GH3-03 — numeric env-parse teeth + dedup for threshold vars
- **FINDING:** `BROADWAY_IDENTIFIER_THRESHOLD` is parsed by THREE independent readers with duplicated default "0.95" — two of them float() AT MODULE IMPORT (onboard/infer.py:10, discover/module.py:24) vs one call-time (reports/audit.py:30). Garbage value ⇒ bare `ValueError: could not convert string to float: 'x'` during import, env var unnamed, site unidentified. Same parse-teeth class: BROADWAY_QQ_MIN_UNIQUE int() (qq.py:111, at least call-time and test-pinned test_qq.py:589).
- **root:** env surface parsed ad hoc at N sites; no named-constructor discipline like the rest of the codebase's ValueError convention.
- **GATE (proposed):** one helper (e.g. utils.env_float/env_int(name, default)) raising ValueError naming the VAR; migrate the 4 sites; move infer.py/discover parses to call time; single tripwire test feeding garbage. ≤3 files beyond the four touched sites — if over budget, minimum slice: name-the-var in existing parses + dedupe default constant.

### PROPOSE-GH3-04 — provenance tooth for the ambient CI switch
- **FINDING:** `CI=true` silently halves/reduces the ingested population (etl ci_sample_size path). Platform twin records a reason string ("CI sampling: -N rows") into the audit trail (module.py:77-78, contract gates.yaml:126) — visible only post-hoc INSIDE artifacts; nothing signals at ENTRY. Legacy twin `sample_for_ci` (project/etl/process.py:43-48, declared LEGACY at gates.yaml:127) emits exactly ONE `logger.info` line and NO reason record. An inherited/exported CI variable in a developer shell (common with CI-helper dotfiles) therefore produces plausible-but-population-shifted evidence with no entry-point warning on either path.
- **root:** behavior keyed to an ambient env var with no provenance signal — identical hazard shape to GATE-CFG-70b (BROADWAY_CONFIGS_DIR, C-S14), which ruled warn-and-continue.
- **GATE (proposed):** parity with S14: when `os.getenv("CI") == "true"` at etl entry, emit warnings.warn/log WARNING naming the effective sample size AND ensure legacy twin records the same reason string the platform twin does (or delete the twin — separate question already tracked as LEGACY). Test-first via monkeypatched env.

### PROPOSE-GH3-05 — MLflow connectivity teeth: widen marker net + preflight precedent
- **FINDING:** setup_mlflow's helpful RuntimeError fires ONLY on three text markers covering connection-REFUSED (mlflow_utils.py:47-51; comment admits MLflow hides __cause__). Timeout/DNS/reset failures (`Read timed out`, `Name or service not known`, connection-reset) fall through as raw MlflowException with the hint lost. Guard covers setup only — subsequent log_*/registry calls during a long training run hit the network unguarded. Store-location duality compounds diagnosis: README `.mlflow.db`@root vs docker-compose `sqlite:///mlflow/mlflow.db` (docker-compose.yml:12) vs file-store `mlruns/`; stale-registry skew is acknowledged (README:16) with only a manual reset ritual.
- **root:** failure-classification by substring against a third party's opaque wrapper, enumerated from one observed failure mode.
- **GATE (proposed):** extend _CONNECTION_REFUSED_MARKERS with timeout/DNS markers + unit test feeding synthetic messages (pure function — cheap); OPTIONALLY clone worker preflight() (03_optuna_worker.py:118-138, already loud/FATAL) into training entry for http URIs. Store-location unification is a docs/compose one-liner riding IMAGE-TAG-COHERENCE (#17) if adopted.

### PROPOSE-GH3-06 — one root-owner for the five output-tree env vars
- **FINDING:** Five output/input roots are each resolved by scattered os.getenv calls with duplicated cwd-relative defaults: ARTIFACTS_DIR ×3 sites (two of them inline re-derivations inside audit.py:672-673 rather than an import of a shared constant), LINEAGE/TIMELINE/REPORTS/DATASET dirs one site each. Defaults are CWD-relative, so off-root invocations scatter artifacts silently into wherever the shell stands (mkdir -p happily creates them). Writer/reader pairs can resolve different trees if modules disagree or env flips mid-process. k8s node-local-disk angle already owned by DEP-M8→DEPLOY-F2 — this proposal is the LOCAL single-source-of-truth half.
- **root:** paths.py exists (reports/paths.py:6) but covers only REPORTS; no sibling owns artifacts/lineage/timeline/dataset roots.
- **GATE (proposed):** extend the paths-module pattern to all five roots; modules import constants; tripwire greps that no other `os.getenv("BROADWAY_(ARTIFACTS|LINEAGE|TIMELINE|REPORTS|DATASET)_DIR")` site appears. Optional rider: resolve defaults against repo-root anchor instead of bare CWD.

### PROPOSE-GH3-07 — unify BROADWAY_MLFLOW_CONFIG resolution style
- **FINDING:** Same var, two styles: eager import-time default evaluation (experiments/mlflow/_common.py:60 — Path built even when env set, harmless but the pattern 03_optuna_worker.py:44-46 explicitly flags and avoids in its own twin) vs lazy membership check (worker :46-47). Drift-prone twins for one contract; neither documented outside code comments/k8s generator.
- **root:** copy-evolution between local runner and k8s worker without a shared helper.
- **GATE (proposed, small):** extract one `resolve_hpo_config_path()` used by both; assert-equality tripwire test. Lowest priority of the seven.

## NOT-GATES (one-line reasons)

- `BROADWAY_CONFIGS_DIR` silent repoint — COVERED: C#20→S14 warn landed in slate.
- `${DATABASE_*}`/`${MLFLOW_TRACKING_URI}` unset passthrough — COVERED: C#21/#22→S1 residual-`${` scan; deploy env contract D#10→DEPLOY-F2.
- `.env` five API keys / example drift — COVERED: SEC-S5→SLATE-A parity probe; zero in-repo consumers (endpoint gates vacuous in-tree).
- Dev literal localhost creds — COVERED: SEC-S6 ratified accepted deviation.
- Ambient `MLFLOW_TRACKING_URI` for HPO tracker — COVERED by DEP-F2 env-block lane (worker side already prefights loudly).
- Wall-clock reads (all 10 sites) — evidence/metric fields only; zero ordering/naming consumers found; class owned by DET-d→REPRO-TIME-FREEZE.
- RNG seeds/config random_state — COVERED: DET-C1/C2/C3/a/b/c/f rulings + test_training_contracts seeding pins.
- `.uv-cache`/`.mplconfig` — script-defaulted (ship.sh:10, run_local_ci.sh:18) + documented (TODO.md:66-67, G0B).
- `/tmp/backend_uri` handshake (entrypoint-mlflow.sh) — container-local lifetime, not a cross-run contract.
- `${TMPDIR:-/tmp}/broadway-mpl` — CI-script-scoped, recreated per run; stale-cache class only.
- k8s lifecycle.sh env family (DB_*/VIEW_*/OPTUNA_*) — self-defaulting ops script; manifest-env validation owned by K8S-CONFIGMAP-ENV lane.
- compose MLFLOW_PORT/DATABASE_PORT defaults — deployment-plane rot owned by IMAGE-TAG-COHERENCE row (#17).
- Package index surface — hash-pinned uv.lock, no custom index anywhere; refresh-moment hygiene already ENV-NUMPY-SINGLE-PIN adjacent.
- `MLFLOW_ALLOW_FILE_STORE` — deliberate written shim for mlflow 3.x file-store unlock, commented at both sites.
- `DISABLE_PANDERA_IMPORT_WARNING` — untracked-WIP experiment, third-party warning suppression only.
- `PARITY_MAIN_DAY` — deleted dialect; test_branch_parity_scripts.py:8 asserts its ABSENCE (a working gate already).

## Tally

- Census: ENV 14 Python-plane names (+11 shell-plane) · FS 7 contract areas · NET 4 endpoint surfaces · TIME 10 wall-clock sites + RNG class clean.
- CANDIDATES: **7** PROPOSE-GH3 blocks (01–07; 02/03/06 are the teeth-spine).
- NOT-GATES: 16 rows, each referenced to an owning ruling or scoped-out reason.
- References honored, none duplicated: C-S14/S1, D SLATE-A/B, DEP-F2/M8, K8S-CONFIGMAP-ENV, IMAGE-TAG-COHERENCE, DET-d/C1/C2/C3/a/b/c/f, ENV-NUMPY-SINGLE-PIN.

## Judgment — single most dangerous ungate external dependency

**The ambient `CI=true` switch (GH3-04).** It is the only external dependency whose failure mode is *scientifically corrupting while looking perfectly healthy*: any stray exported CI variable collapses the ingested population to `ci_sample_size` rows on BOTH etl paths, every downstream stat/training artifact remains internally consistent, plausibly-sized, and green — and the only traces are one post-hoc reason string buried inside platform artifacts and a single `logger.info` line on the legacy twin (process.py:45) that default logging configurations drop. Pasted evidence, project/etl/process.py:43-48:

```python
def sample_for_ci(df: pd.DataFrame) -> pd.DataFrame:
    if os.getenv("CI") == "true":
        logger.info(f"CI mode detected: sampling {cfg.ci_sample_size:,} raw rows for fast processing.")
        if len(df) > cfg.ci_sample_size:
            df = df.sample(n=cfg.ci_sample_size, random_state=cfg.random_state)
    return df
```

and src/broadway/etl/module.py:74-78 (platform twin — reason recorded, severity absent):

```python
    if cfg.etl.ci_sample_size > 0 and os.getenv("CI") == "true":
        n_before = len(df)
        df = df.sample(n=min(cfg.etl.ci_sample_size, len(df)), random_state=rs)
        if len(df) < n_before:
            reasons.append(f"CI sampling: -{n_before - len(df)} rows")
```

Runners-up: the out-of-repo lookup symlink (GH3-01) has wider blast radius on foreign machines but fails LOUD and already has a slated doc remedy; the project-plane configs split-brain (GH3-02) is silent but requires the comparatively rare BROADWAY_CONFIGS_DIR-set scenario.

## Assumption audit

Beyond-brief checks: (1) verified the symlink resolves TODAY (`ls -la data/raw`) so GH3-01 is prevention, not incident response; (2) grep-proved ZERO in-repo consumers/endpoints for the five provider API keys before declaring network-analysis vacuous; (3) confirmed no uv.toml and no custom index config before calling the supply surface PyPI-hash-pinned; (4) found the legacy CI twin DOES log INFO — corrected the brief's implied "fully silent" to "log-only, unrecorded"; (5) checked filename/ordering dependence on clock across writers (none — timestamps never enter filenames or sort keys in prod code), keeping the TIME column all-NOT-GATE per the mission's own rule rather than deferring wholesale; (6) confirmed DATA_MODE is README-documented so only its import-freeze semantics are candidate material (folded into census, no separate block); (7) render_gates.py:402 clock use is ledger tooling, out of product scope.

## OPEN QUESTIONS

1. Should `sample_for_ci` (LEGACY twin) be deleted outright instead of receiving the reason-record tooth? Existing LEGACY designation at gates.yaml:127 suggests yes — needs an owner ruling; GH3-04 works either way.
2. Is `DATA_MODE`'s import-time freeze intentional API (constants-for-speed) or accident? If accident, it folds into GH3-03's call-time migration.
