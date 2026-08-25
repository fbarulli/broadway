# GAP-SHEET — Untracked-Object-Creator Census (GH-6 contract)

- Date: 2026-08-24 · Plane sweep: git-derived / GitHub-side / container / python-toolchain / library-minted / agent-session scratch
- Verification: repo `/home/opc/ONE/broad-way`, branch `sklearn`, HEAD `5016e937e6aa67b301e3b54b78b1891f85784c67` — **matches expected prefix `5016e93` ✓**
- Read-only census; every git read executed under `GIT_OPTIONAL_LOCKS=0`; zero git mutations; this sheet is the only file created.
- Tool availability honestly recorded: `docker` 29.6.2 (daemon UP) · `kind` v0.30.0 (present) · `gh` 2.96.0 (authed as fbarulli; token scopes lack `read:packages` → GHCR package inventory returned HTTP 403/404, recorded as unverifiable, not invented).

Verdict vocabulary: **GATE-CANDIDATE** (real control gap → PROPOSE-GH6-NN) · **ACCEPTED-LOCAL** (documented machine-local, gate exists or law covers it) · **ORPHANED** (nobody owns — flagged loudly).

---

## PLANE 1 — GIT-DERIVED

| # | Object | Creator (site where traceable) | Location | Lifecycle owner | Verdict |
|---|--------|-------------------------------|----------|-----------------|---------|
| 1.1 | Annotated tag `tier-1-complete` → 0d383125 ("Tier 1 complete: F6+TB1-4+D16-PARITY all landed", tagger fbarulli 2026-08-24) | Manual `git tag -a`; NO scripted minter found (`git tag` grep over scripts/agents/.github = zero hits); no custody law names tags | `refs/tags/tier-1-complete` — LOCAL ONLY (absent from `refs/remotes/origin`) | Nobody — no minting law, no retirement law; pins a commit carrying the SECRET-1 old credential per factsheets/2026-08-24-secret-audit.md | **GATE-CANDIDATE PROPOSE-GH6-01** (tag-minting custody) |
| 1.2 | Branch `pr-1` @ a7a6bd6 | Fetch of `refs/pull/1/head` (FETCH_HEAD evidence, 2026-08-24 20:54) | `refs/heads/pr-1` | None — PR #1 CLOSED-unmerged 2026-08-13; branch still carries old credential (secret-audit) | **GATE-CANDIDATE PROPOSE-GH6-02** (stale-ref retirement law) |
| 1.3 | Branch `pr-2` @ 1f76d97 | Same — `refs/pull/2/head` | `refs/heads/pr-2` | None — PR #2 CLOSED-unmerged 2026-08-13; credential-bearing | **GATE-CANDIDATE PROPOSE-GH6-02** (same proposal) |
| 1.4 | Branch `sp_probe` @ ee014cf, author `adversary <adv@probe>`, subject "spaces+unicode" | Adversarial-lane probe commit 2026-08-24 09:27 | `refs/heads/sp_probe` | NOBODY — no owning lane recorded anywhere; named ref is gc-immune indefinitely | **ORPHANED** ⚠ danger #1 |
| 1.5 | Reflog-only commits: **367** commits reachable ONLY via reflog (HEAD reflog 679 entries) | Every reset/rebase/amend/commit of every agent session | `.git/logs/**` + loose/pack objects (`count-objects`: 7292 loose / 6192 in-pack) | Git defaults only (reflogExpire 90d/30d + eventual gc); zero project law | **ORPHANED** (runner-up danger; holds pre-secret-fix lineage) |
| 1.6 | Pseudo-refs `ORIG_HEAD` (7ece650, Aug-24 10:15), `FETCH_HEAD` (pull/1+pull/2, Aug-24 20:54) | Mechanical git bookkeeping (reset/fetch) | `.git/ORIG_HEAD`, `.git/FETCH_HEAD` | Auto-overwritten by next git op | ACCEPTED-LOCAL |
| 1.7 | Remote-tracking refs ×4 (`origin/{main,sklearn,taxi,taxi_work}`) | fetch/push — push solely via `scripts/ship.sh` (raw `git push` = policy violation per MAIN_AGENT_CONTRACT §push-custody) | `refs/remotes/origin/*` | Mirrors; custody ratified D19/e1f7cc62 | ACCEPTED-LOCAL |
| 1.8 | Stashes | — | — | `git stash list` = **0** | clean (none) |
| 1.9 | Worktrees | — | `git worktree list` = **main only**; ZERO leaked detached worktrees — D19/P3 ephemeral-worktree policy VERIFIED CLEAN | Policy holds | clean (verified) |
| 1.10 | Unpushed HEAD: `sklearn` ahead 1 of `origin/sklearn` (5016e93 exists locally only until push) | This session's DOCS-TRUTH r1 commit | local ref only | Custody clear — sole push path is ship.sh full-tier gates | ACCEPTED-LOCAL |

Plane tally: 1 tag · 3 flagged branches · 4 remote refs · 2 pseudo-refs · 367 reflog-only commits · 0 stashes · 0 leaked worktrees. Verdicts: GATE 3 · ACCEPTED 3 · ORPHANED 2.

## PLANE 2 — GITHUB-SIDE (gh CLI available + authed)

| # | Object | Creator | Location | Lifecycle owner | Verdict |
|---|--------|---------|----------|-----------------|---------|
| 2.1 | Workflow-run history: workflow "CI" active + auto-added "Dependency Graph"; **276 runs** | push events (ci.yml) | Actions store (metadata + logs, GitHub-default retention) | GitHub defaults; no project law needed for metadata | ACCEPTED-LOCAL |
| 2.2 | Artifacts: **11 non-expired**, all `verified-images-<sha>` @ 1,743,236,760 B each, Σ **19,175,604,360 B ≈ 17.85 GiB** | `actions/upload-artifact@v4` ci.yml:229 | GitHub artifact store | **GOVERNED — declared `retention-days: 1` (ci.yml:234)**; transient by design | ACCEPTED-LOCAL (gate exists) |
| 2.3 | Actions caches: **19 keys Σ 11,159,207,219 B ≈ 10.39 GiB — sitting AT the ~10 GB/repo LRU ceiling**; `broadway-base-*` ≈742 MiB each; `setup-uv-*` key DUPLICATED ×2 (~434 MiB each, saved twice within 60 s) | `actions/cache@v4` ci.yml:145 | GitHub actions-cache store | Only silent global LRU eviction; no purge/dedupe/size law | **GATE-CANDIDATE PROPOSE-GH6-03** (cache-retention law) |
| 2.4 | GHCR writes: CD job pushes `ghcr.io/<repo>/{broadway-base,broadway-optuna-worker,mlflow-server}:{sha,latest,taxi}` (ci.yml:291–301) | CI CD job on green push | ghcr.io packages | Writer identity defined (CI-only) BUT package inventory/version-count UNVERIFIABLE this session (token lacks `read:packages`: HTTP 403 user-list, 404 direct) — version-cap/deletion law silent | **GATE-CANDIDATE PROPOSE-GH6-04** (registry-write ownership + version cap; state unverifiable — recorded honestly) |
| 2.5 | Releases | — | — | `gh release list` = **0** | clean (none) |
| 2.6 | Labels: 9 stock labels, untouched | GitHub bootstrap | repo settings | stock defaults | ACCEPTED-LOCAL |
| 2.7 | Ledger issues #3/#4/#5: comments 2 / 4 / 30; **attachment links 0/0/0** (asset-link scan over bodies+comments); GitHub stores append-only comment bodies already governed by the §14 event-id protocol | main-agent board rows | GitHub issues | Locked-append-only doctrine (#3/#4/#5) | ACCEPTED-LOCAL |

Plane tally: 276 runs · 11 artifacts (17.85 GiB, gated) · 19 caches (10.39 GiB, AT ceiling) · 0 releases · 9 labels · 36 ledger comments / 0 attachments. Verdicts: GATE 2 · ACCEPTED 5 · ORPHANED 0.

## PLANE 3 — CONTAINER (docker daemon UP; kind present, 0 clusters)

| # | Object | Creator | Location | Lifecycle owner | Verdict |
|---|--------|---------|----------|-----------------|---------|
| 3.1 | Images `broadway-base:{latest,test}`, `broadway-optuna-worker:{latest,test}`, `mlflow-server:{latest,test}` — 6 × ~4.61 GB (shared layers), built 2026-08-19 | `docker build` from k8s/optuna Dockerfiles.{base,worker,mlflow}; loaded into kind via `lifecycle.sh:100 kind load docker-image …`; `:test` variants from probe-era builds | local image store (~27.7 GB nominal, layer-shared) | `teardown.sh` deletes the CLUSTER but never `rmi`s images; no prune law | **GATE-CANDIDATE PROPOSE-GH6-05** (image-retention law / teardown-rmi) |
| 3.2 | Running container `broadway-view-pg` (postgres:16-alpine, cmd=`postgres`, **Up 5 days**, created 2026-08-19T10:57:42Z, NO compose labels, anonymous vol f60f69…) | Manual `docker run` — exact invocation untraceable (compose-project label empty) | docker daemon | NOBODY documented — ungoverned long-running durable DB | **ORPHANED (running)** ⚠ |
| 3.3 | Anonymous volumes ×3: f60f69… (used_by=1, view-pg data) · 43d33c… (used_by=**0**) · c122df… (used_by=**0**) | postgres image VOLUME + removed containers | docker volume store | Dangling pair owned by nothing | **ORPHANED** (dangling pair) |
| 3.4 | kind clusters: **NONE** (`kind get clusters` empty — teardown honored) BUT remnant image `kindest/node:<none>` 1.47 GB (12 months old) + `~/.kube/config` stub (28 B) | prior cluster creation via `k8s/optuna/lifecycle.sh:97` | image store + ~/.kube | Cluster lifecycle governed; node-image/kubeconfig remnants not | **ORPHANED** (remnants; cluster set itself clean) |
| 3.5 | Base pulls `python:3.12-slim`, `postgres:16-alpine` | build/compose inputs | local store | base-image refresh convention | ACCEPTED-LOCAL |
| 3.6 | Operator tooling images: dozzle v10.7.2, redis:7-alpine, busybox, moby/buildkit, shellcheck-alpine, kubeconform v0.6.7 (pinned by ci.yml:67) | manual/harness ops | local store | machine-local tooling | ACCEPTED-LOCAL |
| 3.7 | *Observation:* compose stack (docker-compose.yml services mlflow+postgres, named vols `mlflow_data`/`pg_data`) NEVER materialized — no named volumes exist ⇒ definition without runtime object | — | — | — | (note, no verdict) |

Plane tally: 15 images (6 project-class) · 1 running container · 3 volumes (2 dangling) · 0 clusters. Verdicts: GATE 1 · ACCEPTED 2 · ORPHANED 3.

## PLANE 4 — PYTHON TOOLCHAIN DROPPINGS

| # | Object | Creator | Location | Lifecycle owner | Verdict |
|---|--------|---------|----------|-----------------|---------|
| 4.1 | `.venv` **2.2 G** | `uv sync` per `uv.lock` | repo root | regenerable; `.gitignore:2` | ACCEPTED-LOCAL |
| 4.2 | Repo-root `.uv-cache` **3.4 G** + home `~/.cache/uv` **7.7 G** + `~/.local/share/uv` 347 M — **11.45 G across two cache roots** | **EXACT SITE: `scripts/ship.sh:10` — `export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"`** forks a second cache root into the repo; bare uv uses `$HOME/.cache/uv` | repo root + home | Zero prune law; ship.sh grows the fork on every run | **GATE-CANDIDATE PROPOSE-GH6-06** (cache-retention law; mission-named gap CONFIRMED REAL) |
| 4.3 | `.mplconfig` 48 K (fontlist-v390.json + CACHEDIR.TAG) | same line — ship.sh:10 exports `MPLCONFIGDIR="$PWD/.mplconfig"` (run_local_ci.sh:18 uses the /tmp variant) | repo root | regenerable; `.gitignore:26` | ACCEPTED-LOCAL |
| 4.4 | `.mypy_cache` **78 M** (16 sqlite shards, 3.4–7.8 M each, hot Aug-24 20:24) | mypy incremental cache | repo root | tool-managed; `.gitignore:5` | ACCEPTED-LOCAL |
| 4.5 | `.pytest_cache` 32 K | pytest | repo root | `.gitignore:6` | ACCEPTED-LOCAL |
| 4.6 | `.ruff_cache` 736 K | ruff | repo root | SELF-ignoring (embedded `.gitignore:*`) — hence invisible to status despite no root rule | ACCEPTED-LOCAL |
| 4.7 | `__pycache__` ×**106 dirs / 646 *.pyc** (incl. root + scripts/) | CPython bytecode | throughout src/tests/scripts/root | `.gitignore:3-4` | ACCEPTED-LOCAL |
| 4.8 | `broadway.egg-info` | editable install metadata (uv/pip -e) | repo root | `.gitignore:9` | ACCEPTED-LOCAL |
| 4.9 | `~/.cache/ms-playwright` **1.3 G** (+ `~/.cache` Σ 11 G) | browser binaries pulled by harness/probe sessions | home | machine-local operator footprint | ACCEPTED-LOCAL (noted) |
| 4.10 | `data/processed/` ≈ **551 MB**: taxi_canonical 149.6 M · train_features 119.7 M · train 111.9 M · training_data 107.0 M · val_features 30.1 M · val 28.2 M · joined_sample_live 3.1 M · test_canonical + 3 audit JSONs · **`feature_pipeline.pkl` 3,348 B (= the KNOWN pipeline.pkl)** | src/broadway etl/features pipeline stages | data/processed | Whole tree ignored (`.gitignore:13`), regenerable — BUT two generations coexist same-day (train.parquet 08:18 vs training_data.parquet 09:31 vs *_features 15:47) | ACCEPTED-LOCAL (staleness noted) |

Plane tally: 11.45 G uv caches · 2.2 G venv · 78 M mypy · 551 MB processed data · 106 pycache dirs. Verdicts: GATE 1 · ACCEPTED 9 · ORPHANED 0.

## PLANE 5 — LIBRARY-MINTED FILES AT RUNTIME

| # | Object | Creator | Location | Lifecycle owner | Verdict |
|---|--------|---------|----------|-----------------|---------|
| 5.1 | `./mlruns/` 532 K MLflow file-store with **5 registered models** (m-e61e973c…, m-3823d9b4…, m-37c048f7…, m-38281944…, m-895a6ad7…) each + `model.pkl` + `./.mlflow.db` sqlite 880,640 B | Documented command README:15 `uv run mlflow server --backend-store-uri sqlite:///$(pwd)/.mlflow.db --artifacts-destination file://$(pwd)/mlruns`; tracking via `src/broadway/training/mlflow_utils.py:58,204` | repo root | **LAW EXISTS:** README:16 reset (`rm -f .mlflow.db && rm -rf mlruns`), README:21 "regenerable demo registry … never [committed]"; `.gitignore:7,32` | ACCEPTED-LOCAL (documented machine-local) |
| 5.2 | `/home/opc/ONE/mlruns` 8 K — **OUTSIDE the repo, in the parent dir** | an mlflow run executed with cwd=`~/ONE` (cwd-leak) | `/home/opc/ONE/mlruns` | Nobody; invisible to every repo-hygiene sweep | **ORPHANED** ⚠ |
| 5.3 | Optuna studies: `RDBStorage` wired at `src/broadway/training/optuna.py:63` + `hpo.py:144` → in-kind postgres ⇒ cluster deleted ⇒ **zero surviving study storage** locally; no optuna sqlite at HEAD paths; `.gitignore:30` reserves `data/optuna-backup/` (absent today) | training/HPO stages | was: kind-cluster postgres | lifecycle rides cluster lifecycle (governed) | ACCEPTED-LOCAL (evidence-based absence) |
| 5.4 | Matplotlib outputs OUTSIDE reports/figures: `legend_experiment/right/*.png` (×6), `diagnostics_experiment/{zscore,ratio,bars}/*.png` (×6), `experiment_ols/*/residuals.png` (×4), plus experiments/results renders | legacy experiment-tree scripts rendering at runtime | repo subdirs | Deliberately excluded by `.gitignore` (`*.png` :23; dirs :26–28) — declared-local renders; `reports/figures/*.png` whitelisted (:24) | ACCEPTED-LOCAL (declared) |
| 5.5 | Log files inside repo | — | — | `find -name '*.log'` = **0 hits** (logs live only under /tmp → Plane 6) | clean (none) |

Plane tally: 5 registered models + 1 sqlite (documented) · 1 outside-repo leak · ~16 ignored pngs · 0 repo logs. Verdicts: GATE 0 · ACCEPTED 3 · ORPHANED 1.

## PLANE 6 — AGENT-SESSION SCRATCH (this session's own footprint class)

| # | Object | Creator | Location | Lifecycle owner | Verdict |
|---|--------|---------|----------|-----------------|---------|
| 6.1 | `/tmp/broadway-e2e` 4.1 M — **PRESERVED INTENTIONALLY**; `MANIFEST-run1.sha256` present (5,639 B, Aug-24 22:46) anchoring artifacts-run1/reports-run1; sidecar `/tmp/broadway-e2e-port.txt` (`MLFLOW_PORT=46671`) | e2e verification harness (logs-{ingest,etl,…}.txt ×20, `.mlflow.db` 872 K, mlflow-server.log) | /tmp | intentional preservation (manifest-pinned) | ACCEPTED-LOCAL |
| 6.2 | `/tmp/broadway-e2e-1k` **2.1 M — NEWEST scratch (Aug-24 22:56)**: second full e2e tree, artifacts-A/B/C variants, hyp logs, own port.txt — NO manifest, NO claimant | unnamed later lane/session | /tmp | nobody | **ORPHANED** ⚠ |
| 6.3 | `/tmp/main-synth` **2.2 G — largest scratch item**: full synthetic-data pipeline tree incl. `docker/mlflow/Dockerfile`, artifacts-A/B/C, logs-*-hyp.txt | unknown session; **zero ownership references** greppable in agents/, scripts/, synth.md | /tmp | nobody | **ORPHANED** ⚠ danger #2 |
| 6.4 | `/tmp/broadway_verify_*/zones.csv` ×**10** (Aug-16→19, 1,564 B each) | temp-dir leak pattern; creator site **NOT FOUND** at HEAD (grepped src/, tests/, project/, scripts/, deepseek-harness/) — honestly recorded as untraceable | /tmp | nobody | **ORPHANED litter** |
| 6.5 | Probe debris ×~18: probe_b1{,b,c,d}.py, probe_c_inventory.py, probe_pipeline.py, probe_semantics.py, parity_probe.py, mlflow_probe{,2}.py, mypy_probe.py, guard_probe_test.py, `probe.db` 114 KB sqlite, fw_probe.log, e2e_baseline.out, e2e-clean-sync.log | ad-hoc investigation sessions | /tmp | nobody | **ORPHANED litter (low)** |
| 6.6 | pytest tmpdir mlruns leaks ×3: /tmp/tmpog6vzwce 40 K · tmp9onvft5k 44 K · tmpad_vi80w 164 K | test-session TemporaryDirectory remnants | /tmp | nobody | **ORPHANED litter (low)** |
| 6.7 | Detached worktrees crosscheck | — | `git worktree list` re-run: main only — matches Plane 1.9; nothing leaked | policy holds | clean (verified) |

Plane tally: 1 preserved tree · 1 unclaimed e2e clone · 1 × 2.2 G synth tree · 10 verify-dirs · ~18 probe files · 3 tmpdir leaks. Verdicts: GATE 0 · ACCEPTED 1 · ORPHANED 5.

---

## TOTALS

| Verdict | P1 | P2 | P3 | P4 | P5 | P6 | Σ |
|---|---|---|---|---|---|---|---|
| **GATE-CANDIDATE** | 3 | 2 | 1 | 1 | 0 | 0 | **7** |
| **ACCEPTED-LOCAL** | 3 | 5 | 2 | 9 | 3 | 1 | **23** |
| **ORPHANED** | 2 | 0 | 3 | 0 | 1 | 5 | **11** |
| clean/none rows | 2 | 0(+1 note) | 0(+1 note) | 0 | 1 | 1 | 5 |

Gate proposals minted: **PROPOSE-GH6-01** tag-minting custody · **02** stale-ref retirement (closed-PR snapshots) · **03** CI cache retention/dedupe · **04** GHCR registry-write custody + version cap (state unverifiable) · **05** container-image retention (teardown-rmi) · **06** cache-root fork law (ship.sh UV_CACHE_DIR/MPLCONFIGDIR).

## THREE MOST DANGEROUS ORPHANED OBJECTS

**#1 — branch `sp_probe` (gc-immune, adversary-authored, credential-lineage adjacent, zero owner)**
```
$ git show-ref | grep sp_probe
ee014cffb00e3d31dc4b2cac6677e028d666fbf0 refs/heads/sp_probe
$ git for-each-ref refs/heads/sp_probe --format='%(creatordate:iso-local) | %(committername)'
2026-08-24 09:27:34 +0000 | adversary
$ git log -1 --format='%H %ci %an <%ae>%n%s' sp_probe
ee014cffb00e3d31dc4b2cac6677e028d666fbf0 2026-08-24 09:27:34 +0000 adversary <adv@probe>
spaces+unicode
```
A named ref never expires, no lane claims it, and the secret-audit factsheet lists the sp_probe lineage among locals still carrying the SECRET-1 old credential.

**#2 — `/tmp/main-synth` (2.2 GB, largest unknown footprint, embeds its own Dockerfile, zero ownership references)**
```
$ du -sh /tmp/main-synth
2.2G	/tmp/main-synth
$ ls /tmp/main-synth/docker/mlflow
Dockerfile
$ ls /tmp/main-synth   (head)
artifacts  artifacts-A  artifacts-B  artifacts-C  configs  data  logs-*-hyp.txt … mlruns  port.txt  reports
$ grep -rn 'main-synth' agents/ scripts/ synth.md   →  no hits
```

**#3 — running container `broadway-view-pg` + its anonymous volume (live ungoverned durable store, 5 days up, creator untraced)**
```
$ docker ps -a --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
broadway-view-pg   postgres:16-alpine   Up 5 days
$ docker inspect broadway-view-pg --format '{{.Created}} | {{.Config.Image}} | {{join .Config.Cmd " "}} | compose={{index .Config.Labels "com.docker.compose.project"}}'
2026-08-19T10:57:42.816633533Z | postgres:16-alpine | postgres | compose=
$ docker volume ls -q + used_by check
43d33cea… used_by=0   c122df64… used_by=0   f60f691e… used_by=1  ← view-pg data
```
(Runner-up: the **367 reflog-only commits** — unreachable-from-any-ref history mass held only by git's default expiry, including pre-secret-fix lineage; deterministic ≤90-day decay, no project owner.)

---
*Census method: read-only probes only (`GIT_OPTIONAL_LOCKS=0`; docker/kind/gh queried read-only; gh package API denial recorded verbatim). Single file created per GH-6 contract.*
