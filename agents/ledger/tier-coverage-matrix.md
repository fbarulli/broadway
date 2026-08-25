# Tier-coverage matrix — where each check class actually runs

> **Class: DIAGNOSTIC** — informational only; **no failing gate** references this
> document. Created per the capture-mechanism ruling (#3 diagnostic-only): the
> ruling forbids touching `gates.yaml`/`DIGEST.md`, so this matrix lives as a
> standalone ledger doc. Purpose: make exclusions *visible* — every EXCLUDED
> cell below was, at some point, an implicit one.
>
> Derived from live files at HEAD `1cf33b5`, working tree 2026-08-25.
> BASIS DECLARATION: `.github/workflows/ci.yml` was mid-edit (+66/−2) under a
> sibling lane when derived; cited line numbers are **working-tree**
> coordinates, which became landed coordinates at this batch's landing commit
> (HEAD copy at derivation = 304 lines; final working tree = 373 — the +5 are
> the sibling lane's deferred-stub loudness block inside build-and-boot; all
> 12 spot-checked anchors verified post-derivation by adversarial review).
> Sources: `scripts/run_local_ci.sh` (read fully, 62 lines),
> `.github/workflows/ci.yml` (read fully, 373 lines).
>
> Column semantics per `run_local_ci.sh`: `--tier=fast` (:13) runs the five
> quick gates (<30s); default/`--tier=full` (:14) adds pytest+cov;
> `--static` (:12) runs the SAME five quick gates as fast (pytest is skipped
> by the single condition `$STATIC -eq 0 && $TIER == "full"`, :53) with the
> LOCAL-CI banner vocabulary instead of FAST-GREEN (:17, :58).

## Matrix

| Check class | local-fast | local-full | CI |
|---|---|---|---|
| parity-checker | COVERED — `run_local_ci.sh:43` (`run parity gate_parity`; F1b pin to `refs/remotes/origin/sklearn`, :30–42) | COVERED — same, :43 (runs on every tier) | COVERED — inherited: platform job runs the script whole, `ci.yml:46–47` |
| ruff | COVERED — `run_local_ci.sh:44–45` (scope incl. `experiments/fare_prediction`) | COVERED — same, :44–45 | COVERED — `ci.yml:47` |
| mypy | COVERED — `run_local_ci.sh:46` (scope `src/broadway` ONLY) | COVERED — same, :46 | COVERED — `ci.yml:47` |
| config-parse | COVERED — `run_local_ci.sh:47–51` (loads every `configs/experiment/*.yaml`) | COVERED — same, :47–51 | COVERED — `ci.yml:47`; PLUS partial extra: HPO spec `configs/experiments/mlflow.yaml` parsed in-image during boot sim, `ci.yml:243–253` |
| shell `sh -n` | COVERED — `run_local_ci.sh:52` first half (`for f in k8s/optuna/*.sh; do sh -n`); runs on every tier incl. `--static` (outside the :53 conditional) | COVERED — same, :52 | COVERED — explicit step `ci.yml:54–63` (`sh -n` loop :58–60) AND re-inherited via `ci.yml:47`. Scope `k8s/optuna/*.sh` ONLY |
| shellcheck | COVERED **since 2026-08-25** — `run_local_ci.sh:52` second half (`shellcheck k8s/optuna/*.sh`). Historically local-EXCLUDED until d440d16 (SHELLCHECK-PARITY-1) | COVERED — same, :52 | COVERED — `ci.yml:63` (with runner-guard comment :62). Scope `k8s/optuna/*.sh` ONLY |
| pytest+cov | EXCLUDED — by design: fast-tier contract "<30s" (`run_local_ci.sh:13`); skipped whenever `STATIC≠0 || TIER≠full` (:53) — so also skipped under `--static` | COVERED — `run_local_ci.sh:54–55` (`tests/`, `-n 4 --dist worksteal`, `--cov=src/broadway`, `--cov-fail-under=95`) | COVERED — `ci.yml:47` passes no args ⇒ full tier ⇒ suite included |
| kubeconform | EXCLUDED — needs docker; declared CI-only by name in `run_local_ci.sh:5`; absent from gate list :43–55 | EXCLUDED — same reason | COVERED — manifests sweep `ci.yml:76–96` (image pinned `kubeconform:v0.6.7` pulled :68; train-job Helm placeholders rendered first :81–82; kind-config ignored :91) |
| orchestrator dry-run | EXCLUDED — CI-only per `run_local_ci.sh:5–6` (renders + docker-validates worker Jobs) | EXCLUDED — same reason | COVERED — `ci.yml:101–120` (`render_worker_jobs.py` :107, assertion greps :111–113, kubeconform on rendered jobs.yaml :115–120) |
| build-and-boot | EXCLUDED — CI-only per `run_local_ci.sh:6` (heavy docker builds) | EXCLUDED — same reason | COVERED — job `ci.yml:148–277` (base cache :161–189; six image builds :176/:193/:200/:212/:219/:226; boot checks :236–277) |
| taxi experiments | EXCLUDED — `run_local_ci.sh:7`: "experiments.py verify runs on taxi only"; never invoked by this script on any tier | EXCLUDED — same | COVERED, **branch-gated** — `ci.yml:123–127` (`if:` taxi ref or PR base taxi, :124; `uv run python experiments.py verify`, :127) |

## Findings — the historically implicit exclusions this doc exists to surface

1. **shellcheck was local-EXCLUDED until 2026-08-25**: two reds escaped remote-only
   enforcement (SHELLCHECK-PARITY-1, FIXES.md) before commit d440d16 moved the gate
   into `run_local_ci.sh:52`. The row above keeps both eras visible rather than
   overwriting the history.
2. **kubeconform / orchestrator dry-run / build-and-boot remain CI-only BY DESIGN**
   (`run_local_ci.sh:5–7` names them deliberately): all three require docker, which
   the local tiers refuse to assume. A local developer can reach LOCAL-CI GREEN
   while these three classes have never run on their machine.
3. **taxi experiments is doubly invisible locally**: excluded on every local tier,
   and in CI it fires only on the taxi branch (:124) — an sklearn-tier developer
   never exercises it in either place.
4. **The cov≥95 floor only ever binds full-local and CI**: fast AND static tiers
   share the single skip condition (:53), so a "--static passed" doc edit asserts
   nothing about suite behavior.
5. **Scope asymmetries readable only from this table**: shell syntax/shellcheck
   cover `k8s/optuna/*.sh` but NOT `scripts/*.sh` — `ship.sh` and
   `run_local_ci.sh` themselves are validated manually only (the reason this
   matrix's sibling contract demanded a hand-run `shellcheck`); mypy sees
   `src/broadway` only; config-parse globs `configs/experiment/*.yaml` only, with
   `configs/experiments/*` HPO specs reached solely inside the CI boot simulation.
