# PACKET D — OPS-SECURITY (fact sheets ONLY)

- Packet id: `D-ops-security` · findings: **30 register rows covering 35 fact-sheet items**
  (secret-audit 7 · deploy-diff 12 · det-ledger 11 [3 verdict-context + silence list a–h] ·
  perf-baseline 5; multi-item rows and dedupes noted inline)
- Sources: agents/ledger/factsheets/2026-08-24-{secret-audit,deploy-diff,det-ledger,
  perf-baseline}.md · agents/ledger/FIXES.md §SECRET-1 incident (:195–206) ·
  agents/ledger/STATE.md §hazards. Gate-band findings are OTHER packets' custody.
- Required reading before ruling: all four factsheets end-to-end (short), FIXES.md:119–220.

## FINDING REGISTER — SECRET-AUDIT

1. SEC-S1 — SECRET-1 STILL OPEN: old DB credential sits at HEAD of PUBLIC origin/main + origin/taxi_work (+ locals/tag); removal landed sklearn-lineage only. **HUMAN-CALL candidate: upstream ROTATION is the kill, not history rewrite (D22).**
2. SEC-S2 — rotation NOT-EVIDENCED and ledger citation FALSE: commit `8b73645` claims "Ledger: FIXES.md"; no such row exists. **Backfill = human/board action (FIXES.md is another lane's WIP-modified file this session).**
3. SEC-S3 — D22 rider path wrong: "configs/secret.yaml" never existed in any ref; real subject k8s/optuna/secret.yaml.
4. SEC-S4 — remediation mechanics CLOSED on sklearn lineage only (template placeholders, lifecycle.sh openssl fallback mode 600, .gitignore) — context row: verify closure claim holds before ruling dependents.
5. SEC-S5 — .env vs .env.example drift BOTH directions: five live keys .env-only (DEEPSEEK/NVIDIA/OPENROUTER/ROUTER/ORCA), MLFLOW_TRACKING_URI+DATABASE_* example-only; .env gitignored (exposure via accidental staging, not tracking).
6. SEC-S6 — dev-only literal DB creds committed in configs/environment/development.yaml (len=8 pfx 'pos', differs from pushed value); staging/prod interpolate ${…}.
7. SEC-S7 — out-of-repo lookup path: data/raw/taxi_zone_lookup.csv → symlink OUTSIDE repo (/home/opc/ONE/learning/data/raw/…); consumed as plain path by configs + scripts; fresh-clone bootstrap UNDOCUMENTED (silent reproducibility break).

## FINDING REGISTER — DEPLOY-DIFF

8. DEP-F1a — F-1 STILL-BROKEN: inference/api.py is a docstring-only stub, no app object; manifest invokes bare `inference.api:app`, PYTHONPATH exposes broadway only → correct target broadway.inference.api:app (fastapi/uvicorn installed; pure wiring).
9. DEP-F1b — F-1 companion: api needs MLflow registry per docstring but manifest sets no tracking URI, mounts nothing.
10. DEP-F2 — F-2 STILL-BROKEN: train-job has no env block/volumeMounts, runs --environment production against 7 ${VARS}; unset vars pass literally → database_port int coercion crashes at load; local gates mask it via concrete dev defaults. [resolver ROOT owned by C#21 — rule the deployment consequence here]
11. DEP-M1 — postgres-deployment references configMap `environment` that NO manifest creates → CreateContainerConfigError.
12. DEP-M2 — train-job env: none vs 7 required ${VARS} → MISSING all.
13. DEP-M3 — api-deployment env: none vs MLflow URI need → MISSING.
14. DEP-M4 — HPA cpu 70 vs api_hpa_cpu_threshold: 80 — MISMATCH, config key never wired (and the key itself is dead per C#25).
15. DEP-M5 — mlflow-deployment pins broadway-mlflow:latest; no repo build produces that tag (CI tags <sha>).
16. DEP-M6 — postgres-deployment pins broadway-postgres:latest; no build source exists.
17. DEP-M7 — docker-compose builds ./docker/postgres which holds only init.sql — BROKEN context, no Dockerfile.
18. DEP-M8 — train-job writes artifacts/ cwd-relative with NO PVC — no durable storage.
19. DEP-M9 — root Dockerfile broadway:latest consumed by api+train but never built by CI — UNVALIDATED image.

## FINDING REGISTER — DET-LEDGER

20. DET-C1 (context) — ≥3 independent RNG families CONFIRMED (YAML random_state / sample-spec seed / hardcoded literals incl. more_modeling batch): byte-stability is real where seeded; hazard is the literal class.
21. DET-C2 — uv.lock dual-numpy CONFIRMED (2.3.5 darwin+x86_64 markers AND 2.5.2 negation-markers): cross-platform float divergence guaranteed while golden tests keep exact ==.
22. DET-C3 — shapiro subsample seeded YES but by literal (dedupe: SAME finding as B#18 GATE-STATS-46 — B owns the config-key ruling; cross-ref only here).
23. DET-a — assumptions.py seed literal 0 absent from every YAML/schema/artifact; no index-pinning test. [cross-ref B#18/#22]
24. DET-b — onboard/module.py literal 42 bypasses "YAML single source of truth" rule. [doctrine root shared with C#19]
25. DET-c — qq.py mutates GLOBAL legacy numpy RNG state (process-global side effect).
26. DET-d — wall-clock bytes embedded in artifacts; NO freeze flag exists anywhere.
27. DET-e — Golden-float ULP unfixed: test_ml_pipeline bare == fails on foreign numpy/libm at 16th digit. [sibling of B#17 absence-claim; D owns the exact-equality fragility fix ruling]
28. DET-f — experiment plot scripts unseeded / hardcoded literals instead of config.
29. DET-g+h — trainer timing persist-path UNVERIFIED; structural.py unique() warning examples assume input row order (order-sensitivity undocumented).

## FINDING REGISTER — PERF-BASELINE

30. PERF-P1..P5 — (P1) D23 recorder ABSENT-confirmed: M1–M4 recorded nowhere; COLDPROBE-1 "mandatory" never built; STATE.md ## telemetry section must be created. (P2) same logical data crosses disk ≥8× per full prediction run (training_data ×3, features ×3, ~5 large writes). (P3) contracts+baseline = two redundant full O(N) reads near-zero compute — cheapest fix on the board. (P4) trainer.py time.time() persist-path unverified (=DET-g; one ruling surface). (P5) timed-baseline lane spec exists but ALL machinery unbuilt (M1–M4, Actions-API derivation, rolling window, gating conditions).

Dedupe decisions: #22≡B#18 single finding (B owns). #27 sibling-not-equal to B#17 (absence vs fragility — separate rulings, cross-linked). #29g ≡ #P4 same fact, one ruling. F-2 root ≡ C#21 (C owns resolver fix; deployment rows stay here). log_dataset second-read backlog (FIXES.md:53-54/D9) lands HERE as P2 evidence rather than a separate B row. Secret-audit S1/S2 human calls are NOT deferrable silently — see reminders.

## RULING FORMAT (one block per finding — mandatory shape)

```
FINDING: <register id> — <one-line restatement>
VERDICT: ADOPT | MODIFY(to: <concrete end-state>) | REJECT(with: replacement/do-nothing rationale) | HUMAN-CALL(owner: human, ask: <exact ask>)
root: <deepest cause whose repair kills the CLASS — mandatory non-empty>
rationale: <why; name strongest alternative considered and why it loses>
now-fix (ADOPT/MODIFY only): files=[...] · changed-lines≤N · acceptance=<exact command>
```

## SLATE-vs-DEFER SEPARATION

- SLATE = concrete-now-fix fits THIS packet's window: **≤4 files, ≤60 changed lines total,
  reversible, test-first preferred**. Slate items listed explicitly under `## SLATE`.
- DEFER = everything else → `DEFER → board-row: <proposed row id/scope>` mapped to CHANGE
  BOARD (#5). Slated-with-diffs or deferred-with-mapping; no middle state.

Slate candidates screened (senior still rules):
- #6: tripwire probe asserting configs/environment/development.yaml carries no literal password
  (regex probe in new tests/test_config_secrets_probe.py ≈15 lines),
  acceptance `uv run pytest tests/test_config_secrets_probe.py -q`.
  The credential-value CHANGE itself is HUMAN-CALL, not slateable.
- #7: README fresh-clone bootstrap note for the lookup symlink ≈5 lines doc fix.
- #28: pytest.approx conversion of test_ml_pipeline golden block ≈10 lines (backlog row already
  says land when project/ next touched — project/tests are R2-lane WIP; if custody conflicts ⇒
  DEFER with board mapping).
Everything k8s/docker-shaped (#8–#19) exceeds window ⇒ DEFER rows; recorder build (#P1/P5)
⇒ DEFER as its own lane (spec already written into the factsheet).

## MANDATE REMINDERS (non-negotiable)

- `root:` lines are MANDATORY on every block (MAIN_AGENT_CONTRACT §14 ROOT-CAUSE MANDATE).
- Lukewarm rulings FAIL the mandate — commit to a verdict class and defend it, or ESCALATE
  explicitly per SENIOR vocabulary.
- Human-only calls MUST be flagged `HUMAN-CALL` rather than deferred silently. THIS PACKET
  CARRIES THE SESSION'S PRIME EXAMPLES: upstream credential ROTATION (#1) and cross-line
  propagation ordering (#2) are human decisions; a DEFER row pointing at them is a FAILURE of
  the mandate even though the underlying work is also board-tracked.
