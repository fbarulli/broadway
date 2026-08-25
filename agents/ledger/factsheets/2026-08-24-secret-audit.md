# FACT SHEET — SECRET-AUDIT — 2026-08-24

Investigator: read-only lane SECRET-AUDIT @ HEAD 5016e93 (sklearn).
Credential values masked (length + 3-char prefix only).

## Headline: SECRET-1 IS STILL OPEN (partially remediated)

1. Removal incomplete: mandated "remove" landed only on sklearn lineage
   (fix commit `8b73645`, contained by sklearn + origin/sklearn +
   origin/taxi ONLY). The OLD credential (len=24, pfx `Toz`,
   sha256pfx `3865079c5ad65ce7`) sits at HEAD of PUBLIC refs
   `origin/main` (1860709) and `origin/taxi_work` (5c6370f), plus local
   main/taxi/taxi_work/pr-1/pr-2/sp_probe and tag tier-1-complete.
2. Rotation NOT-EVIDENCED: no SECRET-1 row in ANY committed FIXES.md;
   commit `8b73645` message claims "Ledger: FIXES.md" but no row exists
   (bookkeeping gap). Only doctrine traces: DECISIONS.md:265 rider.
3. Path discrepancy: D22 rider's "configs/secret.yaml" NEVER existed in
   any ref — real subject is `k8s/optuna/secret.yaml`.
4. Remediation mechanics CLOSED on sklearn lineage: template
   placeholders (`k8s/optuna/secret.yaml:1-5,13-16`); lifecycle.sh full
   chain (env vars → gitignored secret.local.yaml → openssl-generated
   fallback persisted once mode 600, lifecycle.sh:27-44,61-82);
   .gitignore:36-38; zero old-value references in tracked worktree.

## Supply-chain sweep

| Area | Verdict | Evidence |
|---|---|---|
| .env vs .env.example drift | ⚠ both directions | .env-only keys: DEEPSEEK_API_KEY(pfx sk-,36), NVIDIA_API_KEY(nva,71), OPENROUTER_API_KEY(sk-,74), ROUTER_API_KEY(sk-,36), ORCA_API_KEY(sk-,52) — .env gitignored; example-only: MLFLOW_TRACKING_URI + DATABASE_* |
| k8s/** literals | ✅ CLEAN | postgres/mlflow mount Secret files (k8s/optuna/postgres.yaml:43-45) |
| docker/** baked creds | ✅ CLEAN | docker-compose.yml:20-22 ${DATABASE_*} interpolation |
| configs/environment | ⚠ dev-only literal creds committed | development.yaml database_user/pw len=8 pfx=pos (differs from pushed value); staging/production interpolate ${…} |
| Out-of-repo lookup path | ⚠ FOUND | data/raw/taxi_zone_lookup.csv → symlink OUTSIDE repo (/home/opc/ONE/learning/data/raw/…); consumed as plain path (configs/dataset/taxi.yaml:33,47; project/data or scripts/02_join_boroughs.py:2); fresh-clone bootstrap NOT documented |
| TLC schema pins | 📌 pinned | configs/dataset/taxi.yaml:1-29 (7 cols, dtypes, null_count:0); enforced project/etl/process.py:125-128 + structural.py dtype gates |

## Human calls required

(i) ROTATE the DB credential upstream; record evidence row (rotation is
what kills value per D22 — history rewrite rejected there).
(ii) Propagate template fix across lines via the D24 single
reconciliation pass (covers taxi; main frozen until main-day — exposure
window documented until then).
(iii) Backfill the missing FIXES.md SECRET-1 row (correcting the false
"Ledger:" citation inside 8b73645's message — noted, message itself
immutable).
