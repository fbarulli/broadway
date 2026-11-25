# LOGGING.md — process logging (SSOT)

Owner: observability / logging law.

Boundaries: `PROFILING.md` owns timing metrics, `OPTIMIZATION.md` owns performance
changes, `LIVE_VERIFICATION.md` owns acceptance evidence. This file owns runtime
logs only. The harness rule "single logging per process, one name always" is
imported by reference and enforced here.

## 1. One process, one setup, one file

- Every process entry point (CLI `ds-pipeline`, pipeline runner, API server, worker)
  calls **one** `setup_logging` exactly once, at startup.
- Library and step modules only obtain `logging.getLogger(__name__)`. They never
  configure handlers, never call `basicConfig`, and never set levels.
- The root has one console handler and one file handler. No duplicate handlers, no
  double formatting, no propagation duplicates.
- The log file name comes from config and is **stable across runs** (never a
  per-run or timestamped filename). Rotation and retention are config values.
- The log path is config-driven; a hardcoded path or filename in code is a defect.
- The CLI is repo-bound: every command resolves `configs/environment/<name>.yaml`
  before dispatch. Bootstrap commands (`init`, `discover`, `columns`) run from a
  checkout that ships that tree; a bare directory is not a supported surface.

## 2. Level from config

- `environment.log_level` is the level SSOT. A hardcoded `INFO` in an entry point is
  a defect even when it matches today's value.
- Per-module overrides are config values, not code edits.
- Production defaults to `WARNING`, development/staging to `INFO`; the environment
  YAML owns the difference.

## 3. What to log (strategic only)

Log at boundaries, decisions, and outcomes:

- step start/end: step name, dataset, environment, row count, artifact path,
  duration;
- decisions and fallbacks with their reason (a fail-soft path without a recorded
  reason is a defect);
- retries, skips, empty results, and why;
- errors with the full traceback (`logger.exception`), never a one-line summary.

Never log inside a loop, per row, per feature, or per batch. Never log secrets,
credentials, or row-level personal data. A log that repeats per iteration is noise,
not observability.

## 4. `print` versus log

- `print` is allowed only for **user-facing command output** in CLI/entry modules
  (tables, report text, prompts). It is never diagnostics.
- Library code logs; it does not print.
- A new `print` outside the user-output allowlist is a defect. The allowlist is
  config, not scattered `# noqa` comments.

## 5. Run correlation

Every invocation gets one `run_id`. The same `run_id` and the same step names appear
in:

- the log lines,
- the profile record (`PROFILING.md`),
- the evidence pack (`LIVE_VERIFICATION.md`),
- the MLflow run and its artifacts.

Without the shared `run_id`, a slow step cannot be tied to the artifact it produced
or the profile that measured it.

## 6. Errors

- Full traceback via `logger.exception` / `exc_info=True`; never truncated.
- A caught error that is handled, downgraded, or recovered records why.
- Silence is not a log level: an empty outcome with no recorded reason masquerades
  as success.

## 7. Enforcement

- Gate: no `basicConfig` or handler construction under `src/broadway/` outside
  `observability/logging.py`; the setup is idempotent (a second call adds no
  handler). Script tooling under `scripts/` is out of this gate's scope.
- Gate: the effective level equals `environment.log_level`.
- Gate: `print` outside the configured user-output allowlist fails.
- Reviewer checks run correlation: the log, profile, and evidence for one invocation
  carry the same `run_id` and step names.
