That's the right instinct, and the call-site count backs it up. I went and counted: the shared `estimation_table` in `robust.py` has exactly **one production call site** (a one-line delegation in `_ols_bp.py`) plus one test file. That's not a pattern spread across a codebase — it's a single choke point. Building a `RobustFit` dataclass, threading it through signatures, for one call site is disproportionate.

Lighter version: **assert at the one entry point, don't wrap the type.** Statsmodels already tags this — `get_robustcov_results()` sets `.cov_type` on the result object, so the check is a single guard clause, not a new class:

```python
def estimation_table(model, alpha: float = 0.05) -> pd.DataFrame:
    cov_type = getattr(model, "cov_type", "nonrobust")
    if cov_type != "HC3":
        raise ValueError(f"estimation_table expects an HC3-fit model, got cov_type={cov_type!r}")
    ...
```

Same shape fixes T-bug-1 with zero new types, zero signature changes elsewhere, and it still turns "silently mislabeled" into "loud at the boundary" — which was the actual goal, not the wrapper.

That reframes the whole scope-check outcome, actually: instead of asking "wrapper type vs. extend pandera," the sharper question per-finding is **"how many real callers does this boundary have?"**

- **1 caller (T-bug-1)** → guard clause at the function, done today, ships with Tier 1.
- **Handful of callers, already contract-shaped (F1, F4)** → belongs in the existing `contracts/pandera.py` path, as I said — extend, don't parallel.
- **Scattered across many files with no shared entry point (F8/F11/F14/F16's dtype checks across those 33 files)** → this is the one case where "draw a boundary" *is* the fix, in the literal sense: the real problem isn't missing types, it's that there's no single function these 33 call sites go through at all. The fix is consolidating them behind one classifier function first — after that, whether it's a guard clause, an enum, or a pandera check is a five-minute decision, because there'd only be one place to make it.

So: no new pattern, no parallel proof system, nothing added to Tier 1's unblocked status. The only real work item this generates is Tier 3's dtype-harmonization findings absorbing a "consolidate call sites into one function" step before the fix itself — which they arguably needed anyway to fix F8/F11/F14/F16 consistently rather than four separate patches. Want that folded into Tier 3 as written, or split out as its own small item?