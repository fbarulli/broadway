Here's the consolidated plan — merging your original scoping instincts with the valid corrections from the critique, structured so it's genuinely reusable across datasets without becoming a multi-month infrastructure project.

## Guiding principle

**Diagnostics generate evidence → rules propose a plan → you (or a light router) decide → execution is transparent → decision trail is preserved.**
Automate the mechanics. Keep the judgment calls visible, not hidden — but *informed by more than a single p-value.*

## Repo structure

```
project/
├── config/
│   └── dataset.yaml          # target, group_col, features, numeric/categorical split, alpha
├── data/
│   └── 00_prepare_data.py    # load → join → sample → results/joined_sample.parquet (cached, seeded)
├── lib/
│   ├── quality.py            # data quality gate
│   ├── group_tests.py        # ANOVA family + routing
│   ├── ols_diagnostics.py    # OLS assumption checks
│   ├── remediation.py        # transform / robust SE / nonlinear escalation
│   ├── effect_size.py        # eta², omega², Cohen's d, CIs
│   └── plan.py                # AnalysisPlan dataclass + executor
├── scripts/                  # your existing 01-12, now calling lib/
├── results/                  # gitignored: parquet cache, plans, fitted models, reports
└── report.py                 # renders results/*.json → markdown/report
```

## Core data structure: the Analysis Plan

Every routing decision returns this, not a bare string:

```python
@dataclass
class AnalysisPlan:
    analysis_type: str              # "group_comparison" | "regression"
    selected_test: str
    diagnostics: dict               # every test run + its p-value/statistic
    effect_size: dict                # type + value + CI
    reason: list[str]                # human-readable evidence trail
    posthoc: dict | None = None
    warnings: list[str] = field(default_factory=list)  # e.g. "underpowered", "n imbalance"
```

This is what gets logged to `results/` and rendered in the report — the audit trail survives regardless of dataset.

## Phased build (stop wherever fits your time budget)

**Phase 1 — Foundational correctness (build this now)**
1. **Data quality gate** (`lib/quality.py`): missingness, group-size imbalance, constant columns, minimum-n check. Hard gate before any test runs — cheap, prevents nonsense results on any future dataset.
2. **Effect size + CI everywhere** (`lib/effect_size.py`): every test function returns effect size alongside p-value. Fixes the large-n-makes-everything-significant problem, which will bite you on any big dataset, not just this one.
3. **AnalysisPlan structure** (`lib/plan.py`): architecture-only change, no new statistics. Router functions return a plan, not a string.

**Phase 2 — Better routing (do this next)**
4. **Group-comparison router**, revised per the critique:
   - Levene's test → default to **Welch** when variances/sizes look questionable (Welch's cost is low, so bias toward it rather than a hard Shapiro gate)
   - Non-normality alone does **not** auto-route to Kruskal-Wallis — combine with sample size and skew/tail metrics; note it as a flag, not a hard trigger
   - Omnibus p < α **and** effect size non-trivial → Games-Howell post-hoc
5. **OLS remediation ladder**, revised: assumption failure → try transform → robust SE → interaction/nonlinear terms, in that order, *before* ever considering escalation to a tree model. Track pass/fail at each rung.
6. **Escalation gate to LGBM**, reframed correctly: escalate when the *remediation ladder is exhausted and residual structure still shows nonlinearity* (e.g., partial residual plots, remaining heteroskedasticity after HC3 + log), not merely "OLS failed 2 diagnostics." Log the specific evidence.

**Phase 3 — Robustness (nice to have, do if time allows)**
7. **Influence diagnostics**: Cook's distance / leverage on OLS; report result with vs without influential points, flag if conclusion is stable.
8. **Minimal sensitivity check**: run 2-3 reasonable specs (raw, log, +HC3) and report whether direction/significance holds across them — you're already doing this in 08/09, just formalize the comparison table.

**Explicitly out of scope for this portfolio project** (correct in general, wrong ROI here): multiple-testing correction (FDR/Holm — only matters once you're testing many hypotheses at once, which you're not), causal-vs-predictive gating, power analysis, clustered/panel SE detection. Note these in a `LIMITATIONS.md` so the portfolio shows you know the boundary of what you built rather than pretending the system does more than it does.

## What stays manual, by design

Choosing α, choosing *which* candidate remediations belong in the ladder (log vs Box-Cox), and any causal language — these remain human calls. The system's job is to make the evidence for those calls explicit and reproducible, not to remove the human.

Want me to write `lib/quality.py` and `lib/effect_size.py` first, since those are Phase 1 and touch every downstream script?