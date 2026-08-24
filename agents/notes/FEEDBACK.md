Good pushback on myself — no, I was too quick to sign off. Here are the five that actually matter:

**1. "Trust" and "monitoring" are marked ✅ in the alignment table but are only stubs in the build order.**
The gap table calls these High severity, the resolution section gives them full module breakdowns (drift.py, leakage.py, fairness.py, etc.), and then step 17 quietly reduces all of it to "stubs" alongside `selection/` and `unsupervised/`. The table's checkmarks say "resolved," the implementation order says "not actually built yet." That's a real discrepancy, not a minor one — if you show this repo to someone and they check `trust/drift.py`, it'll be empty. Pick one: either mark these honestly as "structurally reserved, not implemented" in the table, or move them earlier in the build order so the checkmark is true.

**2. Decisioning & Impact is the pillar stakeholders care about most, and it's the one being deferred.**
This is the "so what" pillar — the one that answers the question every stakeholder actually asks. Everything upstream of it (EDA, stats, experiments, trust) is groundwork; decisioning is the payoff. Leaving it at just `should_promote` and pushing optimization/cost-benefit indefinitely is fine for an engineering sprint, but risky for a platform whose stated purpose is presenting *to stakeholders*. I'd reconsider whether this should really be last-priority.

**3. There's no actual stakeholder-facing surface — this is a pipeline repo, not a platform.**
You call it a "project platform... used to showcase," but everything a stakeholder would see is a static HTML file in `artifacts/reports/`. There's no index across projects, no dashboard, no way to browse or compare runs. If "platform" is meant literally, this is the biggest structural gap in the whole plan and none of the six pillars caught it, because the pillars are about DS methodology, not about the delivery mechanism.

**4. `experiments/` is called the critical gap, then made optional and built 14th of 21 steps.**
If it only runs when a config has a design section, and it's built after training/evaluate rather than alongside foundational modules, there's a real risk it just never shows up in an actual demo run. "Critical gap, fixed" should probably mean "on by default in the reference pipeline," not "available if invoked."

**5. Tests land at step 20, almost last — after the statistically riskiest code is already written.**
`experiments/design.py` (power calc), `experiments/multiple.py` (correction methods), and `trust/drift.py` (PSI/KS tests) are exactly the modules where a subtle bug is invisible until someone acts on a wrong p-value. Writing these without tests until the very end is the opposite of where you want test discipline concentrated.

If I had to rank these, #1 and #5 are the ones I'd fix before writing any code — they're internal inconsistencies in the plan itself, not judgment calls.








For the `experiments/` module you scaffolded, here's what I'd actually reach for, mapped to the submodules you already defined:

**`design.py` — power analysis, sample size, MDE**
- **`statsmodels.stats.power`** — `TTestIndPower`, `NormalIndPower`. Covers the standard two-sample power/MDE calculations and is already a dependency if you're using `statsmodels` in `stats/regression.py`.
- **`scipy.stats`** — for the underlying distributions when you need something statsmodels doesn't expose directly.

**`assignment.py` — randomization, stratification, blocking**
- No heavyweight library needed here — this is mostly `numpy`/`pandas` (hash-based bucketing, `sklearn.model_selection.StratifiedKFold`-style logic repurposed for stratified assignment). If you want deterministic hash-based bucketing for consistent user assignment across sessions, that's typically hand-rolled (e.g., `hashlib.md5(user_id + salt)` → bucket).

**`analysis.py` — t-test, ANOVA, chi-square**
- **`scipy.stats`** — `ttest_ind`, `chi2_contingency`, `f_oneway`. You already have Welch's/Kruskal-Wallis in `stats/anova.py`, so this can share that code rather than duplicating it.
- **`pingouin`** — worth considering as an upgrade over raw scipy; it returns effect sizes and confidence intervals alongside p-values by default, which matters a lot for stakeholder-facing output (a p-value alone is a poor deliverable).

**`multiple.py` — corrections**
- **`statsmodels.stats.multitest`** — `multipletests()` gives you Bonferroni, Benjamini-Hochberg (FDR), Holm, and others through one function with a `method` argument. This is the standard choice; I wouldn't look elsewhere.

**`sequential.py` — sequential monitoring, early stopping**
- This is the one place off-the-shelf Python libraries genuinely thin out. Options:
  - **`sequential`** (small niche package) or hand-implementing **mSPRT / always-valid inference** (Johari et al.) — this is what most industry A/B platforms actually use, but you'll likely need to implement it yourself; there isn't a dominant, well-maintained Python package here the way there is for the other submodules.
  - If you want a ready-made framework instead of building primitives: **`confidence`** (by Zalando) or **`expan`** (Zalando's older one, less maintained) implement sequential-testing-adjacent logic, worth evaluating before you build from scratch.

**`hte.py` — heterogeneous treatment effects, uplift**
- **`econml`** (Microsoft) — the strongest general-purpose library for CATE/HTE estimation (double ML, causal forests, meta-learners like T/S/X-learner). This would likely become your main dependency for this file.
- **`causalml`** (Uber) — the other major option, particularly strong on uplift modeling specifically; some teams use it instead of or alongside econml.
- **`dowhy`** (Microsoft) — more for causal graph specification and identification/refutation than HTE estimation itself, but pairs well with econml if you want to formalize assumptions before estimating effects.

**One structural note:** since your `stats/` module already has a Spark-backed `StatsContext` for sampling and joins, check whether `econml`/`causalml` play well with your data volumes — they're generally numpy/pandas-native and will expect data pulled into memory, so for large experiment datasets you may need a sampling or aggregation step between Spark and the HTE estimators rather than feeding them directly.