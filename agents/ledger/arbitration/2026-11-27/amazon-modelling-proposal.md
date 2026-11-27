# Amazon modelling chain — selection, HPO, holdout promotion

Status: **implemented and executed; independent adversarial review pending.**
Owner scope: Amazon only, target `Shopping_Satisfaction` (ordinal 1–5) treated
as interval-scale regression — no calibrated-ordinal or production claim.

## What was added

- `selection/module.py` + `candidates.py`: naive reference, linear, random
  forest, LightGBM on identical repeated shuffled KFold (5×5, seed 2026).
  `selection_rule` is config (`point` used here); the record also states what
  the `one_se` rule would pick (linear). Champion decision = paired row-level
  improvement over the naive mean on out-of-fold predictions
  (`MetricBootstrap.improvement_interval`); the champion-vs-linear interval is
  evidence in the same record.
- `selection/tuning.py` + `experiment/amazon_hpo.yaml`: 30 Optuna trials on
  search seed 7; the reported best-vs-default result is measured on a disjoint
  report seed 99 with the same row-level bootstrap.
- `training/module.py`: a no-HPO fit publishes its own train-vs-naive verdict on
  `train.cv_random_state` (4242) — not the selection/HPO partitions.
- `evaluate/module.py`: one promotion decision on the holdout; the reference is
  the registry champion when present, otherwise the naive training-mean
  predictor on the same rows. A first promotion registers the candidate.
- `scripts/check_experiment_outputs.py`: the enforced set requires a published,
  validated record; the step module is resolved through `STEP_MODULES` (so
  `hpo` → `selection/tuning.py` is checked). `causal` carries an explicit
  not-applicable ruling (no Amazon causal config), printed every run.
- Evidence paths: `project/experiments/results/pipeline/<step>/`; baseline
  `baseline.json` and `training_result.json` are tracked (evaluate consumes
  them), the per-row error ledger stays ignored.

## Published decisions (real run, n=481 train / 121 holdout)

| step | metric | value [95%] | floor | verdict |
|---|---|---|---|---|
| baseline | improvement_over_naive_mae | 0.2861 [0.1737, 0.3933] | 0.02 | act |
| selection | improvement_over_naive_rmse | 0.1834 [0.1318, 0.2371] | 0.02 | act |
| hpo | improvement_over_default_rmse | 0.0060 [-0.0019, 0.0137] | 0.01 | investigate |
| training | improvement_over_naive_rmse | 0.1892 [0.1366, 0.2433] | 0.02 | act |
| evaluate | improvement_rmse (vs naive holdout) | 0.2312 [0.1140, 0.3258] | 0.05 | act |

Champion: rf (0.8355 CV RMSE). Its lead over linear is 0.0093
[-0.0210, 0.0387] — not distinguishable; the one-SE rule would pick linear.
The HPO verdict is `investigate`, so no tuned experiment is shipped and the
promotion record compares the default champion only.

## Limits

- The folds within a repeat are correlated; the candidate CV intervals are
  fold-level (optimistic) while the paired improvements are row-level OOF.
- MLflow is a local external service; tracked records do not depend on it, but
  re-running train/evaluate does.
- No independent adversarial review was performed in this session.
