# Timeline — taxi_hypothesis

| # | Step | Question | Status |
| --- | --- | --- | --- |
| 1 | Describe groups | Do the groups contain enough observations? | completed with note |
| 2 | Normality diagnostics | Is the distributional shape problematic? | completed with note |
| 3 | Variance homogeneity | Is error/group variance homogeneous? | warning |
| 4 | Choose principal method | Which principal method should answer the question? | completed |
| 5 | Principal analysis | Do the group means differ? | completed with note |
| 6 | Choose post-hoc method | Which post-hoc comparison is appropriate? | completed |
| 7 | Post-hoc comparisons | Which specific group pairs differ? | completed |
| 8 | Conclusion | What is the conclusion? | completed |

## Describe groups

- ramification: group sizes are imbalanced (imbalance ratio 2162.6747).
- evidence_refs: describe.json
- result_summary:
  - total_n: 200130
  - imbalance_ratio: 2.16e+03
  - absent_groups: 0
  - n_total: 200130
  - n_used: 199420
  - n_excluded: 710
  - exclusion_reason: unlisted group
  - N used 199420 of 200130 (710 excluded: unlisted group); see audit
![How to read: each box spans the interquartile range with a line at the median; boxes at different heights indicate different group centers.](figures/describe_boxplot.png)
![How to read: bar height is the number of observations per group; very unequal bars indicate imbalance.](figures/describe_group_sizes.png)

## Normality diagnostics

- ramification: distributional shape is skewed/heavy-tailed in some groups; consider this alongside sample size before choosing a method.
- evidence_refs: normality.json
- result_summary:
  - Manhattan.skew: 2.61
  - Manhattan.kurtosis: 13
  - Manhattan.shapiro_p: < 0.001
  - Brooklyn.skew: 1.2
  - Brooklyn.kurtosis: 1.24
  - Brooklyn.shapiro_p: < 0.001
  - Queens.skew: 0.841
  - Queens.kurtosis: 1.52
  - Queens.shapiro_p: < 0.001
  - Bronx.skew: 1.14
  - Bronx.kurtosis: 1.22
  - Bronx.shapiro_p: < 0.001
  - Staten Island.skew: 1.02
  - Staten Island.kurtosis: 0.606
  - Staten Island.shapiro_p: < 0.001
  - standardization: per-group z-score
![How to read: each trace is one group's standardized values; traces hugging the diagonal are approximately normal; curvature or heavy tails indicate non-normality.](figures/normality_qq.png)

## Variance homogeneity

- ramification: variance evidence favors considering Welch's ANOVA (or a rank-based alternative) over standard ANOVA.
- evidence_refs: variance.json
- result_summary:
  - statistic: 4.2e+03
  - p_value: < 0.001

## Principal analysis

- ramification: reject H0: at least one group mean differs (p=< 0.001) eta² = 0.999 (proportion of outcome variance explained by group membership; can be inflated under extreme imbalance); omega² = 0.122 (corrects for small-sample bias; the more conservative estimate). eta² and omega² diverge here because group sizes are extremely imbalanced; report omega².
- evidence_refs: omnibus.json
- result_summary:
  - method: welch
  - statistic: 2.97e+05
  - p_value: < 0.001
  - passed: yes
  - eta_squared: 0.999
  - omega_squared: 0.122
  - eta² = 0.999: proportion of outcome variance explained by group membership (can be inflated under extreme imbalance)
  - omega² = 0.122: corrects for small-sample bias; the more conservative estimate

## Post-hoc comparisons

- ramification: Games-Howell found 19 significant pairwise difference(s) at alpha=0.05.
- evidence_refs: posthoc.json
- result_summary:
  - method: games_howell
  - pairs: 21
  - significant_pairs: 19
  - 19 of 21 pairs significant:
    - Bronx vs Brooklyn: p < 0.001, Cohen's d 0.278, Hedges' g 0.278
    - Bronx vs EWR: p < 0.001, Cohen's d 0.591, Hedges' g 0.59
    - Bronx vs Manhattan: p < 0.001, Cohen's d 3.07, Hedges' g 3.07
    - Bronx vs Queens: p < 0.001, Cohen's d 0.492, Hedges' g 0.492
    - Bronx vs Staten Island: p < 0.001, Cohen's d 0.682, Hedges' g 0.682
    - Bronx vs Unknown: p < 0.001, Cohen's d 1.42, Hedges' g 1.42
    - Brooklyn vs EWR: p 0.024, Cohen's d 0.32, Hedges' g 0.32
    - Brooklyn vs Manhattan: p < 0.001, Cohen's d 2.25, Hedges' g 2.25
    - Brooklyn vs Queens: p < 0.001, Cohen's d 0.0545, Hedges' g 0.0545
    - Brooklyn vs Staten Island: p < 0.001, Cohen's d 0.412, Hedges' g 0.412
    - Brooklyn vs Unknown: p < 0.001, Cohen's d 0.846, Hedges' g 0.846
    - EWR vs Manhattan: p < 0.001, Cohen's d 1.4, Hedges' g 1.4
    - EWR vs Unknown: p 0.001, Cohen's d 0.91, Hedges' g 0.91
    - Manhattan vs Queens: p < 0.001, Cohen's d -2.01, Hedges' g -2.01
    - Manhattan vs Staten Island: p < 0.001, Cohen's d -1.13, Hedges' g -1.13
    - Manhattan vs Unknown: p < 0.001, Cohen's d -0.229, Hedges' g -0.229
    - Queens vs Staten Island: p < 0.001, Cohen's d 0.597, Hedges' g 0.597
    - Queens vs Unknown: p < 0.001, Cohen's d 1.11, Hedges' g 1.11
    - Staten Island vs Unknown: p 0.003, Cohen's d 0.707, Hedges' g 0.707

## Conclusion

- ramification: group means differ (welch p=< 0.001), with 19 significant pairwise difference(s).
- evidence_refs: conclusion.json
- result_summary:
  - verdict: group means differ (welch p=< 0.001), with 19 significant pairwise difference(s).
  - principal_method: welch
  - p_value: < 0.001
  - effect_size: eta²=0.999, omega²=0.122
  - significant_pairs: 19