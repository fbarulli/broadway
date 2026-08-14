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

- ramification: group sizes are imbalanced (imbalance ratio 2.16e+03).
- result_summary:
  - imbalance_ratio: 2.16e+03
  - absent_groups: 0
  - n_total: 200130
  - n_used: 199420
  - n_excluded: 710
  - exclusion_reason: unlisted group (values outside the configured group list; see audit)
  - N used 199420 of 200130 (710 excluded: unlisted group (values outside the configured group list; see audit)); see audit
![How to read: the top panel is a boxplot of the target by group, each box spanning the interquartile range with a line at the median; the bottom panel shows the number of observations per group; very unequal bars indicate imbalance.](figures/describe.png)

## Normality diagnostics

- ramification: distributional shape is skewed/heavy-tailed in some groups; consider this alongside sample size before choosing a method.
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
![How to read: one plot per group; each group's standardized values are plotted against theoretical normal quantiles; points following the fitted reference line are approximately normal; curvature or heavy tails indicate non-normality.](figures/normality_qq.png)

## Variance homogeneity

- ramification: variance evidence favors considering Welch's ANOVA (or a rank-based alternative) over standard ANOVA.
- result_summary:
  - Levene statistic: 4.2e+03
  - p_value: < 0.001

## Principal analysis

- ramification: reject H0: at least one group mean differs (p=< 0.001) eta² = 0.982 (proportion of outcome variance explained by group membership; can be inflated under extreme imbalance); omega² = 0.123 (corrects for small-sample bias; the more conservative estimate). eta² and omega² diverge here because group sizes are extremely imbalanced; report omega².
- result_summary:
  - method: welch
  - F: 7e+03
  - p_value: < 0.001
  - passed: yes
  - eta_squared: 0.982
  - omega_squared: 0.123
  - eta² = 0.982: proportion of outcome variance explained by group membership (can be inflated under extreme imbalance)
  - omega² = 0.123: corrects for small-sample bias; the more conservative estimate

## Post-hoc comparisons

- ramification: Games-Howell found 17 significant pairwise difference(s) at alpha=0.05.
- result_summary:
  - method: games_howell
  - pairs: 21
  - significant_pairs: 17
  - 17 of 21 pairs significant:
    - Bronx vs Brooklyn: p < 0.001, Cohen's d 0.277, Hedges' g 0.277
    - Bronx vs EWR: p < 0.001, Cohen's d 0.556, Hedges' g 0.555
    - Bronx vs Manhattan: p < 0.001, Cohen's d 3, Hedges' g 3
    - Bronx vs Queens: p < 0.001, Cohen's d 0.455, Hedges' g 0.455
    - Bronx vs Staten Island: p < 0.001, Cohen's d 0.656, Hedges' g 0.655
    - Bronx vs Unknown: p < 0.001, Cohen's d 1.34, Hedges' g 1.34
    - Brooklyn vs Manhattan: p < 0.001, Cohen's d 2.19, Hedges' g 2.19
    - Brooklyn vs Staten Island: p < 0.001, Cohen's d 0.405, Hedges' g 0.405
    - Brooklyn vs Unknown: p < 0.001, Cohen's d 0.842, Hedges' g 0.841
    - EWR vs Manhattan: p < 0.001, Cohen's d 1.39, Hedges' g 1.39
    - EWR vs Unknown: p 0.001, Cohen's d 0.808, Hedges' g 0.807
    - Manhattan vs Queens: p < 0.001, Cohen's d -2.01, Hedges' g -2.01
    - Manhattan vs Staten Island: p < 0.001, Cohen's d -1.13, Hedges' g -1.13
    - Manhattan vs Unknown: p < 0.001, Cohen's d -0.218, Hedges' g -0.218
    - Queens vs Staten Island: p < 0.001, Cohen's d 0.601, Hedges' g 0.601
    - Queens vs Unknown: p < 0.001, Cohen's d 1.12, Hedges' g 1.12
    - Staten Island vs Unknown: p 0.004, Cohen's d 0.649, Hedges' g 0.648

## Conclusion

- ramification: group means differ (welch p=< 0.001), with 17 significant pairwise difference(s).
- result_summary:
  - verdict: group means differ (welch p=< 0.001), with 17 significant pairwise difference(s).
  - principal_method: welch
  - p_value: < 0.001
  - significant_pairs: 17
  - eta_squared: 0.982
  - omega_squared: 0.123
  - eta² = 0.982: proportion of outcome variance explained by group membership (can be inflated under extreme imbalance)
  - omega² = 0.123: corrects for small-sample bias; the more conservative estimate