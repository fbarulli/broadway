# Principal analysis

## Question

Do the group means differ?

## What was run

welch

## What it found

- method: welch
- statistic: 7e+03
- p_value: < 0.001
- passed: yes
- eta_squared: 0.982
- omega_squared: 0.123

## Why it matters

reject H0: at least one group mean differs (p=< 0.001) eta² = 0.982 (proportion of outcome variance explained by group membership; can be inflated under extreme imbalance); omega² = 0.123 (corrects for small-sample bias; the more conservative estimate). eta² and omega² diverge here because group sizes are extremely imbalanced; report omega².

## Effect size

- eta² = 0.982: proportion of outcome variance explained by group membership (can be inflated under extreme imbalance)
- omega² = 0.123: corrects for small-sample bias; the more conservative estimate
