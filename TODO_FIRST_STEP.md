| Question                                                            | What we're trying to learn                                                           | If  problem exists                                  

| **1. Is the mean relationship correctly specified?**                | Is OLS capturing the underlying relationship between X and Y?                        | Transform variables, add nonlinear terms/splines, or change model                   |
| **2. Are the observations independent?**                            | Are we treating correlated observations as if they're independent?                   | Clustered SE, mixed model, GEE, time-series/panel approach                          |
| **3. Is the error variance constant?**                              | Are conventional SEs, p-values and CIs reliable?                                     | Use HC3/robust SE; potentially model variance                                       |
| **4. Is the result being driven by a few observations?**            | Is the model brittle/influenced by outliers or high-leverage points?                 | Sensitivity analysis, investigate observations, possibly robust regression          |
| **5. Is residual non-normality problematic for inference?**         | Particularly with small samples, can normal-theory inference be trusted?             | Transformation, bootstrap/permutation, robust/nonparametric inference               |
| **6. Is there problematic multicollinearity?**                      | Are coefficient estimates unstable because predictors contain redundant information? | Re-specify, combine variables, regularize, or retain with warning                   |
| **7. Is the sample sufficient for the inference we're attempting?** | Do we have enough information to detect a meaningful effect?                         | Power/MDE warning, collect more data, don't interpret null as evidence of no effect |
8. Is the estimated effect confounded?      Is X–Y association plausibly explained by omitted common causes?                                    Identify appropriate pre-treatment confounders; adjust/re-specify; consider design-based methods; avoid adjusting for mediators/colliders


Each diagnostic should answer: Question → Evidence → Ramification

For example:

Is error variance constant?
BP/White + residual plot → heteroskedasticity detected → don't trust conventional SEs → use HC3.

Whereas:

Is functional form correct?
Residual pattern → nonlinear structure detected → don't immediately switch to ML → try an appropriate functional-form remediation.

So the first step isn't really "test whether OLS assumptions pass."

It's:

"What could make our estimated effect or its uncertainty unreliable, and what should we do about each failure?"

That should be the foundation of the router.



Step 1: Add confounding / omitted-variable bias
Question:
Is the estimated relationship between X and Y plausibly attributable to X, rather than to variables related to both X and Y?


What to examine:

A causal/subject-matter DAG or explicit causal assumptions
Important pre-treatment covariates
Whether X is associated with known predictors of Y
Whether plausible confounders were omitted
Whether adjustment variables are actually post-treatment variables or colliders

Evidence → Ramification:

Covariate structure / DAG → plausible confounding remains → the OLS coefficient may not identify the intended effect → adjust for appropriate pre-treatment confounders or explicitly frame the estimate as associational.

That gives you a new category: not "did an OLS assumption fail?" but "does the coefficient answer the question we think it answers?"


++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


1- Is the mean relationship correctly specified?
        after fitting OLS, are the residuals vs fitted value, do the model's systematic errors show a pattern.

Do the residuals look randomly scattered around zero, or do they contain systematic structure?

Random cloud around 0                   = Functional form is reasonably specified
U-shape / ∩-shape                       = Missing quadratic/nonlinear relationship
S-shape	                                = More complex nonlinear relationship
Clear trend	                            = Mean relationship is misspecified
Separate curves/groups                  = Missing interaction or group-specific relationship
Systematic pattern against a predictor  = Relationship with that predictor may be incorrectly specified


The important point is that this is not primarily a test of whether residuals are "good." You're looking for remaining predictable structure.

Is the mean relationship correctly specified?
Residual-vs-predictor/fitted plots show systematic structure → OLS's assumed mean function is inadequate → modify the functional form (e.g., polynomial term, transformation, spline, interaction) and refit.

Don't use a significant p-value from a nonlinear-term test as the sole diagnostic. The residual plot tells you what kind of misspecification exists and whether the proposed remediation makes substantive sense.