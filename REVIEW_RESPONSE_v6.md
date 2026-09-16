# Response to methodological review — BBB v6

This revision addresses the issues identified in the review of the earlier
brain-biodistribution machine-learning pipeline.

1. **Selection and assessment are separated.** The primary result now estimates
   the performance of the complete selection procedure. In each outer
   `GroupKFold`, feature set, algorithm, target transform, and hyperparameters
   are selected exclusively by inner grouped cross-validation. Outer held-out
   studies are used only once for final assessment.

2. **R² is interpreted by magnitude.** The revised pipeline reports pooled outer
   out-of-fold R² with a study-cluster bootstrap confidence interval. The result
   is described as weak or non-generalizing when appropriate; a merely positive
   value is not treated as evidence of useful prediction.

3. **Duplicate Q² is removed.** Under the former definition, Q² was identical to
   pooled out-of-fold R². It is therefore omitted and this equivalence is stated.

4. **Hyperparameters are tuned independently of final assessment.** Tuning is
   nested inside each outer training partition. The primary outer test data do
   not influence the ranking or chosen configuration.

5. **Uncertainty is reported.** MAE, RMSE, R², study-macro errors, and correlation
   metrics receive 95% confidence intervals from a cluster bootstrap in which
   `Study_Group`, not the individual row, is resampled.

6. **Differences between procedures are formally assessed.** Family-specific
   nested procedures use identical outer splits. Their paired study-level MAE
   differences receive bootstrap confidence intervals, paired sign-flip
   permutation tests, and Holm multiplicity correction.

7. **Spearman inference is group-aware.** The inferential analysis aggregates
   observed and outer-predicted values to one mean per `Study_Group` and obtains
   a two-sided permutation p-value across study units. Observation-level
   Spearman correlation is descriptive only.

8. **Zero values were traced to their source.** The seven zeros all originate
   from DOI `10.18869/acadpub.ijrr.18.3.539`. Its Table 1 explicitly reports brain
   values as `0.0 ± 0.0 %ID/g`, but does not give an organ-specific LOD/LOQ. The
   primary analysis retains them as reported rounded zeros without claiming
   proven absence of uptake. A sensitivity analysis excludes the entire source
   study.

9. **Feature Set C is correctly labelled.** It is explicitly a combined model of
   nanoparticle descriptors and experimental/biological/measurement context.
   Differences from A and B are not attributed solely to nanoparticle properties.

10. **Material–study dependence is audited.** Every study in the current dataset
    contains exactly one material. The report therefore warns that `Material` is
    a study-level characteristic and may proxy study origin. A prespecified
    full-context feature set without `Material` is included.

11. **All-data SHAP is removed.** Interpretability now uses permutation
    importance evaluated only on outer held-out studies, with stability summaries
    across outer folds.

12. **Warnings remain visible.** No global warning suppression is used. Constant
    inputs to Spearman correlation are handled explicitly as undefined rather
    than silenced.

13. **The environment is recorded.** Exact Python, operating-system, pandas,
    NumPy, SciPy, scikit-learn, matplotlib, openpyxl, and joblib versions are
    written to `environment_versions.json` for every run.

## Main scientific result from the verified run

- Study-macro MAE: **0.258994 %ID/g** (95% group-bootstrap CI:
  **0.187921–0.350551**)
- Observation-weighted RMSE: **0.463233 %ID/g** (95% CI:
  **0.280537–0.652578**)
- Pooled outer out-of-fold R²: **−0.001464** (95% CI:
  **−0.185856–0.051820**)
- Group-mean Spearman rho: **0.118003**, permutation **p = 0.502675**
- Dummy baseline study-macro MAE: **0.258669 %ID/g**
- Minimum Holm-adjusted p-value across primary paired procedure comparisons:
  **1.000000**

The corrected analysis therefore does **not** support a claim of useful
generalization to unseen studies. This is reported as the substantive result,
not reframed as a successful predictive model.
