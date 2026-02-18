# Task 10 Assumptions — Elastic Net Regression + Model Card

## Feature Handling

1. **NaN imputation for COT features.** COT columns are structurally NULL for
   non-futures (ETFs, equities). Since Elastic Net cannot handle NaN, these are
   filled with 0.0 before scaling. After StandardScaler, 0.0 maps to the feature
   mean, making the imputation neutral. The cluster fixed effects absorb any
   structural difference between futures and non-futures, so the COT coefficient
   reflects within-futures variation only.

2. **MTFA flag NaN handling.** If WeeklyTrendAligned or MonthlyTrendAligned is None
   (HTF data was not provided), it is filled with 0.0 (same neutral imputation as
   COT). This is a temporary measure — in production, HTF data should always be
   available.

3. **Boolean features** (CleanPullback, WR_Divergence, WeeklyTrendAligned,
   MonthlyTrendAligned, VIX_Regime) are cast to float (0.0 / 1.0) before scaling.

4. **AssetCluster one-hot encoding.** Cluster is one-hot encoded via
   `pd.get_dummies`. One cluster is dropped (first alphabetically) to avoid
   multicollinearity with the intercept. The cluster dummy columns are appended
   after the 9 candidate features and are NOT standardized (they are already 0/1).

## Scaling

5. **StandardScaler on candidate features only.** The 9 candidate features are
   scaled (zero mean, unit variance). Cluster dummies are not scaled. This matches
   the pre-registration: "StandardScaler in Pipeline (before regression)".

6. **Scaler fit on full IS data.** The scaler sees all IS observations (minor
   within-IS leakage for alpha selection folds, but no IS→OOS leakage). The same
   scaler object must be used for OOS scoring (Task 13).

## Cross-Validation for Alpha Selection

7. **PurgedTimeSeriesSplit splits passed to ElasticNetCV.** The splits are
   pre-computed from trigger dates and passed as a list of (train, test) index
   tuples. This ensures the same purge-gap logic used everywhere else.

8. **l1_ratio grid.** ElasticNetCV searches over l1_ratio = [0.1, 0.5, 0.7, 0.9,
   0.95, 0.99, 1.0]. This spans the Elastic Net mixing parameter from mostly
   Ridge (0.1) to pure Lasso (1.0).

## Tercile Assignment

9. **IS tercile cutoffs.** Trades are ranked by predicted score and split into
   three equal groups. Remainder is assigned to the top group (same convention
   as ranking_gate.py). Cutoffs are saved for later OOS application.

## Diagnostics

10. **Correlation threshold.** Feature pairs with |Pearson r| > 0.70 are flagged
    as potentially multicollinear. This is informational — the Elastic Net handles
    multicollinearity via regularization. No features are dropped automatically.

11. **Sign check.** Each coefficient is compared to its pre-registered expected
    sign. Zeroed coefficients (Elastic Net set to exactly 0.0) are reported
    separately. Two or more sign flips triggers a PAUSE per kill criteria.

## Scope Limits

12. **No OOS scoring in this task.** The model is trained and scored on IS data only.
    OOS scoring is deferred to Task 13.

13. **No parameter sensitivity.** Testing the model at EMA 9/10/11 is Task 11.

14. **Model card is a template.** Dataset hash and coefficient values will be
    populated when run on real data. The model card structure and all fixed
    content are locked here.
