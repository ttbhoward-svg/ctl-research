# Task 13 Assumptions — OOS Test + Calibration

## Scope

1. **No model retraining or tuning.** This module applies a locked IS-trained
   model (Task 10 `ModelResult`) to OOS data. Scaler transform (not fit),
   model predict (not fit).

2. **Feature lock enforcement.** Only the 9 pre-registered candidate features
   + cluster dummies are used for scoring. The module raises if any required
   feature is missing from the OOS dataset. No new features may be introduced.

3. **OOS dataset schema.** OOS data follows the same `SCHEMA_COLUMNS` as IS
   data (Task 8 output). The caller is responsible for splitting IS vs OOS
   by date; this module accepts any DataFrame with the required columns.

## Scoring

4. **Scaler: transform only.** The IS-fitted `StandardScaler` is applied via
   `scaler.transform()` to OOS features. No re-fitting.

5. **Cluster dummies: aligned to IS columns.** OOS may have different cluster
   values. We use `pd.get_dummies` then `reindex(columns=model_result.cluster_columns)`
   with `fill_value=0` to ensure the OOS feature matrix matches IS dimensions.

6. **Raw score = model.predict(X_oos).** This is the predicted TheoreticalR.

## Bucket Assignment

7. **OOS terciles use IS thresholds.** Per standard practice, OOS trades are
   classified using the tercile cutoffs from IS training (`ModelResult.tercile_thresholds`).
   This avoids information leakage from OOS score distribution.

8. **OOS quintiles for calibration.** Quintile assignment uses OOS-internal
   quantiles (20th/40th/60th/80th percentiles of OOS scores) since calibration
   diagnostics are purely descriptive, not predictive.

## Gate Criteria (Computed, Not Decided)

9. **Gate 1 criteria evaluated but not finalized.** This module computes
   pass/fail for each criterion and returns structured results. The final
   gate decision is Task 14's responsibility.

10. **Criteria from pre_registration_v1.yaml and Phase Gate Checklist:**
    - `oos_trade_count >= 30`
    - `oos_tercile_spread >= 1.0R` (top avg R − bottom avg R)
    - `score_monotonicity`: top > mid > bottom on avg R
    - `quintile_calibration`: monotonically improving avg R across quintiles
    - Kill: top-tercile avg R < 0.5R → REJECT
    - Kill: score-R correlation < 0.05 → REJECT

## Metric Computation

11. **Avg R by bucket.** Mean of `TheoreticalR` (or `RMultiple_Actual` if
    specified) grouped by assigned bucket.

12. **Win rate by bucket.** Fraction of trades with `TheoreticalR > 0` per bucket.

13. **Score-outcome correlation.** Pearson correlation between raw OOS scores
    and realized `TheoreticalR`.

14. **Calibration reliability table.** For each quintile band, report:
    predicted score range, realized avg R, trade count, win rate.

## Implementation

15. **Deterministic.** Given the same `ModelResult` and OOS DataFrame, outputs
    are identical. No randomness in this module.

16. **Export formats.** JSON summary (gate-facing) + CSV bucket table.
