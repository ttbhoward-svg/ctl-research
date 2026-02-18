# Task 7 Assumptions — COT + VIX Integration

## Features Implemented (2 COT + 1 VIX)

1. **COT_20D_Delta** — 20-trading-day change in commercial net position. Since COT data
   is weekly (one observation per week), this equals a 4-row lookback:
   `commercial_net[t] - commercial_net[t-4]`. Positive = commercials adding longs
   (bullish structural signal).

2. **COT_ZScore_1Y** — Z-score of commercial net position over trailing 52 weeks:
   `(commercial_net[t] - rolling_mean_52[t]) / rolling_std_52[t]`.
   Captures how extreme current positioning is relative to the past year.

3. **VIX_Regime** — Boolean. True if prior trading day's VIX close < 20.
   Implements Spec §11 definition: `VIX_Regime = 1 if prior day VIX close < 20`.

## Data Source Expectations

4. **COT input schema**: A single CSV (or DataFrame) with columns:
   - `publication_date` (datetime) — Friday release date. NOT the as-of Tuesday.
   - `symbol` (str) — Must match universe symbols (e.g., `/ES`, `/CL`).
   - `commercial_net` (float) — Commercial Long minus Commercial Short positions.
   User is responsible for pre-processing raw CFTC data into this format.
   A future parser for raw CFTC disaggregated reports can be added later.

5. **VIX input schema**: A single CSV (or DataFrame) with columns:
   - `date` (datetime) — Trading day.
   - `vix_close` (float) — VIX daily closing level.
   Optional column `vix3m_close` reserved for term-structure regime (not used in Phase 1a).

6. **COT applicability**: Only the 15 futures symbols in the universe have COT data.
   ETFs (7) and equities (7) return None for both COT features.
   This is structurally expected — the Elastic Net handles via cluster fixed effects.

## Lag / No-Lookahead Rules

7. **COT merge rule**: For a trigger on date D, use the most recent COT publication
   with `publication_date < D` (strict less-than). A trigger on Friday uses the
   PRIOR Friday's COT, not the same-day release (conservative — same-day release
   timing is uncertain and may occur after market close).

8. **VIX merge rule**: For a trigger on date D, use VIX close from the most recent
   trading day with `date < D` (strict less-than). This is the "prior day close"
   rule from the spec. A Monday trigger uses Friday's VIX close.

9. **Missing data handling**: If no COT/VIX row exists before a trigger date,
   the feature is set to None. No forward-fill, no imputation. Missing values
   are explicit and left for the regression pipeline to handle.

## VIX Regime — Term-Structure Fallback

10. **Phase 1a**: VIX-level-only regime (< 20 threshold). Term-structure data
    (VIX/VIX3M ratio for contango/backwardation) is not available in current scope.
    The `load_vix` function accepts an optional `vix3m_close` column; if present,
    a richer regime classification could be derived in a future phase.
    Documented here as an unresolved data-source dependency — not a blocker
    for Phase 1a since the level-only fallback is spec-compliant.

## Design Decisions

11. **COT delta window**: 4 weekly observations = approximately 20 trading days.
    Not exactly 20 calendar days due to holidays, but this is standard practice
    for weekly COT data and matches the feature name intent.

12. **Z-score minimum std**: If rolling std is zero (constant positioning for 52 weeks),
    z-score returns NaN → mapped to None in the merge. This is a degenerate case
    unlikely in practice but handled defensively.

13. **Merge is a separate pipeline step**: Unlike confluence/MTFA flags (computed
    during trigger detection), COT/VIX features are merged AFTER detection via
    `external_merge.merge_external_features()`. This keeps the detector independent
    of external data availability.

14. **B1Trigger fields**: Three optional fields added to B1Trigger dataclass:
    `cot_20d_delta`, `cot_zscore_1y`, `vix_regime` — all default to None.
    Set by the merge module, not by the detector.
