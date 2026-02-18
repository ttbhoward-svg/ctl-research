# Task 11c Assumptions — Entry Degradation Test

## Scope

1. **Phase 1a exploratory tolerances.** Per Gate 1 item 7 and Phase Gate Checklist:
   - total_r degradation <= 25%
   - win_rate degradation <= 5 percentage points
   - MAR degradation <= 30%
   These are *exploratory* tolerances. Production gate (Gate 5) may tighten them.

2. **Default degradation fraction: 30%.** Per `phase1a.yaml`:
   `entry_degradation_pct: 0.30`, `entry_degradation_seed: 42`.
   30% of trades are randomly selected for degradation.

## Degradation Modes

3. **Delayed entry (+1 bar).** The entry is moved from `entry_bar_idx` to
   `entry_bar_idx + delay_bars` (default 1). The new entry price is the Open
   of the delayed bar + original slippage. If the delayed bar is beyond data
   or produces risk <= 0, the trade is excluded.

4. **Adverse fill.** Entry price is worsened by a configurable dollar amount
   (`adverse_fill_amount`, default = ATR * 0.1 approximated as risk * 0.1).
   The new entry price = original entry + adverse amount. Risk is reduced.
   If risk <= 0, the trade is excluded.

5. **Combined mode.** Both delay and adverse fill are applied together for a
   worst-case assessment.

## Trade Re-simulation

6. **Re-simulate from new entry, not just adjust R.** For delayed entries, the
   trade is re-walked from the new entry bar using the same TP levels and stop
   price. This correctly handles cases where TPs are hit on different bars or
   where the stop triggers before a TP after delay.

7. **For adverse-fill-only mode,** the entry bar doesn't change — only the entry
   price worsens. R-multiples are recomputed using the slippage stress pattern:
   adjusted risk, theoretical R from perfect thirds with new entry.

8. **TP levels unchanged.** Degradation only affects entry — TP1/TP2/TP3 and
   stop_price remain the same (they come from the trigger, not the entry).

## Metric Computation

9. **Degradation percentages.** Computed the same way as slippage stress:
   - total_r_pct = (degraded - baseline) / |baseline| * 100
   - win_rate_pp = (degraded - baseline) * 100
   - mar_pct = (degraded_mar - baseline_mar) / |baseline_mar| * 100

10. **Baseline is zero-slippage simulation results.** The degradation test
    starts from the same baseline trade results used throughout Phase 1a.

## Implementation

11. **Deterministic selection.** Which trades get degraded is determined by
    `np.random.default_rng(seed).choice(n, k, replace=False)` where
    k = int(n * degradation_pct). Same seed => same selection.

12. **Self-contained module.** Accepts `List[TradeResult]` + OHLCV data dict,
    returns a structured report. Callable from future gate workflow.

13. **No OOS data.** Entry degradation is applied to IS results only.
