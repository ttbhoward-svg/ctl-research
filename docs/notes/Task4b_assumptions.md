# Task 4b Assumptions — Slippage Stress Test

## Analytical vs Re-simulation

1. **Analytical adjustment**: The stress module adjusts baseline TradeResults analytically rather than re-running the full simulator. This is exact for slippage's effect on entry and stop-exit prices. The alternative (re-simulation) is just calling `simulate_all` with a different `SimConfig.slippage_per_side` — the module provides the metrics/gate framework either way.

2. **Baseline slippage**: By default, assumes the input TradeResults were generated with 0 slippage (`SimConfig` default). A `baseline_slippage` parameter allows correction if the baseline used non-zero slippage.

## What Changes Under Slippage

3. **Entry price**: Worsened by slippage delta (long entry moves higher).
4. **Stop exit price**: Worsened by slippage delta (next-bar open minus slippage).
5. **TP exit price**: Unchanged (resting limit order fills at level exactly per spec §4.4).
6. **"Open" exit price**: Unchanged (close-of-last-bar snapshot, not a fill).
7. **TP hit/miss status**: Unchanged (TP evaluation is High-based, unaffected by slippage).
8. **Risk per unit**: Increases with slippage (wider entry-to-stop distance). Trades where slippage makes risk <= 0 are excluded and counted in `n_excluded`.

## Metrics

9. **R-multiple**: Both actual and theoretical R are recomputed per scenario using the adjusted entry, exit, and risk. TheoreticalR (perfect thirds) is the primary metric.
10. **Win rate**: Fraction of trades with adjusted TheoreticalR > 0 (profit-based). The TP1-hit rate is constant across scenarios since TP evaluation is High-based, so using it for win rate would hide the slippage impact.
11. **MAR proxy**: `total_r / max_drawdown_in_cumulative_r`. This is a trade-level proxy using cumulative R drawdown, not annualized CAGR/MaxDD. Sufficient for comparing degradation across scenarios.
12. **"Profitable"**: Total TheoreticalR > 0 across all trades in the scenario.

## Gate Logic

13. **Gate 1 item 8**: "Slippage stress test: profitable at 2 ticks per side." Pass if total_r > 0 at the 2-tick scenario.
14. **Kill criterion**: "Edge evaporates at 2 ticks" is equivalent to total_r <= 0 at 2 ticks. Triggers REJECT.
15. **Robustness note**: Per Tracker v3, "if profitable at 3 ticks, edge is robust." Tracked in report but not a gate requirement.
