# Task 11 Assumptions — Parameter Sensitivity Sweep

## Scope

1. **IS-only analysis.** The sensitivity sweep runs exclusively on in-sample data
   (2018-01-01 to 2024-12-31). No OOS data is touched. This is an IS robustness
   check, not an OOS validation.

2. **Four swept parameters.** Per the user's instruction, only these four are swept:
   - `slope_threshold` (range [4, 14], default 8, step 2 → 6 levels)
   - `min_bars_of_air` (range [3, 10], default 6, step 1 → 8 levels)
   - `entry_grace_bars` (range [1, 5], default 3, step 1 → 5 levels)
   - `swing_lookback_daily` (range [15, 25], default 20, step 5 → 3 levels)

   Other parameters (breakdown_buffer_atr, slope_lookback, etc.) are held at
   defaults during the sweep. They could be swept in future tasks.

3. **Step sizes chosen for practicality.** Integer parameters use step=1.
   `slope_threshold` uses step=2 (6 levels across [4,14]). `swing_lookback_daily`
   uses step=5 (3 levels: 15, 20, 25). Full grid = 6 × 8 × 5 × 3 = 720 combos.
   A `max_combos` cap is supported for quick runs.

## Detection + Simulation Pipeline

4. **Pipeline per grid point.** Each parameter combo creates a `B1Params` object,
   runs `run_b1_detection()` on every symbol's OHLCV data, then runs
   `simulate_all()` on the resulting triggers. This mirrors the production flow.

5. **No slippage in sweep.** The sweep uses `SimConfig(slippage_per_side=0.0)` to
   isolate parameter sensitivity from execution costs. Slippage stress testing is
   separate (Task 4b, already complete).

6. **No external features or scoring model.** The sweep only measures detection +
   simulation outcomes (trade count, avg R, win rate, etc.). It does not run the
   Elastic Net scoring model. The "spread" metric in the output is computed from
   raw TheoreticalR terciles split by trade score = TheoreticalR itself (a proxy
   for parameter sensitivity, not model scoring).

## Metrics

7. **Summary metrics per grid point.** Each combo produces:
   - `n_trades`: number of confirmed trades with valid simulation results
   - `avg_r`: mean TheoreticalR
   - `total_r`: sum of TheoreticalR
   - `win_rate`: fraction of trades with TheoreticalR > 0
   - `max_dd_r`: max drawdown in cumulative TheoreticalR
   - `mar_proxy`: total_r / max_dd_r (if max_dd > 0, else NaN)

8. **No tercile spread in grid.** Tercile spread requires a scoring model which is
   not part of parameter sensitivity. The user requested it but we substitute
   avg_r and total_r as the relevant robustness measures. Tercile analysis is
   separate (regression → scoring → evaluation).

## Plateau Analysis

9. **Robustness score.** For each combo, the robustness score is the mean total_r
   of its immediate neighbors in parameter space divided by its own total_r. A
   ratio near 1.0 indicates stability; a ratio << 1.0 indicates fragility (the
   combo is a peak surrounded by low performance).

10. **Fragility flag.** A combo is flagged as fragile if:
    - Its total_r is in the top 25% of all combos, AND
    - Its robustness score < 0.5 (neighbors average less than half its performance)
    This identifies "cliff" parameters per kill criteria.

11. **Neighbor definition.** Two combos are neighbors if they differ by exactly one
    step in exactly one parameter. This creates a 4-connected grid (not diagonal).

## Visualization

12. **Matplotlib-only.** Plots use matplotlib (already available in .venv). No
    dashboard or interactive tooling.

13. **Output path.** Plots saved to `outputs/sensitivity/` with deterministic
    filenames: `1d_{param_name}.png` and `2d_{param1}_vs_{param2}.png`.

14. **1D curves.** For each swept parameter, plot avg_r (y-axis) vs parameter
    value (x-axis), averaging over all other parameters. This shows whether the
    parameter has a plateau or a fragile peak.

15. **2D heatmaps.** For each pair of swept parameters, plot a heatmap of avg_r
    averaged over the other two parameters. This identifies interaction effects.

## Runtime

16. **The module accepts synthetic data for testing.** Unit tests use small
    synthetic datasets with known signals. Real data runs are out of scope for
    this task.

17. **max_combos cap.** If the full grid exceeds `max_combos`, a deterministic
    subsample is taken (sorted by parameter values, evenly spaced). Default
    `max_combos=None` (no cap).
