# Task 6 Assumptions — Confluence Flags

## Implemented Now (7 flags, all from Spec §8)

1. **WR_Divergence (§8.1)** — Deterministic, OHLCV-only. Scans backward from N-1 through air bars to find most recent bar where Low <= EMA10 (prior touch). Max scan = 100 bars. False if no prior touch found.

2. **CleanPullback (§8.2)** — Deterministic. Strict inequality: `High[N-1] < High[N-2] < High[N-3]` AND `Low[N-1] < Low[N-2] < Low[N-3]`. Ties fail.

3. **VolumeDeclining (§8.3)** — Deterministic. Compares average volume of bars [N-1..N-3] vs [N-4..N-6]. Requires 6 bars of history before trigger; False otherwise.

4. **GapFillBelow (§8.4)** — Deterministic. Scans backward from N for gap-down bars (Open[j] < Close[j-1]). Checks whether gap bottom is within 2% below StopPrice and gap is unfilled. Uses `gap_scan_window` parameter (default 100, pre-registered range 50-200 per §11).

5. **MultiYearHighs (§8.5)** — Deterministic. Compares trigger's SwingHigh to 252-bar rolling high ending at N-1. True if SwingHigh >= 95% of that yearly high. If insufficient history (< 252 bars), uses available history (conservative).

6. **SingleBarPullback (§8.7)** — Deterministic. Caution flag (NOT positive confluence). True if High[N-1] equals the SwingLookback-bar maximum — meaning only 1 bar of pullback before trigger.

7. **FibConfluence (§8.6)** — Requires `weekly_df`. Uses weekly SwingHigh/Low over `swing_lookback * 2` weekly bars to compute HTF 0.618 fib level. True if TP1 within 1% of that level. Returns `None` when weekly data not provided (same pattern as MTFA flags).

## Deferred to Later Phases

- **COT_Commercial_Pctile_3yr, COT_Commercial_Zscore_1yr** — External data, Task 7.
- **VIX_Regime** — External data, Task 7.
- **H4 context flags** (h4_above_200sma, h4_trend_aligned, etc.) — Research columns, logged as NULL in Phase 1a.
- **Fib retracement nest** (pullback_fib_618_level, fib_retracement_nest) — Research columns.
- **Meta-labeling fields** (take_label, skip_reason, meta_score) — Phase 2+.
- **FedLiquidityExpanding, Above200DMA** — Phase 2 features.

## Design Decisions

8. **GapFillBelow proximity**: "Within 2% below StopPrice" = `stop_price * 0.98 <= gap_bottom < stop_price`. Gap must be below the stop but not more than 2% below.

9. **NaN handling**: All flag functions return False when encountering NaN in required fields.

10. **Confluence flags are always computed**: Unlike MTFA flags (which are None when HTF data absent), the 6 OHLCV-only confluence flags are always computed (True/False) since they only need the daily DataFrame that's already present. Only FibConfluence can be None.
