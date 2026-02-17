# B1 Detector: Implementation Assumptions & Ambiguities

Covers decisions made while translating B1_Strategy_Logic_Spec_v2.md into `src/ctl/b1_detector.py`.

---

## C1: Slope

- **EMA warmup.** The spec says seed with first available close. We use `ewm(span=10, adjust=False)` which does exactly this. However, the first ~200 bars produce EMA values that haven't converged. We skip the first `max(slope_lookback, 200)` bars before evaluating triggers, matching the spec's "200-bar warmup" validation note (§1.3).
- **NaN handling.** If `Slope_20` is NaN (insufficient history), the bar is silently skipped — no trigger fires.

## C2: Bars of Air

- **"Bars before N."** The spec says count starts at N-1, not N. Bar N is the trigger bar whose Low touches the EMA, so it is excluded from the air count. Implemented as `range(1, max_lookback+1)` scanning backward from N.
- **Edge: bars-of-air lookback hits the start of data.** If we reach index 0 without finding a touching bar, we return the number of bars scanned. This is conservative — it may over-count air for the very first bars after warmup, but those bars are already inside the warmup exclusion zone in practice.

## C3: EMA Violation

- **Inclusive comparison.** Spec: `Low[N] <= EMA10[N]`. We use `<=` (touch OR pierce). A bar whose low is exactly equal to the EMA10 value qualifies.

## C4: Not Breakdown

- **ATR14 computation.** We use Wilder smoothing (`ewm(alpha=1/14, adjust=False)`) which is the standard TradeStation ATR definition. The spec does not explicitly state Wilder vs simple ATR; we chose Wilder for TradeStation cross-validation compatibility.
- **Boundary.** Spec: `Close[N] > EMA10[N] - (ATR14[N] * 0.5)`. The `>` is strict. A close exactly at the boundary does NOT qualify — it is treated as a breakdown.

## Entry / Confirmation

- **Grace bar indexing.** Spec §3.1 is explicit: grace bar 1 = bar N+1. Confirmation requires `Close > EMA10` (strict `>`). Earliest entry = bar N+2 open.
- **Entry bar beyond data.** If confirmation occurs on the last bar of data, the entry bar (next day) doesn't exist yet. We still record the trigger as confirmed but set `entry_price = None`. The simulator or live pipeline is responsible for filling that entry.

## Retrigger / Multi-Position

- **Retrigger suppression (§3.4).** While a pending trigger awaits confirmation, all new triggers are ignored. This is stateful — the detector must be run sequentially, not bar-by-bar independently.
- **Multi-position suppression (§3.5).** Phase 1a default: ignore triggers while in an open position on the same symbol+timeframe. Because the detector does not simulate exits, we use a 60-bar heuristic timeout to reset the "in position" flag for standalone detection runs. The canonical path is detect+simulate together, where the simulator explicitly releases the position on exit.
- **Cross-timeframe independence.** A daily position does NOT suppress a weekly trigger on the same symbol (or vice versa). The detector is called separately per timeframe.

## Fibonacci Targets

- **Swing high tie-break.** Spec §5.1: "if multiple bars share the highest high, use the MOST RECENT." We reverse the lookback window before calling `argmax` so the first match in the reversed array corresponds to the most recent bar.
- **Lookback window for swing high.** `Highest(High[N-1], SwingLookback)` — the window ends at bar N-1 (trigger bar excluded) and extends back SwingLookback bars. If fewer bars are available, we use what exists.
- **TP levels computed at trigger time.** They are frozen when the trigger fires and never recalculated. SwingLow = Low[N] (trigger bar low).

## Williams %R

- **Division by zero.** If Highest(High,10) == Lowest(Low,10) over the lookback window (perfectly flat bars), the denominator is zero. We return NaN for that bar. This does not affect trigger detection (Williams %R is a confluence flag, not a trigger condition).

## Not Implemented (Phase 1a exclusions)

- No gap filter (§3.3) — all entries taken regardless of gap.
- No pyramiding — one position per symbol+timeframe.
- Stop management modes 2 & 3 — static stop only.
- Nested HTF trigger (§9.3) — Phase 2.
