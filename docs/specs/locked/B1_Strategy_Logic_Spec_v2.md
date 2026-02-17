# B1 Strategy Logic Specification (Language-Agnostic) — v2
## Updated Feb 17, 2026 — Integrated Codex Round 2 feedback

## Purpose

This document defines the B1 (10 EMA Retest) strategy logic with sufficient precision to implement in ANY programming language and produce identical results. Every edge case is resolved. Every calculation is deterministic. This is the "source of truth" — if the code disagrees with this document, the code is wrong.

---

## 1. Data Requirements

### 1.1 Bar Data

Each symbol requires three timeframes of OHLCV bar data:

| Timeframe | Label | Bar Definition |
|-----------|-------|---------------|
| Entry TF | `daily` | Calendar day, exchange session hours |
| Higher TF 1 | `weekly` | Monday 00:00 to Friday close (exchange session) |
| Higher TF 2 | `monthly` | First trading day to last trading day of calendar month |

For futures: use back-adjusted (Panama Canal) continuous contracts.
For equities: use split/dividend-adjusted close.

### 1.2 Weekly/Monthly Bar Alignment Rule

When evaluating a daily trigger on date D:
- **"Last completed weekly bar"** = the weekly bar whose final session close is the most recent end-of-week session BEFORE date D. Use the `exchange_calendars` Python library to determine actual session end dates — do NOT hardcode "Friday." Thanksgiving weeks, half-days, and early closes will break a Friday assumption.
- **"Last completed monthly bar"** = the monthly bar whose last trading day is in the most recent complete calendar month BEFORE date D's month. If D is March 15, the last completed monthly bar ended on the last trading day of February (per exchange calendar).

**CRITICAL:** A weekly bar is NOT complete until its final session close is finalized. A monthly bar is NOT complete until its last-trading-day close is finalized. You may never use data from the currently developing higher-TF bar.

**Implementation:** Use `exchange_calendars.get_calendar('XNYS')` (or appropriate exchange) to get the actual trading schedule. Group daily bars into weeks by the exchange's week boundaries, not calendar weeks.

### 1.3 EMA Calculation

Use the standard exponential moving average formula:

```
multiplier = 2 / (period + 1)
EMA[0] = Close[0]  (seed with first available close)
EMA[t] = Close[t] * multiplier + EMA[t-1] * (1 - multiplier)
```

For a 10-period EMA: multiplier = 2/11 = 0.181818...

**Validation:** On any symbol, the 10 EMA computed by this formula must match TradeStation's `XAverage(Close, 10)` within 0.01% after a 200-bar warmup period. If using pandas: `df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()` — this matches the formula above.

### 1.4 Indicator Definitions

```
EMA10 = EMA(Close, 10)          # per formula above
SMA21 = SMA(Close, 21)          # simple arithmetic mean of last 21 closes
EMA50 = EMA(Close, 50)
SMA200 = SMA(Close, 200)
WilliamsR = ((Highest(High, 10) - Close) / (Highest(High, 10) - Lowest(Low, 10))) * -100
```

Williams %R range: -100 (oversold) to 0 (overbought). Threshold: -80 = oversold, -20 = overbought.

---

## 2. B1 Core Trigger Conditions

A B1 trigger fires on bar N if ALL FOUR conditions are TRUE on bar N:

### C1: Uptrend (Slope)
```
Slope_20 = ((EMA10[N] / EMA10[N - SlopeLookback]) - 1) * 100
C1 = Slope_20 >= SlopeThreshold
```
- SlopeLookback: default = 20
- SlopeThreshold: default = 8 (meaning EMA10 has risen 8%+ over 20 bars)

### C2: Bars of Air
```
BarsOfAir = 0
For i = 1 to MaxBarsOfAirLookback:
    If Low[N - i] <= EMA10[N - i]:
        BarsOfAir = i - 1
        BREAK
    Else:
        BarsOfAir = i

C2 = BarsOfAir >= MinBarsOfAir
```
- MaxBarsOfAirLookback: default = 50
- MinBarsOfAir: default = 6

**Note:** We look BACKWARD from bar N (bar N-1, N-2, etc.) to count how many consecutive bars had Low > EMA10. Bar N itself is the trigger bar (its Low is touching the EMA), so we start counting at N-1.

### C3: EMA Violation
```
C3 = Low[N] <= EMA10[N]
```
The low of the trigger bar touches or pierces the 10 EMA.

### C4: Not Breakdown
```
C4 = Close[N] > EMA10[N] - (ATR14[N] * BreakdownBuffer_ATR)
```
- BreakdownBuffer_ATR: default = 0.5
- Pre-registered sweep: 0.3 - 0.7
- Stability test: must work at 0.4 and 0.6

The close must remain within 0.5 ATR below the EMA. This filters out hard breakdowns where price slices through the EMA on heavy selling. ATR-based buffer self-calibrates across instruments of different price levels and volatility regimes (e.g., /PA at $1,200 vs /ES at $5,000 vs $MU at $90).

### Trigger
```
B1_Trigger[N] = C1 AND C2 AND C3 AND C4
```

---

## 3. Entry Logic

### 3.1 Confirmation Window

After a trigger fires on bar N, we wait for CONFIRMATION:

```
For grace_bar = 1 to EntryGraceBars:
    Check bar [N + grace_bar]:
        If Close[N + grace_bar] > EMA10[N + grace_bar]:
            CONFIRMED on bar [N + grace_bar]
            Entry on bar [N + grace_bar + 1] at OPEN
            BREAK
```

- EntryGraceBars: default = 3
- **Indexing rule:** Bar N is the trigger bar. Bar N+1 is grace bar 1 (first bar checked for confirmation). Bar N itself NEVER counts for confirmation. Earliest possible entry = bar N+2 open.

If no confirmation within EntryGraceBars, the trigger EXPIRES. No entry.

### 3.2 Entry Price

```
EntryPrice = Open[N + grace_bar + 1]
```

This is the open of the bar AFTER confirmation. In a real trade, this represents a market order at the open.

**Slippage:** Add slippage per instrument class (Section 2.4 of Codex Spec):
```
AdjustedEntryPrice = EntryPrice + SlippagePerSide
```

### 3.3 Gap Filter (NOT in current spec — DECISION NEEDED)

If `Open[entry_bar] > EMA10[entry_bar] * 1.05` (5% gap above EMA), should we skip the entry?

**Current rule: NO gap filter.** All entries taken regardless of gap. This may inflate backtest results with entries you'd never actually take. Monitor during Phase 1a and add as a filter in Phase 1b if needed.

### 3.4 Retrigger Rule

```
If PendingEntry == TRUE (waiting for confirmation on a prior trigger):
    IGNORE any new B1 trigger
    Do NOT overwrite TriggerBar, StopPrice, TP levels, or expiry
```

### 3.5 Multiple Position Rule

```
If already in an OPEN B1 trade on this symbol+timeframe:
    OPTION A: Ignore new triggers entirely (default for Phase 1a)
    OPTION B: Allow pyramiding (FUTURE — not Phase 1a)
```

**Cross-timeframe:** A daily B1 and a weekly B1 on the same symbol ARE treated as separate trades. You can be in both simultaneously. They have separate stops, targets, and R-multiples.

---

## Execution Conventions (Canonical — applies to all trade simulation)

These rules govern how fills are modeled in the Python simulator and must be consistent across all documents:

- **Entry fill:** Open of entry bar + slippage (per instrument class table in Section 6).
- **Stop evaluation:** Close-based. Stop breaches when `Close[bar] < StopPrice`.
- **Stop fill:** Next-bar open + slippage. `ExitPrice = Open[bar+1] - SlippagePerSide`.
- **TP evaluation:** High-based. TP hit when `High[bar] >= TP_level`.
- **TP fill:** At TP level exactly (resting limit order assumption).
- **Same-bar collision (TP hit AND stop breached):** TP wins (default). Configurable via `CollisionRule` parameter.

---

## 4. Stop Loss

### 4.1 Initial Stop

```
StopPrice = Low[N]  (low of the trigger bar, NOT the entry bar)
```

### 4.2 Stop Evaluation (Close-Based)

The stop is evaluated at each bar's CLOSE, not intrabar:

```
For each bar after entry:
    If Close[bar] < StopPrice:
        EXIT at Close[bar]  (or next bar open — see 4.3)
```

### 4.3 Stop Exit Timing

**Rule (per Codex recommendation):** Stop triggers when the close breaches the stop level. Exit occurs at the NEXT bar's open, plus slippage. No same-bar close fills on stops.

```
If Close[bar] < StopPrice:
    STOP TRIGGERED on bar [bar]
    ExitPrice = Open[bar + 1] - SlippagePerSide
    ExitDate = Date[bar + 1]
```

Rationale: This is the most conservative AND most realistic assumption. In practice, you see the close breach your stop at end of day, then exit at the next session's open. It avoids the unrealistic assumption that you can execute at the exact closing price that triggered the stop.

### 4.4 Same-Bar TP and Stop Collision

```
If High[bar] >= TP_level AND Close[bar] < StopPrice:
    TP TAKES PRIORITY (default — assumes resting limit orders at TP levels)
    Exit at TP level, not stop
    SET SameBarCollision = TRUE (logged for monitoring)
```

Rationale: During the bar, price hit the TP first (the high exceeded the level). If a resting limit order was placed at the TP, it would have filled before the close. The close subsequently fell below the stop, but the trade was already closed.

**Conservative alternative (Gemini recommendation):** Stop takes priority. Use this if NOT running resting limit orders. Set via input parameter: `CollisionRule = 'TP_wins'` (default) or `'Stop_wins'`.

**Monitoring rule:** If SameBarCollision occurs on more than 5% of trades, investigate whether TP levels are too tight or stop levels are too loose. A high collision rate suggests the TP and stop are too close together relative to the instrument's typical bar range.

---

## 5. Targets

### 5.1 Fibonacci Levels

```
SwingHigh = Highest(High[N-1], SwingLookback)
    Tie-break: if multiple bars share the highest high, use the MOST RECENT
SwingLow = Low[N]  (trigger bar low)
Range = SwingHigh - SwingLow

TP1 = SwingLow + Range * 0.618
TP2 = SwingLow + Range * 0.786
TP3 = SwingLow + Range * 1.000  (= SwingHigh)
TP4 = SwingLow + Range * 1.618
TP5 = SwingLow + Range * 2.618
```

- SwingLookback: timeframe-dependent
  - Daily: default = 20 (sweep 15-25, stability test at 18 and 22)
  - Weekly: default = 12 (sweep 8-16, stability test at 10 and 14)

### 5.2 Partial Exit Schedule

```
At TP1: exit 1/3 of position
At TP2: exit 1/3 of position
At TP3: exit remaining 1/3
```

### 5.3 Fractional Contract Rounding

For futures where 1/3 of position is fractional:
```
If TotalContracts == 1:
    Exit 100% at TP1 (cannot split 1 contract)
If TotalContracts == 2:
    Exit 1 at TP1, exit 1 at TP3 (skip TP2)
If TotalContracts >= 3:
    TP1_lots = Floor(TotalContracts / 3)
    TP2_lots = Floor(TotalContracts / 3)
    TP3_lots = TotalContracts - TP1_lots - TP2_lots
```

For equities: same logic but in shares. Minimum lot = 1 share.

### 5.4 TP Evaluation

```
For each bar after entry:
    If High[bar] >= TP1 AND TP1 not yet hit:
        Partial exit at TP1 price (not at High)
    If High[bar] >= TP2 AND TP2 not yet hit:
        Partial exit at TP2 price
    ... etc.
```

**Multi-TP same bar:** If High exceeds multiple TPs on the same bar, exit at the HIGHEST qualifying TP:
```
If High[bar] >= TP3:
    Exit all remaining at TP3 (even if TP1 and TP2 weren't hit yet)
```

---

## 6. Stop Management After Entry

### 6.1 Modes (input parameter: StopMgmtMode)

| Mode | Behavior |
|------|----------|
| 1 (Static) | Stop stays at trigger bar low. Never moves. **Phase 1a default.** |
| 2 (Breakeven) | After TP1 hit, stop moves to EntryPrice. |
| 3 (Trailing) | After TP1 hit, stop trails under EMA10 (current bar's EMA10). |

**Phase 1a:** Use Mode 1 only. Modes 2 and 3 are future enhancements.

---

## 7. R-Multiple Calculation

### Actual R (Account-Constrained)
```
RiskPerUnit = EntryPrice - StopPrice  (always positive for longs)

For each partial exit:
    PartialR = (ExitPrice - EntryPrice) / RiskPerUnit

WeightedR = sum of (PartialR * fraction_of_position) across all exits

# Example: 1/3 exits at TP1 (2.0R), 1/3 at TP2 (3.0R), 1/3 stopped at breakeven (0R)
# WeightedR = (2.0 * 0.333) + (3.0 * 0.333) + (0.0 * 0.333) = 1.67R
```

### Theoretical R (Unlimited Splitting)

Always compute as if the position can be split into perfect thirds, regardless of actual contract count:
```
TheoreticalR = (TP1_R * 0.333) + (TP2_R * 0.333) + (TP3_R * 0.333)
    where TP1_R = (TP1 - EntryPrice) / RiskPerUnit (if TP1 was hit)
    If trade stopped before TP1: TheoreticalR = (ExitPrice - EntryPrice) / RiskPerUnit
    If trade hit TP1 but stopped before TP2: the TP1 partial is locked in, remainder stopped
```

**Why both:** At small account sizes (1-2 contracts), Actual R is capped by the inability to split positions. Theoretical R measures signal quality independent of account size. The Elastic Net regression uses Theoretical R as the dependent variable.

### Day 1 Fail Flag
```
Day1Fail = (Low[entry_bar] < StopPrice)
```
If the low of the entry bar drops below the stop price, the trade went against you immediately. This is a caution flag — the trade still runs (stop is close-based, not intrabar), but it's logged for research. A high rate of Day 1 Fails may indicate the entry timing is too slow.

### MFE and MAE

```
MFE_R = (Highest High during trade - EntryPrice) / RiskPerUnit
MAE_R = (EntryPrice - Lowest Low during trade) / RiskPerUnit
```

These are logged for research — they tell you how much R was "available" vs. how much was "risked."

---

## 8. Confluence Flags

Each flag is computed at the time of the trigger (bar N) and logged as a column in the output. All are deterministic.

### 8.1 Williams %R Divergence
```
PriorTouch = most recent bar before N where Low <= EMA10
WR_Divergence = (WilliamsR[N] > WilliamsR[PriorTouch]) AND (Low[N] <= Low[PriorTouch])
```
Price made a lower low (or equal) but %R made a higher low = bullish divergence.

If no prior touch exists within 100 bars: WR_Divergence = FALSE.

### 8.2 Clean Pullback
```
CleanPullback = (High[N-1] < High[N-2] < High[N-3]) AND (Low[N-1] < Low[N-2] < Low[N-3])
```
Three consecutive bars of lower highs AND lower lows approaching the EMA. Orderly stair-step decline.

### 8.3 Volume Declining on Pullback
```
RecentAvgVol = Average(Volume, 3) of bars [N-1, N-2, N-3]
PriorAvgVol = Average(Volume, 3) of bars [N-4, N-5, N-6]
VolumeDeclining = RecentAvgVol < PriorAvgVol
```

### 8.4 Gap Fill Below
```
GapFillBelow = FALSE
For i = 1 to GapScanWindow (default 100):
    If Open[N-i] < Close[N-i-1]:  // gap down exists
        GapTop = Close[N-i-1]
        GapBottom = Open[N-i]
        If GapBottom is within 2% below StopPrice:
            // Check if gap has been filled
            Filled = any bar between N-i and N where High >= GapTop
            If NOT Filled:
                GapFillBelow = TRUE
                BREAK
```
An unfilled gap below the trigger provides a "magnet" — price may revisit it, adding risk.

### 8.5 Multi-Year Highs
```
MultiYearHighs = (SwingHigh >= Highest(High, 252)[N-1] * 0.95)
```
The setup's swing high is within 5% of the 252-day (1 year) high. Trading near highs = less overhead resistance.

### 8.6 Fib Confluence (Multi-TF)
```
HTF_SwingHigh = Highest(High, SwingLookback * 2) of weekly bars
HTF_SwingLow = Lowest(Low, SwingLookback * 2) of weekly bars
HTF_Fib618 = HTF_SwingLow + (HTF_SwingHigh - HTF_SwingLow) * 0.618
FibConfluence = abs(TP1 - HTF_Fib618) / TP1 < 0.01  // within 1%
```
Daily fib target aligns with weekly fib level = structural confluence.

### 8.7 Single Bar Pullback (Caution Flag)
```
SingleBarPullback = (High[N-1] == Highest(High[N-1], SwingLookback))
```
The bar immediately before the trigger was the swing high. Only 1 bar of pullback = potentially premature trigger.

---

## 9. Multi-Timeframe Alignment (MTFA) Flags

### 9.1 WeeklyTrendAligned
```
Weekly_EMA10 = EMA(Weekly_Close, 10) of LAST COMPLETED weekly bar (see Section 1.2)
Weekly_Slope20 = ((Weekly_EMA10[current] / Weekly_EMA10[20 weeks ago]) - 1) * 100

WeeklyTrendAligned = (Weekly_Slope20 >= SlopeThreshold) AND (Weekly_Close > Weekly_EMA10)
```

Both conditions: the weekly 10 EMA is rising AND the weekly close is above it.

### 9.2 MonthlyTrendAligned
```
Monthly_EMA10 = EMA(Monthly_Close, 10) of LAST COMPLETED monthly bar
Monthly_Slope20 = ((Monthly_EMA10[current] / Monthly_EMA10[20 months ago]) - 1) * 100

MonthlyTrendAligned = (Monthly_Slope20 >= SlopeThreshold) AND (Monthly_Close > Monthly_EMA10)
```

Same logic as weekly but on monthly bars.

### 9.3 Nested HTF Trigger (State Machine) [PHASE 2]

Not implemented in Phase 1a. Documented in Codex Spec v2.2 Section 4.6.

---

## 10. Output Schema

Every trigger produces one row with these columns:

```
# Identification
Date, Ticker, Timeframe, SetupType

# Trade levels
EntryPrice, StopPrice, TP1, TP2, TP3, TP4, TP5

# Technical features (computed at trigger time)
BarsOfAir, Slope_20

# Confluence flags (all boolean except where noted)
WR_Divergence, CleanPullback, VolumeDeclining,
FibConfluence, GapFillBelow, MultiYearHighs, SingleBarPullback

# MTFA flags
WeeklyTrendAligned, MonthlyTrendAligned

# External data (merged by date in Python pipeline)
COT_Commercial_Pctile_3yr, COT_Commercial_Zscore_1yr, VIX_Regime

# Cluster identifier (for fixed effects in regression)
AssetCluster  # one of: IDX_FUT, METALS_FUT, ENERGY_FUT, RATES_FUT,
              # GRAINS_FUT, SOFTS_FUT, LIVESTOCK_FUT, FX_FUT,
              # ETF_SECTOR, EQ_COMMODITY_LINKED, EQ_MACRO_BELLWETHER

# Tradable status
TradableStatus  # 'tradable' or 'research_only'

# Phase 2 feature columns (logged as NULL in Phase 1a, populated in Phase 2)
FedLiquidityExpanding   # boolean: Fed BS 4wk change > 0 OR FedWatch cut prob > 50%
Above200DMA             # boolean: Close > SMA(Close, 200) at trigger time

# Meta-labeling fields (logged as NULL in Phase 1a backtest, populated in forward testing Phase 5+)
take_label              # boolean: NULL in backtest, TRUE/FALSE in forward testing
skip_reason             # string: NULL in backtest; codes: 'regime_conflict', 'event_risk',
                        #   'low_liquidity', 'theme_concentration', 'discretionary', 'unavailable'
meta_score              # float: NULL until meta-model built (Phase 2+)
meta_model_version      # string: NULL until meta-model built

# Research columns (logged for analysis, NOT used as Phase 1a regression features)
touch_depth_atr         # float: (EMA10[N] - Low[N]) / ATR14[N]. 0 = exact touch, higher = deeper.
is_drag_up              # boolean: TRUE if qualifying touch was retroactive (EMA caught up to old bar)
entry_price_n1          # float: Open of bar N+1 (for N+1 vs N+2 entry comparison)
swing_high_recent       # float: most recent local swing high (not necessarily absolute highest)
swing_high_short        # float: highest high in shorter lookback (10 bars)
tp1_recent, tp2_recent, tp3_recent  # fib targets from recent swing high
tp1_short, tp2_short, tp3_short     # fib targets from short lookback
stop_atr                # float: EntryPrice - (ATR14[N] * 1.5)
stop_swing_low          # float: Lowest(Low, 5) from trigger bar
weekly_trigger_daily_confirmation   # boolean: weekly triggers only — daily close above weekly EMA within 5 bars?
wide_stop_flag          # boolean: TRUE if stop distance > 2x ATR14 on trigger timeframe

# H4 context (computed at trigger time from instrument's native H4 bars)
h4_above_200sma         # boolean: H4 close > H4 SMA(200) at trigger time
h4_200sma_distance_atr  # float: (H4 close - H4 SMA200) / ATR14. Positive = above, negative = below.
h4_trend_aligned        # boolean: H4 close > H4 EMA10 AND H4 slope positive

# Fib retracement nest on pullback (EMA sitting near 61.8/78.6 of the move that created bars-of-air)
pullback_fib_618_level  # float: 61.8% retracement of swing_low_launch → SwingHigh
pullback_fib_786_level  # float: 78.6% retracement of swing_low_launch → SwingHigh
ema_fib_proximity_atr   # float: min distance from EMA10[N] to either fib level, in ATR units
fib_retracement_nest    # boolean: TRUE if ema_fib_proximity_atr < 0.5
                        # swing_low_launch = Lowest(Low, 12) around bar where air sequence began

# Outcome (filled after trade resolves)
RMultiple_Actual, TheoreticalR, MFE_R, MAE_R, Day1Fail,
TradeOutcome (Win/Loss/Expired)
ExitDate, ExitPrice, ExitReason (TP1/TP2/TP3/Stop/Expired)
```

---

## 11. Default Parameters (Phase 1a)

| Parameter | Default | Pre-Registered Range | Description |
|-----------|---------|---------------------|-------------|
| SlopeThreshold | 8 | 4–14 | Min % EMA rise |
| SlopeLookback | 20 | 15–30 | Bars for slope |
| MinBarsOfAir | 6 | 3–10 | Min air count |
| MaxBarsOfAirLookback | 50 | 20–100 | Max bars searched |
| EntryGraceBars | 3 | 1–5 | Confirmation window |
| BreakdownBuffer_ATR | 0.5 | 0.3–0.7 | Max ATR below EMA for close |
| SwingLookback (daily) | 20 | 15–25 | Bars for swing high (daily) |
| SwingLookback (weekly) | 12 | 8–16 | Bars for swing high (weekly) |
| StopMgmtMode | 1 | 1 only | Static stop only |
| GapScanWindow | 100 | 50–200 | Bars searched for gaps |

The "Pre-Registered Range" is the range used for the parameter sensitivity sweep. The strategy must remain profitable across the ENTIRE range, not just at the default.

### Parameter Stability Test (Pre-Registered)
The B1 strategy must remain profitable when:
- EMA period changed from 10 to 9 or 11
- SlopeThreshold changed ±2 from default
- MinBarsOfAir changed ±1 from default
If results collapse at these adjacent values, the default is likely curve-fit.

### Feature Set (Phase 1a) — 9 Candidates + 1 Control

| Feature | Type | Description |
|---------|------|-------------|
| BarsOfAir | Technical (continuous) | Count of bars with Low > EMA10 |
| Slope_20 | Technical (continuous) | % change in EMA10 over 20 bars |
| CleanPullback | Technical (boolean) | 3 bars of lower highs + lower lows |
| WR_Divergence | Technical (boolean) | Bullish Williams %R divergence |
| WeeklyTrendAligned | MTFA (boolean) | Weekly slope + close > weekly EMA10 |
| MonthlyTrendAligned | MTFA (boolean) | Monthly slope + close > monthly EMA10 |
| COT_Commercial_Pctile_3yr | External (continuous) | 3-year rolling percentile of commercial net position |
| COT_Commercial_Zscore_1yr | External (continuous) | 1-year z-score of commercial net position |
| VIX_Regime | External (boolean) | Prior day VIX close < 20 |
| AssetCluster | Control (categorical) | 11 clusters (see Output Schema) |

**Note on feature count:** 9 candidates + 1 cluster control. The cluster variable is a control (fixed effect) in the regression, not a feature being tested for predictive power. Elastic Net operates on the 9 candidates. Per Codex guidance, if any two features are >90% correlated (check via correlation heatmap before running Elastic Net), drop the less interpretable one.

**COT split rationale (per Codex):** 1-year z-score captures recent positioning shifts (responsive). 3-year percentile captures structural extremes (stable). Both are pre-registered as candidates. Elastic Net may zero out one or both.

**COT for non-futures:** ETFs and equities do not have COT data. These fields are NULL for ETF_SECTOR, EQ_COMMODITY_LINKED, and EQ_MACRO_BELLWETHER clusters. The Elastic Net handles this via cluster fixed effects — these clusters get their own intercepts that absorb the missing COT information. No imputation. Phase 2 substitute: short interest ratio (equities) or ETF fund flows.

---

## 12. Decisions Deferred to Phase 1a Results

These are open questions that will be resolved by examining Phase 1a data:

1. **Gap filter:** Should entries be skipped when price gaps >X% above EMA on entry bar?
2. **Stop exit timing:** Option A (exit at breaching close) vs Option B (exit at next open)?
3. **Multiple positions per symbol:** Allow or restrict?
4. **Minimum R to TP1:** Should we skip trades where TP1 is less than 1.5R from entry?
5. **Earnings filter:** Skip triggers within N days of earnings? (Equities only)

Each will be evaluated by comparing the full backtest results with and without the filter. If the filter improves the MAR ratio on both IS and OOS data, it gets added. Otherwise it stays out.
