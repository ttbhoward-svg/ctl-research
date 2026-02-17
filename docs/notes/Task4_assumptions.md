# Task 4 Assumptions — Cross-Validation & Ranking Gate

## Purge Gap

1. **Calendar days, not trading days**: `purge_gap_days` is implemented as calendar days (matching the config field name `purge_gap_days: 30`). The locked spec requires the purge gap in **trading days** ("purge >= max holding period, default 30 trading days" — Phase Gate Checklist v2; "TimeSeriesSplit, 30-day purge gap (≥ max holding period per Codex)" — Tracker v3). The current implementation uses a calendar-day approximation, which is **temporary** — 30 calendar days ≈ 21 trading days, under-purging by ~30%.

   **TODO: Before any Gate decision relies on CV results, switch `PurgedTimeSeriesSplit` to trading-day purge logic** (either accept a trading-day calendar / exchange holidays list, or convert internally using `pd.bdate_range`). Until then, set `purge_gap_days: 45` in config to approximate 30 trading days conservatively.

2. **Forward purge only**: Training observations near the test boundary are removed. No embargo on test observations — walk-forward expanding windows don't re-use test data in subsequent folds.

3. **Fold skipping**: If purging leaves fewer than `min_train_size` (default 10) training samples in a fold, that fold is silently skipped rather than raising an error.

## Tercile / Quintile Assignment

4. **Rank-based splits**: Terciles use `argsort` + floor division. The top group absorbs the remainder when n is not divisible by 3 (or 5 for quintiles).

5. **Tercile monotonicity is strict**: `top > mid > bottom` (strict inequality). Ties fail. Matches Gate Checklist wording: "top > mid > bottom on avg R."

6. **Quintile calibration is non-decreasing**: `bin[i] <= bin[i+1]` for all adjacent pairs. Less strict than the tercile check — with 5 smaller bins, exact ties are more likely by chance. Matches "directionally calibrated" language in Gate 1 item 9.

## Gate 1 Logic

7. **Severity ordering**: REJECT > PAUSE > ITERATE > INCOMPLETE > PASS. A kill REJECT overrides everything. A kill PAUSE prevents PASS even if all 9 items pass.

8. **ITERATE vs INCOMPLETE**: If any evaluated item is False, the result is ITERATE regardless of pending (None) items — we already know it won't pass. INCOMPLETE only applies when all evaluated items pass but some items haven't been checked yet.

9. **Items not evaluated in Task 4**: Feature cap (#4), model card (#5), negative controls (#6), entry degradation (#7), slippage stress (#8). These are set by callers in later tasks via keyword arguments.

## Kill Criteria Scope

10. **Score-R correlation threshold**: "OOS score-R correlation < 0.05" is interpreted as the raw Pearson correlation, not the absolute value. A negative correlation (model actively wrong-directional) also triggers the kill.

11. **Cluster concentration check deferred**: "> 60% of IS top-tercile from single cluster" requires cluster labels and IS data. It is not computed by the ranking gate module.
