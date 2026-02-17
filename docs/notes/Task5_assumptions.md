# Task 5 Assumptions — MTFA Flags

## Flag Conditions

1. **Both conditions required per Spec §9**: WeeklyTrendAligned and MonthlyTrendAligned each require (1) HTF Slope_20 >= SlopeThreshold AND (2) HTF Close > HTF EMA10. The user's prompt mentions only Close > EMA10 as shorthand — full spec §9 implemented since the locked spec is authoritative.

2. **SlopeThreshold shared across timeframes**: The same `slope_threshold` parameter (default 8.0) is used for daily C1, weekly, and monthly slope checks. Spec §9 uses `SlopeThreshold` without distinguishing by timeframe.

## Weekly Bar Lookup

3. **Weekly completion = bar date**: Weekly bars are assumed to be dated on their completion date (last trading day of the week, typically Friday). `Date <= trigger_date` naturally yields the last completed weekly bar. For triggers on the same day as a weekly bar close (e.g., both on Friday), the bar IS completed at trigger evaluation time — both daily and weekly bars close simultaneously at market close.

4. **Holiday-shortened weeks**: No assumption about Friday specifically. The lookup uses `Date <= trigger_date` on whatever dates exist in the weekly DataFrame. If a week ends on Thursday due to a Friday holiday, that Thursday-dated bar is found correctly.

## Monthly Bar Lookup

5. **Prior month only**: Per user instruction, the monthly flag uses the last completed monthly bar from the calendar month BEFORE the trigger's month. Cutoff = last calendar day of the prior month. If trigger is March 15, cutoff is February 28/29. This is more conservative than "Date <= trigger_date" — it excludes any partial-month bar for the current month.

## Edge Cases

6. **Insufficient HTF history**: If EMA10 or Slope_20 cannot be computed due to insufficient bars (fewer than slope_lookback periods), the flag defaults to False. Conservative — no alignment claimed without evidence.

7. **Flags are None when HTF data not provided**: If `weekly_df` or `monthly_df` is not passed to `detect_triggers`, the corresponding flag remains `None` (not evaluated), distinct from `False` (evaluated, not aligned). This preserves backward compatibility — existing callers that don't supply HTF data see no behavior change.
