# Task C Assumptions — Reconciliation Engine + Health Gating

1. **Primary vs secondary.** Databento is the primary source; Norgate is the
   secondary. Both provide canonical-schema DataFrames (Task A). Reconciliation
   compares matched bars on the same symbol/timeframe.

2. **Join key.** Bars are matched on `timestamp` (after both providers
   normalise to datetime64[ns, UTC]).  Unmatched timestamps become "missing
   bars" in the respective provider.

3. **Close divergence in ticks.** Computed as
   `|primary_close - secondary_close| / tick_value` per bar, then the mean
   across matched bars is the check metric. Tick values come from
   `universe.TICK_VALUES` (futures) and `universe.EQUITY_TICK` (equities/ETFs).

4. **Volume divergence %.** Computed as
   `|primary_vol - secondary_vol| / primary_vol * 100` per bar (primary is
   denominator). Bars where primary volume is zero are excluded from the mean.

5. **Roll date alignment (futures only).** A "potential roll date" is any bar
   where `|close[t] - close[t-1]| > roll_threshold_ticks * tick_value`. The
   sets of roll dates from each provider are compared. Misaligned roll dates
   flag a WATCH or ALERT. Non-futures skip this check.

6. **Missing bars.** Timestamps present in one provider but not the other.
   Counted separately per side (missing-in-primary, missing-in-secondary).

7. **Duplicate bars.** Repeated timestamps within a single provider's DataFrame.
   Any duplicate is an ALERT.

8. **Thresholds (defaults, configurable).**
   | Check              | OK         | WATCH         | ALERT         |
   |--------------------|------------|---------------|---------------|
   | Bar count diff     | <= 2       | 3–5           | > 5           |
   | Close (mean ticks) | <= 1.0     | 1.0–3.0       | > 3.0         |
   | Volume (mean %)    | <= 5.0     | 5.0–15.0      | > 15.0        |
   | Missing bars       | <= 2       | 3–5           | > 5           |
   | Duplicate bars     | 0          | —             | > 0           |
   | Roll misalignment  | 0          | 1             | > 1           |

9. **Status hierarchy.** Per-symbol status = worst check. Aggregate status =
   worst symbol.

10. **Gate behaviour.** `report.gate_allows_downstream` returns `False` when
    aggregate status is ALERT. Downstream scoring/signal modules should check
    this flag before proceeding.

11. **Persistence.** `save_report()` writes a JSON artifact with full details
    and a companion CSV with one row per symbol (symbol, status, check values).

12. **No trigger parity.** Per constraints — trigger-level reconciliation is
    deferred to a later task.

13. **Roll threshold.** Default 5 ticks. This is deliberately generous to
    catch Panama-Canal adjustments while ignoring normal daily volatility.
