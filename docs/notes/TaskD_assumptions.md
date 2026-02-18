# Task D Assumptions — Cutover Parity Test Harness

1. **Three S-criteria.** EMA10 reproduction, trigger date parity, and trade
   outcome (R-multiple) parity. Each produces a PASS/FAIL result.

2. **Input format.** Both primary (Databento path) and reference (TradeStation
   archive) data are standard OHLCV DataFrames with columns
   ``Date, Open, High, Low, Close, Volume``. This is the format consumed by
   ``b1_detector.compute_indicators`` and ``simulator.simulate_all``.

3. **EMA parity.** EMA10 is computed independently on each dataset using the
   same ``indicators.ema(Close, 10)`` function. Divergence is
   ``|ema_primary - ema_reference| / ema_reference * 100``. The max divergence
   across all bars must be <= threshold (default 0.01%). Bars where the
   reference EMA is zero are excluded.

4. **Trigger date parity.** B1 detection runs identically on both datasets
   (same ``B1Params``). Trigger dates are compared as sets. PASS requires exact
   set equality. Mismatches are reported as "extra in primary" and
   "extra in reference".

5. **Trade outcome parity.** For triggers that appear in both datasets (matched
   by trigger_date), R-multiples are compared. Max absolute difference must be
   <= threshold (default 0.05 R). Unmatched triggers do not count toward this
   criterion — they are flagged by trigger parity instead.

6. **No threshold changes.** Per constraints — thresholds are fixed at build
   time and not modified in this task.

7. **Harness is provider-agnostic.** Accepts plain DataFrames, not provider
   objects. This decouples the harness from stub implementations and allows
   testing with synthetic data now and real data later.

8. **Symbols for live use.** The intended symbols are /PA, /ES, /CL daily,
   but the harness accepts any symbol string. Tests use synthetic data.

9. **Four output artifacts.**
   - ``ema_reproduction.csv`` — per-bar comparison (date, ema_primary,
     ema_reference, divergence_pct)
   - ``trigger_parity_report.csv`` — per-trigger match status
   - ``trade_outcome_parity.csv`` — per-matched-trade R comparison
   - ``cutover_parity_summary.json`` — PASS/FAIL by criterion + metadata

10. **Warmup bars excluded.** The first ``ema_period - 1`` bars are excluded
    from EMA divergence computation since the EMA is still seeding.

11. **Date alignment.** Primary and reference DataFrames are joined on Date.
    Bars present in one but not the other are reported but do not fail the
    EMA check — only overlapping bars are compared.

12. **Deterministic tests.** All unit tests use fixed-seed synthetic data so
    results are reproducible across machines.
