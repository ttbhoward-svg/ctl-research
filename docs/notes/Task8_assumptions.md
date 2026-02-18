# Task 8 Assumptions — Final DB Assembly + Health Checks

## Assembly Design

1. **One row per trade result.** The assembler combines a B1Trigger (all flags) with
   its corresponding TradeResult (outcomes). Unconfirmed/expired triggers that produce
   no TradeResult are excluded from the dataset. They can be analyzed separately if needed.

2. **Trigger-to-result matching** uses the composite key `(symbol, trigger_bar_idx)`.
   This is unique per trigger within a detection run.

3. **Universe metadata injection.** `AssetCluster` and `TradableStatus` are looked up
   from the Universe config at assembly time, not stored on the trigger or result.
   This ensures the dataset always reflects the current universe configuration.

4. **Deterministic sort order:** `Date` ascending, then `Ticker` ascending. This ensures
   reproducible row ordering across runs.

5. **Stable schema.** The assembler enforces a fixed column order defined in
   `SCHEMA_COLUMNS`. Any new column must be explicitly added to this list.

## Column Naming

6. **External features** use the Task 7 naming convention: `COT_20D_Delta`,
   `COT_ZScore_1Y`, `VIX_Regime`. These correspond to B1Trigger fields
   `cot_20d_delta`, `cot_zscore_1y`, `vix_regime`.

7. **Confluence/MTFA columns** use PascalCase in the output DataFrame to match the
   spec §10 schema (e.g., `WR_Divergence`, `WeeklyTrendAligned`), even though
   the B1Trigger fields are snake_case.

## Immutable Artifact

8. **File naming:** `phase1a_triggers_{version}_{YYYYMMDD}.csv`. Never overwrite —
   new versions get new filenames. Version string defaults to `v1`.

9. **SHA-256 hash** is computed on the CSV byte representation of the DataFrame
   (`df.to_csv(index=False).encode("utf-8")`). This is deterministic as long as
   the DataFrame content and column order are identical.

10. **Manifest file** is a JSON sidecar (`_manifest.json`) containing the hash,
    row/column counts, column list, and date range.

## Health Checks

11. **Critical fields** that must be non-null: `EntryPrice`, `StopPrice`, `TP1`,
    `SetupType`, `RMultiple_Actual`, `TheoreticalR`, `AssetCluster`.

12. **Structurally expected nulls:** COT fields are NULL for non-futures (ETFs,
    equities). MTFA flags can be NULL if HTF data was not provided. These are
    NOT health-check failures.

13. **Duplicate-key check** uses `(Date, Ticker, Timeframe, TriggerBarIdx)`.
    Multiple triggers on the same symbol on the same day at different bar indices
    are valid (if retrigger suppression allows).

14. **Symbol coverage** is reported but may not be 29/29 — not all symbols produce
    triggers in every data sample. It's an informational metric, not a hard fail.

15. **Date-range check** reports min/max dates and whether they span the expected
    IS period (2018-01-01 to 2024-12-31) and OOS period (2025-01-01+). Partial
    coverage is a warning, not a fail — depends on data availability.
