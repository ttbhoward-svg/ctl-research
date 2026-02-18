# Task G Assumptions — Parity Prep & Overlap Validator

1. **Databento continuous files.** Output of Task F's `build_all()` at
   `data/processed/databento/cutover_v1/continuous/{SYM}_continuous.csv`.
   Schema: `Date, Open, High, Low, Close, Volume, contract, adjustment`.
   Date is ISO-formatted (YYYY-MM-DD), parsed as `datetime64`.

2. **TradeStation reference files.** Expected at
   `data/raw/tradestation/cutover_v1/TS_{SYM}_1D_*.csv` (glob pattern).
   The `*` wildcard allows for any suffix (date ranges, version tags).
   Schema: at minimum `Date, Open, High, Low, Close, Volume` (standard
   TradeStation CSV export). Column names are case-insensitive (normalised
   via `str.lower()`).

3. **TS filename convention.** One file per symbol. If multiple files
   match the glob for a single symbol, the first sorted match is used
   and a warning is logged. Example filenames:
   - `TS_ES_1D_20180101_20260217.csv`
   - `TS_CL_1D_backadjusted.csv`
   - `TS_PA_1D.csv`

4. **Symbols.** Three futures roots for cutover: ES, CL, PA.

5. **Overlap window.** Computed per symbol as the intersection of dates
   present in both the Databento continuous file and the TradeStation
   reference file: `first_common_date`, `last_common_date`,
   `overlap_bar_count`.

6. **Preflight gate threshold.** Each symbol requires >= 500 overlapping
   daily bars to PASS. This ensures sufficient history for EMA warmup,
   trigger detection, and trade simulation.

7. **INCOMPLETE status.** If a TradeStation file is not yet delivered
   for a symbol, that symbol's status is `INCOMPLETE` (not FAIL). This
   allows the prep script to run before all TS files arrive. The overall
   gate status is `INCOMPLETE` if any symbol is INCOMPLETE, `PASS` if
   all symbols pass, and `FAIL` if any symbol has < 500 overlap bars
   despite both files being present.

8. **Report outputs.** Two artifacts:
   - `parity_prep_report.json` — machine-readable, includes gate status
     and per-symbol detail.
   - `parity_prep_report.csv` — one row per symbol with status, counts,
     and date ranges.

9. **No parity scoring.** This task only validates readiness. The actual
   EMA/trigger/trade parity checks (Task D harness) run separately once
   the gate passes.

10. **Date column parsing.** Both file types are parsed via
    `pd.to_datetime()`. Databento continuous files have a `Date` column;
    TradeStation files have a `Date` (or `date`) column. The join key
    is the date component only (no time).

11. **Required columns.** Databento continuous: `Date, Open, High, Low,
    Close, Volume`. TradeStation: `Date, Open, High, Low, Close, Volume`
    (after case normalisation). Missing columns → FAIL for that symbol.

12. **Output directory.** Reports written to
    `data/processed/cutover_v1/`. Directory is created if absent.

## Run instructions

```bash
# Build continuous series first (Task F):
python -c "
from ctl.continuous_builder import build_all
from pathlib import Path
base = Path('data/raw/databento/cutover_v1/outrights_only')
out = Path('data/processed/databento/cutover_v1/continuous')
build_all(['ES', 'CL', 'PA'], base, out)
"

# Place TradeStation files:
#   data/raw/tradestation/cutover_v1/TS_ES_1D_*.csv
#   data/raw/tradestation/cutover_v1/TS_CL_1D_*.csv
#   data/raw/tradestation/cutover_v1/TS_PA_1D_*.csv

# Run parity prep:
python -c "
from ctl.parity_prep import run_parity_prep
report = run_parity_prep()
print(report.summary_text())
"
```
