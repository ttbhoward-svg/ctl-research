# Task F Assumptions — Continuous Series Builder

1. **Input format.** Databento OHLCV-1D CSV.ZST files per outright contract.
   Columns: `ts_event, rtype, publisher_id, instrument_id, open, high, low,
   close, volume, symbol`. One file per contract month (e.g. `ESH5.csv.zst`).

2. **Contract code parsing.** Symbol column values like `ESH5` parse to:
   root=`ES`, month=`H` (March), year=`5` (2025). Standard futures month
   codes: F G H J K M N Q U V X Z → Jan–Dec.

3. **Year disambiguation.** Single-digit year codes in data range 2018-01-01
   to 2026-02-17 map: 8→2018, 9→2019, 0→2020, 1→2021, ... 6→2026.

4. **Contract ordering.** Contracts are sorted by expiration order
   (year × 12 + month_index) to determine front/next chain.

5. **Roll trigger.** Volume-based: roll from contract A to contract B when
   B's volume > A's volume for 2 consecutive trading days. Roll takes effect
   on the second day of the crossover.

6. **Panama-style back-adjustment.** Additive. At each roll, compute
   `diff = new_close - old_close` on the roll date. Subtract this cumulative
   diff from all historical OHLC prices (backwards). Volume is NOT adjusted.

7. **Output schema.** `Date, Open, High, Low, Close, Volume, contract,
   adjustment`. Date is YYYY-MM-DD. contract = active contract symbol at that
   bar. adjustment = cumulative additive adjustment applied.

8. **Roll log.** One row per roll event: `date, from_contract, to_contract,
   from_close, to_close, adjustment, cumulative_adjustment`.

9. **Three symbols.** ES, CL, PA — per task constraints.

10. **Decompression.** Uses pandas `read_csv` with `.csv.zst` which pandas
    handles natively via zstandard.

11. **Missing days.** If a contract has no bar for a given date, that date is
    skipped for that contract. Only dates present in the front contract
    appear in the continuous series.

12. **No threshold changes.** Roll trigger (2 consecutive days) and adjustment
    method (additive Panama) are fixed.
