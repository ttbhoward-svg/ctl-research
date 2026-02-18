# Task 12 Assumptions — Chart Study Session Infrastructure

## Scope

1. **No model or scoring changes.** This module creates review workflows only.
   It reads the assembled dataset (Task 8 output) and regression scores but
   never modifies them.

2. **Trade identity.** The assembled dataset has no explicit trade ID column.
   A deterministic ID is computed as `{Date}_{Ticker}_{Timeframe}_{SetupType}`
   (ISO date, underscores replacing `/`). This is unique within a single
   assembled dataset.

3. **Score column.** When the user requests selection "by score", this refers
   to the `ScoreBucket` column (set by regression Task 10) or a raw score
   column if present. If `ScoreBucket` is null/missing for all rows, score-based
   selection falls back to `TheoreticalR`.

## Selection Logic

4. **Top/bottom N.** `select_top_n` and `select_bottom_n` sort by the chosen
   column (default `TheoreticalR`) and return the first/last N rows. Ties are
   broken by Date (earliest first) for determinism.

5. **Stratified sample.** `select_stratified` splits the dataset into terciles
   using the same `assign_terciles` logic from `regression.py` on the `by`
   column, then samples `n_per_tercile` trades from each group. Sampling uses
   `DataFrame.sample(n, random_state=seed)` for determinism.

6. **Filtering.** `filter_dataset` applies optional filters (symbol, timeframe,
   date range, score bucket) before any selection. All filters are AND-combined.

## Output Artifacts

7. **Study queue format: JSON.** A JSON file containing a list of trade records
   with trade_id, selection metadata, and key trade fields. JSON is chosen over
   CSV because the observation schema has nested/typed fields that CSV handles
   poorly.

8. **Notes template format: Markdown.** A single markdown file with one section
   per trade, pre-filled with trade context and empty observation fields.
   This is for human review convenience.

9. **Observation schema fields (fixed).** Per the task spec:
   - `setup_quality` (int 1-5)
   - `pullback_character` (str, constrained choices: textbook/deep/shallow/choppy/other)
   - `volatility_context` (str, constrained choices: low/normal/high/extreme)
   - `regime_alignment_note` (str, free text)
   - `execution_quality_note` (str, free text)
   - `post_trade_reflection` (str, free text)

10. **Observation reloading.** Completed observations are stored as JSON
    (one file per study session) so they can be parsed programmatically for
    later analysis.

## Implementation

11. **Pure functions.** All selection and generation functions are pure
    (no side effects except file writes). File I/O is isolated to explicit
    `save_*` functions.

12. **No external dependencies beyond pandas/numpy.** No charting or
    visualization in this module (that lives in `chart_inspector.py`).

13. **Module docstring includes usage examples** per task spec.
