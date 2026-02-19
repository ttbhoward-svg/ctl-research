# Task H Assumptions — Roll Manifest Alignment + Dual-Mode Parity

**Task**: Data Cutover Task H
**Date**: 2026-02-18
**Status**: In progress

---

## Assumptions

1. **Roll manifest is the single source of truth for Databento roll events.**
   The `_build_roll_log()` output from `continuous_builder.py` already captures per-roll diagnostics. Task H extends this into a full JSON manifest with additional metadata fields (convention, session_template, close_type, trigger_reason, confirmation_days).

2. **TradeStation roll events can be inferred from step changes in the unadjusted close series.**
   When TS rolls from one contract to another, the unadjusted close-to-close difference exceeds a configurable tick threshold (`min_gap_ticks`). This heuristic is sufficient for L2 comparison; false positives are acceptable and flagged as WATCH.

3. **Roll date alignment tolerance is ±2 trading days.**
   The `max_day_delta` parameter defaults to 2. Rolls matched within this window are PASS; rolls shifted by 1–2 days are WATCH; unmatched rolls are FAIL.

4. **PASS/WATCH/FAIL is the status vocabulary for Task H.**
   This differs from Task C reconciliation (OK/WATCH/ALERT) and Task D parity (passed: bool). Task H uses PASS/WATCH/FAIL to indicate: PASS = aligned, WATCH = minor discrepancy (within tolerance), FAIL = misaligned or unmatched.

5. **Dual-mode build labels are metadata only; they do not change computation.**
   `parity_mode` labels output as TradeStation-bridge calibration (convention="add"). `canonical_mode` labels output as production (convention="subtract"). The label is stored in the manifest but does not alter any logic.

6. **L2 diagnostics compare roll schedule alignment.**
   Output: per-roll-pair match status (PASS/WATCH/FAIL), day delta, contract identifiers.

7. **L3 diagnostics compare roll gap magnitudes.**
   Output: per-matched-roll gap from canonical vs gap from TS, absolute difference, relative difference.

8. **L4 diagnostics measure adjusted series drift.**
   Close-to-close absolute difference between canonical adjusted and TS adjusted series, computed on overlapping dates. Drift explanation identifies which roll boundaries contribute most to cumulative error.

9. **L4 drift explanation is a JSON artifact.**
   It reports per-roll-interval mean drift, max drift, and cumulative contribution to overall series divergence. This is diagnostic only — no gate or threshold decisions.

10. **No changes to Task D thresholds or gate constants.**
    `EMA_MAX_DIVERGENCE_PCT`, `R_DIFF_THRESHOLD`, and all other cutover_parity.py constants remain unchanged.

11. **No changes to strategy logic.**
    B1 detector, simulator, and scoring modules are not touched.

12. **Roll manifest JSON is per-symbol, not cross-symbol.**
    Each symbol gets its own `{symbol}_roll_manifest.json` file. The combined roll log CSV remains as-is.

---

## File Plan

| File | Action |
|------|--------|
| `docs/notes/TaskH_assumptions.md` | NEW — this file |
| `src/ctl/roll_reconciliation.py` | NEW — roll manifest loading, TS roll derivation, schedule comparison, step-change explanation |
| `src/ctl/continuous_builder.py` | EDIT — add `export_roll_manifest()`, `BuildMode` type, extend `ContinuousResult` |
| `src/ctl/cutover_diagnostics.py` | NEW — L2/L3/L4 diagnostic orchestrator |
| `tests/unit/test_roll_reconciliation.py` | NEW — unit tests |
| `tests/unit/test_cutover_diagnostics.py` | NEW — unit tests |

---

## Run Instructions

```bash
# After implementation
.venv/bin/python -m pytest tests/unit/test_roll_reconciliation.py -v
.venv/bin/python -m pytest tests/unit/test_cutover_diagnostics.py -v
.venv/bin/python -m pytest -v  # full suite
```
