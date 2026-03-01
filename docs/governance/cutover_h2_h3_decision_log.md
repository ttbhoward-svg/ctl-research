# Cutover H.2/H.3 Decision Log

## Purpose
Record data-cutover reconciliation decisions, rationale, and next-gate criteria in a stable governance document (separate from volatile run artifacts in `data/processed/`).

## Policy
- Raw/generated diagnostics artifacts (CSV/JSON under `data/processed/`) are operational outputs and are not committed.
- Decision outcomes and rationale are captured here and committed.
- Any cutover status change (NO-GO/WATCH/CONDITIONAL GO/GO) requires a new dated entry.

## Current Status Snapshot
- Date: 2026-02-28
- Scope: H.2 (DP alignment) + H.3 (CL roll-policy calibration)
- Overall cutover state: **NO-GO (strict)**, **WATCH (policy/bridge mode)**
- Canonical interpretation: TradeStation remains a bridge diagnostic reference, not production truth.

## Decision Entry — H.2/H.3

### Inputs
- ES diagnostics after H.2:
  - `strict_status=FAIL`
  - `policy_status=WATCH`
  - `n_canonical=32`, `n_ts=32`, `n_paired=31`, `n_fail=2`
  - Day-delta histogram: `{2: 21, 1: 10}`
- CL diagnostics before H.3:
  - `strict_status=FAIL`
  - `policy_status=FAIL`
  - `n_canonical=98`, `n_ts=96`, `n_paired=71`, `n_fail=52`
- CL calibration result (H.3 best variant):
  - Variant: `consecutive_days=2`, `eligible_months=all`, `roll_timing=next_session`
  - CL rerun outcome:
    - `strict_status=FAIL`
    - `policy_status=WATCH`
    - `n_paired=90`, `n_fail=14`
    - Unmatched: `canonical=8`, `ts=6`

### Decision
- Accept H.3 recommended CL variant for bridge diagnostics:
  - `cd=2`, `all months`, `next_session`, `convention=add` (parity mode).
- Maintain **strict NO-GO** for full cutover until strict criteria pass.
- Maintain **policy WATCH** bridge status for ES and CL pending next reconciliation stage.

### Rationale
- H.2 fixed matching-policy brittleness and produced interpretable roll alignment diagnostics.
- H.3 materially reduced CL mismatch burden (paired rolls up, fails down).
- Remaining divergence is concentrated in roll schedule alignment quality, not broad pipeline instability.

## Next Gate (H.4 Target)
1. Run Task D parity rerun for ES/CL with current recommended policies and record deltas.
2. Run no-roll controls (AAPL, XLE) to isolate non-roll data integrity issues.
3. Define explicit cutover threshold for moving from policy WATCH to CONDITIONAL GO.
4. Update this memo with a dated H.4 decision entry.

## Decision Entry - 2026-03-01 (H.4 Validation Outcome)

### Scope
- H.4 cross-provider validation using Norgate exports as an additional external reference.

### Decision
- Databento canonical series remains the system-of-record for cutover v1.
- TradeStation and Norgate remain external validation references, not production truth sources.

### Why
- Strict cross-provider parity is structurally limited by:
  - futures continuous roll policy differences and path-dependent back-adjustment drift,
  - session convention differences,
  - equity/ETF adjustment-basis differences (raw vs adjusted).

### Evidence Snapshot
- ES: trigger parity aligns in validator comparisons; EMA/trade strict thresholds still fail.
- CL: strict divergence persists, but H.3 policy status improved to WATCH under calibrated roll timing.
- AAPL/XLE: large divergence indicates normalization-basis mismatch, not missing files.

### Governance Rule (effective 2026-03-01)
- Keep dual gates:
  - `strict_status` for hard comparability diagnostics.
  - `policy_status` for operational progression decisions.
- Progression may continue under policy WATCH only when divergence is explainable and documented.

### Source-of-Truth Rule
- Futures (ES/CL/PL): canonical Databento continuous builder plus roll manifest.
- Equities/ETFs (AAPL/XLE): parity interpretation requires explicit normalization mode metadata.

### Next Required Engineering Action
- Implement explicit normalization modes in code and rerun the parity matrix under declared modes before final cutover v1 lock.

## Future Entry Template
### Decision Entry — YYYY-MM-DD
- Scope:
- Inputs:
- Decision:
- Rationale:
- Gate impact:
- Next actions:
