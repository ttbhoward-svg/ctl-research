# Phase 1a Tracker Delta Audit (One-Page)

Date: 2026-03-02  
Scope: Reconcile current repo state vs `CTL_Phase1a_Project_Tracker_v3` task list and clarify where cutover hardening fits.

## Executive Status

- Phase 1a Tasks 1-15 are implemented and closed in governance artifacts:
  - `docs/governance/phase1a_completion_snapshot.md`
  - `docs/governance/phase1a_closeout_report.md`
- Current H-series (H.2+) is post-Phase1a cutover hardening, not a replacement for Phase1a scope.
- Broad-spec momentum can continue by treating cutover as operational hardening while starting Phase1b-style expansion in parallel.

## Task Crosswalk (1-15)

| Tracker Task | Current Evidence | Audit Status |
|---|---|---|
| 1 | `src/ctl/universe.py`, `src/ctl/data_loader.py`, `src/ctl/providers/*`, `src/ctl/symbol_map.py` | Complete |
| 2 | `src/ctl/b1_detector.py`, `tests/unit/test_b1_detector.py` | Complete |
| 3 | `src/ctl/simulator.py`, `src/ctl/chart_inspector.py`, tests | Complete |
| 4 | `src/ctl/reconciliation.py`, `src/ctl/cutover_parity.py`, `src/ctl/roll_reconciliation.py` | Complete (extended by Task H/H-series) |
| 4b | `src/ctl/slippage_stress.py`, `tests/unit/test_slippage_stress.py` | Complete |
| 5 | MTFA flags in `src/ctl/b1_detector.py` (weekly + monthly logic present), tests in `test_b1_detector.py` | Complete in code |
| 6 | Confluence logic in detector + `tests/unit/test_confluence.py` | Complete |
| 7 | `src/ctl/cot_loader.py`, `src/ctl/vix_loader.py`, `src/ctl/external_merge.py` + tests | Complete |
| 8 | `src/ctl/dataset_assembler.py`, `src/ctl/health_check.py` + tests | Complete |
| 9 | `docs/governance/pre_registration_v1.md`, config locks | Complete |
| 10 | `src/ctl/regression.py`, model card artifact | Complete |
| 11/11b/11c | `param_sensitivity.py`, `negative_controls.py`, `entry_degradation.py` + tests | Complete |
| 12 | `src/ctl/chart_study.py` + tests | Complete |
| 13 | `src/ctl/oos_evaluation.py` + tests | Complete |
| 14 | `src/ctl/gate_decision.py` + governance format docs | Complete |
| 15 | `src/ctl/drift_monitor.py`, `src/ctl/archive.py` + tests | Complete |

## Where Cutover Hardening Fits

Cutover stream (H.2-H.30) is best classified as:

- Task 4 extension: cross-provider futures roll/diagnostics hardening (`L2/L3/L4`, overlap enforcement, acceptance framework).
- Task 14/15 operationalization: locked operating profile + gate-first execution + weekly ops + notifications + closeout verification.

Primary files:
- `configs/cutover/operating_profile_v1.yaml`
- `scripts/check_operating_profile.py`
- `scripts/run_weekly_b1_portfolio.py`
- `scripts/run_weekly_ops.py`
- `src/ctl/operating_profile.py`
- `src/ctl/run_orchestrator.py`
- `src/ctl/ops_notifier.py`

## Material Deltas / Residuals

1. Tracker version drift
- Two tracker variants exist (`docs/specs/locked/...v3.md` and `docs/governance/...v3.md` with v3.1/API-first notes).
- Action: maintain one canonical tracker reference to prevent scope ambiguity.
- Status update: canonical tracker explicitly designated; historical tracker marked provenance-only.

2. `run_phase1a.py` orchestration
- Still placeholder (already noted in completion snapshot).
- Action: either wire it to current module workflow or officially deprecate and point to cutover runner scripts.
- Status update: deprecated fail-fast guard added with canonical entrypoint guidance.

3. Validation provenance format
- Code/tests are complete, but manual spot-check proofs referenced in tracker prose are scattered across logs.
- Action: add one index doc linking exact verification artifacts/commands for auditability.

4. Detour governance
- Need explicit place to park non-critical side work and revisit triggers.
- Status update: `docs/governance/detour_backlog.md` added with priority rubric (`P0..P3`) and active deferred items.

## Recommended Next Step (Aligned)

1. Keep cutover profile locked and continue weekly operations (current stable path).  
2. Start research-tier onboarding/backtest expansion (non-gating) as Phase1b preparation.  
3. Promote symbols/features only on walk-forward stability, not in-sample fit.

This keeps momentum on the broader spec while preserving the risk controls gained in cutover hardening.
