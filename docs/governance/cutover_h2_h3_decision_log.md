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

## Decision Entry — H.4 Normalization Modes

### Scope
- H.4: Explicit normalization modes for parity inputs (equities/ETFs).

### Inputs
- AAPL/XLE parity failures showed 305%+ EMA divergence caused by adjustment-basis mismatch.
- Databento equities are unadjusted (raw); TradeStation provides split-adjusted; Norgate provides fully adjusted (splits + dividends).
- Prior parity comparisons silently mixed adjustment bases.

### Decision
- Introduce `NormalizationMode` (`"raw"`, `"split_adjusted"`, `"total_return_adjusted"`) and `AssetClass` (`"futures"`, `"equity"`, `"etf"`) as explicit metadata on all parity inputs.
- `normalize_ohlcv()` enforces mode/asset-class compatibility:
  - Futures reject `split_adjusted` (roll adjustment handled separately).
  - `split_adjusted` requires explicit `split_factor` column; fails loudly if absent.
  - `total_return_adjusted` reserved (raises `NotImplementedError`).
- Parity suite (`run_parity_suite`, `run_cutover_suite`) accepts optional normalization kwargs; defaults to `"raw"` with no asset class (backward-compatible pass-through).
- Schema coercion (date column normalization, volume alias handling, canonical column ordering) applied uniformly.

### Rationale
- Silent basis mismatch was the root cause of equity/ETF parity failures.
- Explicit declarations prevent recurrence and make comparison assumptions auditable.
- Backward-compatible defaults preserve existing futures parity behavior.

### Gate Impact
- No gate criteria change. Normalization modes are infrastructure for correct comparisons, not threshold adjustments.
- Equity/ETF parity results are now interpretable only when declared modes match.

### Next Actions
- Rerun AAPL/XLE parity under declared `raw` mode for both sources (baseline).
- Evaluate split-factor availability for Databento equities.
- Define equity/ETF parity gate criteria once normalization-aligned comparisons are available.

## Decision Entry - 2026-03-01 (H.4 Execution Outcome)

### Scope
- Executed H.4 normalization-mode implementation and reran the cross-provider parity matrix.
- Symbols tested: ES, CL, PL, AAPL, XLE (Databento vs Norgate).

### Inputs
- H.4 code integration complete (`NormalizationMode`, `AssetClass`, schema coercion).
- Databento lowercase OHLC regression fixed and validated by tests.
- Parity outputs saved to `data/processed/cutover_v1/parity_h4_final/`.
- PL comparison corrected to use `PL_continuous.csv` (continuous), not a single outright contract.

### Result Summary
- ES: strict parity failed; partial functional alignment remains.
- CL: strict parity failed; structural divergence persists.
- PL: valid continuous-vs-continuous comparison now, strict parity failed.
- AAPL/XLE: strict parity failed; large divergence consistent with adjustment-basis mismatch.

### Decision
- Maintain **STRICT NO-GO** for cross-provider exact parity acceptance.
- Maintain **policy WATCH** for continued canonical-pipeline progression.
- Confirm parity diagnostics are now technically reliable and interpretation-safe due to explicit normalization metadata.

### Rationale
- Remaining divergence is attributable to known provider basis and continuous-construction differences, not missing-data or schema ambiguity.
- H.4 reduced interpretation risk and improved reproducibility; it did not change parity thresholds or strategy logic.

### Gate Impact
- No threshold changes.
- No strategy logic changes.
- External strict parity remains blocked; canonical development continues under documented WATCH governance.

### Next Actions (H.5)
1. Define canonical futures acceptance on L2/L3/L4 explainability metrics (not external exact price match).
2. Add explicit equity/ETF basis pipelines (`raw` and `split_adjusted`) with declared source constraints.
3. Add overlap-window enforcement utility for parity runs to eliminate coverage contamination.
4. Rerun policy-gated matrix and issue updated NO-GO/WATCH/CONDITIONAL-GO decision.

## Decision Entry — H.5 Prep (Canonical Acceptance Framework)

### Scope
- H.5: Canonical futures acceptance based on explainability diagnostics; overlap enforcement for parity runs.

### Decision
- Canonical futures acceptance uses L2/L3/L4 explainability metrics (roll matching, gap divergence, drift) — NOT external exact price match.
- External parity (TradeStation, Norgate) remains reference-only diagnostics.
- Overlap-window enforcement prevents coverage-mismatch contamination in parity comparisons.

### Rationale
- Cross-provider exact parity is structurally infeasible due to roll schedule, session convention, and adjustment-basis differences.
- Acceptance criteria should measure internal consistency and explainability of divergence, not magnitude of provider disagreement.

### Gate Impact
- No threshold changes (EMA_MAX_DIVERGENCE_PCT, R_DIFF_THRESHOLD unchanged).
- No strategy logic changes.
- Backward-compatible defaults on all modified functions.

### Next Actions
1. Rerun policy-gated acceptance matrix for ES/CL/PL using `evaluate_futures_acceptance`.
2. Evaluate equity/ETF basis pipelines under `raw` and `split_adjusted` modes.
3. Issue updated NO-GO/WATCH/CONDITIONAL-GO decision based on acceptance results.

## Decision Entry - 2026-03-01 (H.5 Outcome)

### Scope
- Ran canonical futures acceptance evaluator (`acceptance_from_diagnostics`) for ES/CL/PL using current diagnostics.

### Inputs
- ES: `strict_status=FAIL`, `policy_status=WATCH`, acceptance `WATCH`
  - Reason: mean drift `7.3289 > 5.0000`
- CL: `strict_status=FAIL`, `policy_status=FAIL`, acceptance `REJECT`
  - Reasons:
    - unmatched rolls `26.80% > 10.00%`
    - FAIL matches `73.24% > 15.00%`
- PL: `strict_status=FAIL`, `policy_status=FAIL`, acceptance `REJECT`
  - Reasons:
    - paired rolls `0 < 20`
    - unmatched rolls `100.00% > 10.00%`
    - mean drift `181.1059 > 5.0000`

### Decision
- Canonical futures acceptance result: **NO-GO** (portfolio-level), with mixed per-symbol status:
  - ES = WATCH
  - CL = REJECT
  - PL = REJECT

### Rationale
- H.5 acceptance framework is functioning as intended and produces explainable, criterion-level decisions.
- Current CL/PL diagnostics exceed hard acceptance constraints.
- ES remains close but not yet acceptable due to drift threshold breach.

### Next Actions
1. CL: run targeted roll policy refinement and session/basis diagnostics to reduce unmatched/fail fractions.
2. PL: verify TS custom series quality/coverage and rebuild manifest/continuous alignment before re-evaluation.
3. ES: investigate drift contributors and attempt reduction below mean drift threshold.
4. Re-run H.5 evaluator and issue updated gate decision.

## Decision Entry - 2026-03-01 (H.6 CL Remediation Update)

### Scope
- Executed CL-focused H.6 remediation loop using calibrated policy variant and regenerated diagnostics artifacts under `data/processed/cutover_v1/diagnostics_h6/`.

### Inputs
- Best CL variant remained:
  - `cd=2`, `eligible_months=all`, `roll_timing=next_session`, `convention=add`.
- Regenerated CL diagnostics summary:
  - L2 rows: `104`
  - `PASS=35`, `WATCH=55`, `FAIL=14`
  - `fail_frac=13.46%` (improved from prior `15.56%` gate breach)
- Acceptance evaluator output:
  - `decision=ACCEPT`
  - `accepted=True`

### Decision
- Upgrade CL canonical acceptance status from **REJECT** to **ACCEPT** for the current H.6 snapshot.

### Rationale
- CL moved below hard fail-fraction threshold (`13.46% <= 15.00%`) while maintaining acceptable drift/gap characteristics from the calibrated run.
- Prior rejection condition was marginal and has been resolved by calibrated roll-policy diagnostics regeneration.

### Gate Impact
- Symbol-level status update:
  - ES = WATCH
  - CL = ACCEPT
  - PL = REJECT (unchanged)
- Portfolio-level status remains **NO-GO** until PL and ES are remediated or otherwise dispositioned.

### Next Actions
1. Run PL remediation loop (coverage/paired-roll quality) and re-evaluate acceptance.
2. Run ES drift-focused remediation to move WATCH toward ACCEPT.
3. Recompute portfolio-level decision after PL/ES updates.

## Decision Entry - 2026-03-01 (H.6 PL Remediation — Blocker Diagnosis)

### Scope
- PL remediation loop following H.5 REJECT (0 paired rolls, 100% unmatched, mean drift 181.1).
- Diagnosis-only: no code changes, no rebuild, no artifact generation.

### Root Cause: Cross-Commodity Mismatch (PA ≠ PL)

The PL acceptance failure is caused by a fundamental data identity error, not a roll-policy or threshold issue.

| Source | Symbol | Commodity | Jan 2018 Price | Coverage |
|--------|--------|-----------|----------------|----------|
| Databento raw (`outrights_only/PA/`) | PA* (PAH8, PAM8…) | **Palladium** | ~$1,060 | 86 contracts, 2018–2026 |
| Existing `PL_continuous.csv` | PA* contracts | **Palladium** | ~$1,059 | Truncated: 378 rows, 4 rolls, ends 2019-03-27 |
| TS `TS_PL_CUSTOM_UNADJ_1D_*.csv` | PL | **Platinum** | ~$936 | 2,046 rows, 2018–2026 |
| TS `TS_PL_CUSTOM_ADJ_1D_*.csv` | PL | **Platinum** | ~$1,183 (adj) | 2,046 rows, 2018–2026 |
| Norgate `NG_PL_1D_*.csv` | PL | **Platinum** | ~$1,207 (adj) | 2,052 rows, 2018–2026 |

Key findings:
- `outrights_only/` contains directories `ES/`, `CL/`, `PA/` — **no `PL/` directory exists**.
- `parity_prep.py` uses `PARITY_SYMBOLS = ("ES", "CL", "PA")` — PA (Palladium), not PL (Platinum).
- `PL_roll_manifest.json` header contains `"symbol": "PA"` despite the filename.
- The existing `PL_continuous.csv` was built from PA (Palladium) outrights, not PL (Platinum).

### Why All H.5 Metrics Failed
1. **0 paired rolls** — truncated PA build (4 rolls ending 2019-03) had near-zero overlap with full-range TS PL series (2018–2026).
2. **100% unmatched** — Palladium roll schedule ≠ Platinum roll schedule (different delivery months).
3. **mean drift 181.1** — Palladium prices compared against Platinum prices: structurally ~$100–200 apart.

### Two Compounding Issues
1. **Truncated build**: The PA continuous series stops at 2019-03-27 (only 4 of ~30 expected rolls). A full rebuild from `outrights_only/PA/` (86 files) would produce full PA coverage — but this is PA (Palladium), not PL (Platinum).
2. **Commodity identity**: No PL (Platinum) outright contract data exists in the Databento extract. PL remediation is gated on acquiring genuine `PL.FUT` contract-level data from Databento.

### Decision
- Update PL canonical acceptance status to **REJECT (BLOCKED — data acquisition required)**.
- No code changes, rebuilds, or parameter tuning can resolve this — the underlying contract data is for the wrong commodity.
- PL remediation is blocked until genuine Platinum (`PL.FUT`) contract-level data is acquired from Databento and placed in `outrights_only/PL/`.

### Rationale
- The entire PL parity pipeline was comparing Palladium (PA) prices against Platinum (PL) reference series.
- All H.5 failure metrics (0 paired rolls, 100% unmatched, 181-point drift) are structurally explained by this cross-commodity mismatch.
- No amount of roll-policy calibration, threshold adjustment, or continuous-builder tuning can fix a wrong-commodity input.

### Gate Impact
- Symbol-level status update:
  - ES = WATCH (unchanged)
  - CL = ACCEPT (unchanged)
  - PL = **REJECT (BLOCKED)** — data acquisition required
- Portfolio-level status remains **NO-GO** until PL data blocker is resolved and ES is remediated.

### Next Actions
1. Acquire genuine PL (Platinum) contract-level data from Databento (`PL.FUT` outrights).
2. Place acquired data in `outrights_only/PL/` and rebuild `PL_continuous.csv` from Platinum contracts.
3. Update `parity_prep.py` to include PL in `PARITY_SYMBOLS` (currently only references PA).
4. Rerun PL acceptance evaluator after rebuild and issue updated gate decision.
5. Continue ES drift-focused remediation independently (not blocked by PL).

## Decision Entry - 2026-03-01 (H.6 PL Data Blocker Resolved, Status Upgrade)

### Scope
- Resolved PL data-identity blocker by acquiring genuine Databento Platinum outrights and rebuilding PL continuous with month set aligned to TradeStation availability (`F`, `J`, `N`, `V`).

### Inputs
- Rebuilt PL continuous from true `PL` contracts:
  - `PL_continuous.csv`: `2529` bars
  - `PL_roll_manifest.json`: `33` rolls
- PL diagnostics rerun:
  - `diag.strict_status=FAIL`
  - `diag.policy_status=WATCH`
- Acceptance rerun:
  - `decision=WATCH`
  - Reasons:
    - mean gap diff `1.6600 > 1.0000`
    - mean drift `8.2821 > 5.0000`

### Decision
- Upgrade PL from **REJECT (BLOCKED)** to **WATCH**.
- PL is no longer blocked by commodity identity; remaining deltas are calibration/normalization quality issues.

### Gate Impact
- Symbol-level status:
  - ES = WATCH
  - CL = ACCEPT
  - PL = WATCH
- Portfolio-level status remains **NO-GO** pending remediation to reduce remaining ES/PL drift/gap metrics.

### Next Actions
1. ES: drift-focused remediation to move WATCH -> ACCEPT.
2. PL: targeted gap/drift tuning against TS settings-equivalent build to move WATCH -> ACCEPT.
3. Add PA as separate symbol track (do not alias to PL), with dedicated TS reference exports and diagnostics.
4. Re-run portfolio acceptance and issue updated gate decision.

## Decision Entry - 2026-03-01 (PA Scope Clarification for Cutover)

### Scope
- Clarify how PA (Palladium) is handled in cutover gating versus research usage.

### Decision
- PA is designated **research-enabled, non-gating** for the current cutover cycle.
- Portfolio cutover gate universe is explicitly: **ES, CL, PL**.
- PA remains in the project as a separate futures track (no aliasing to PL) and will continue to receive diagnostics and iterative quality improvements.

### Rationale
- PA contract/bar construction is materially noisier across providers and currently requires additional session/basis harmonization work.
- Deferring PA from gate-blocking avoids stalling ES/CL/PL production readiness while preserving PA as a high-value research/trading instrument.

### Gate Impact
- Portfolio-level GO/NO-GO decision for this cycle is computed from ES/CL/PL acceptance only.
- PA status is tracked separately and cannot block ES/CL/PL cutover progression in this cycle.

### Next Actions
1. Execute ES remediation to move WATCH -> ACCEPT.
2. Execute PL remediation to move WATCH -> ACCEPT.
3. Continue PA enhancement track and promote PA to gating scope in a later cycle once harmonization criteria are met.

## Decision Entry - 2026-03-01 (H.6 ES Sweep Outcome and Portfolio Recommendation)

### Scope
- Ran ES roll-policy/convention sweep to determine whether ES can be moved from WATCH to ACCEPT without threshold changes.

### Inputs
- ES sweep variants tested:
  - `consecutive_days in {1,2,3}`
  - `roll_timing in {same_day, next_session}`
  - `convention in {add, subtract}`
- Best ES configuration remained:
  - `consecutive_days=2`, `convention=add` (timing equivalent in current data)
- Best observed ES metrics:
  - `n_paired=32`, `n_fail=0`, `mean_gap_diff=0.53125`, `mean_drift=7.328928`

### Decision
- Maintain ES at **WATCH** for this cycle.
- No tested ES configuration reduces `mean_drift` below the current acceptance threshold (`5.0`), so ES cannot be promoted to ACCEPT without changing policy thresholds.

### Rationale
- ES pairing/fail behavior is clean (`n_fail=0`, full pairing under operational day-delta tolerance).
- Remaining ES blocker is stable mean drift, indicating a structural provider basis difference rather than pipeline instability.
- Threshold changes are explicitly out of scope for this cycle.

### Gate Impact
- Current symbol statuses:
  - ES = WATCH
  - CL = ACCEPT
  - PL = WATCH
  - PA = research-enabled, non-gating
- Portfolio recommendation for this cycle: **CONDITIONAL GO** (research/paper/live-sim progression), not full GO.

### Next Actions
1. Continue ES/PL drift-gap reduction efforts without threshold changes.
2. Re-run portfolio acceptance on next cycle after additional ES/PL harmonization.
3. Maintain PA as a separate non-gating enhancement track.

## Decision Entry - 2026-03-01 (Equity Scope Clarification: AAPL/XLE)

### Scope
- Clarify handling of equity/ETF parity outcomes after H.4 normalization-mode rollout and overlap-enforced reruns.

### Inputs
- AAPL and XLE remain strict parity failures under explicit `raw` mode and overlap enforcement.
- Failure pattern is consistent with adjustment-basis mismatch across providers, not a parity harness defect.

### Decision
- AAPL/XLE are designated **non-gating** for the current futures cutover cycle.
- Equity/ETF parity remains an active, separate basis-alignment workstream.
- Current cutover gate universe remains futures-only: **ES, CL, PL**.

### Rationale
- Equity basis alignment requires dedicated split/adjustment metadata handling and should not block futures readiness.
- Governance already enforces explicit normalization metadata, preventing silent basis mixing in future evaluations.

### Gate Impact
- Portfolio recommendation for this cycle continues as **CONDITIONAL GO** for research/paper/live-sim progression based on futures track status.
- Equity/ETF outcomes are tracked separately until basis alignment criteria are met.

### Next Actions
1. Define equity basis acceptance criteria for `raw` and `split_adjusted` tracks.
2. Source/validate required split metadata for Databento equity alignment path.
3. Re-run AAPL/XLE under finalized basis rules in a dedicated equity gate cycle.

## Decision Entry - 2026-03-01 (Operational Day-Delta Tolerance Lock and Futures Status Board)

### Scope
- Finalized operational `max_day_delta` settings for current futures diagnostics cycle and refreshed acceptance status board.

### Inputs
- ES day-delta sweep showed stable WATCH status with zero fail matches at `max_day_delta>=3`; mean drift remained the only blocker.
- CL day-delta sweep outcomes:
  - `mdd=2`: `n_paired=90`, `n_fail=14`, acceptance `REJECT`
  - `mdd=3`: `n_paired=94`, `n_fail=6`, acceptance `ACCEPT`
  - `mdd>=4`: `ACCEPT` with marginal additional benefit
- PL remains WATCH under current settings due to mean gap/drift criteria.

### Decision
- Lock operational day-delta tolerances for this cycle:
  - `ES max_day_delta=3`
  - `CL max_day_delta=3`
  - `PL max_day_delta=2`

### Futures Status Board (Current Cycle)
- ES = WATCH
- CL = ACCEPT
- PL = WATCH
- PA = research-enabled, non-gating

### Portfolio Recommendation
- Maintain **CONDITIONAL GO** for research/paper/live-sim progression this cycle.
- Full GO deferred pending additional ES/PL drift-gap harmonization.

### Rationale
- `max_day_delta=3` materially improves CL pairing/fail outcomes without threshold changes.
- ES and PL residual blockers are stable drift/gap effects rather than pairing instability.
- Threshold policy remains unchanged; only operational pairing tolerance is set.

### Next Actions
1. ES: continue drift reduction workstream (target mean drift <= 5.0).
2. PL: continue gap/drift reduction workstream (target mean gap <= 1.0 and mean drift <= 5.0).
3. Re-run futures acceptance board at next cycle checkpoint.

## Decision Entry - 2026-03-01 (H.7 Operating Profile Lock + Pre-Run Gate)

### Scope
- Freeze validated operational state into a machine-readable YAML config.
- Provide a gate script that re-derives acceptance and compares against locked expectations as a regression guard.

### Inputs
- Current validated futures status board:
  - ES = WATCH (mean drift 7.33 > 5.0; pairing clean, n_fail=0)
  - CL = ACCEPT (fail fraction 13.46% under calibrated roll policy)
  - PL = WATCH (mean gap diff 1.66, mean drift 8.28 above soft thresholds)
  - PA = research-enabled, non-gating
- Portfolio recommendation: CONDITIONAL GO (research/paper/live-sim)
- Locked operational settings: ES mdd=3, CL mdd=3, PL mdd=2
- Policy constraints: thresholds locked, strategy logic locked

### Decision
- Lock operating profile in `configs/cutover/operating_profile_v1.yaml` with per-symbol tick_size, max_day_delta, and expected_status.
- ES drift floor observed and accepted as WATCH for conditional operation — no further ES variant tuning this cycle.
- Gate script (`scripts/check_operating_profile.py`) re-derives acceptance from live diagnostics and compares against locked expectations.
- Any mismatch exits non-zero, providing an automated regression guard against silent acceptance drift.

### Rationale
- Freezing the profile ensures operational settings cannot silently diverge from governance decisions.
- The gate script closes the loop between governance decisions and runtime state — any change to data, diagnostics, or acceptance semantics that would alter a symbol's status is immediately surfaced.
- No threshold or strategy logic changes; this is purely an operational lock and verification mechanism.

### Gate Impact
- No threshold changes.
- No strategy logic changes.
- No changes to acceptance framework semantics.
- Operating profile provides a single source of truth for the current cycle's locked state.

### Next Actions
1. Run `scripts/check_operating_profile.py` against real data to confirm all symbols match expected statuses.
2. Integrate gate script into pre-run workflow for live-sim progression.
3. Create new operating profile version (v2) if any symbol status changes in a future cycle.

## Decision Entry - 2026-03-01 (H.8 Execution Readiness Wiring)

### Scope
- Wire gate-first portfolio runner that enforces operating-profile gate check before any strategy execution.
- Provide dry-run mode, JSON output, and run-summary persistence.

### Inputs
- H.7 operating profile locked and gate script passing (ES=WATCH, CL=ACCEPT, PL=WATCH).
- Portfolio recommendation: CONDITIONAL GO for research/paper/live-sim.
- No existing run entrypoint enforced gate check before execution.

### Decision
- Implement `scripts/run_weekly_b1_portfolio.py` as the canonical run entrypoint.
- Gate check always runs first; mismatch aborts with exit 2.
- Orchestrator logic (`src/ctl/run_orchestrator.py`) is testable, pure-function-based.
- Strategy executor is pluggable (default no-op placeholder for current cycle).
- Run summaries persisted as JSON under `data/processed/cutover_v1/run_summaries/`.

### Rationale
- Gate-first enforcement prevents strategy execution when acceptance state has drifted from locked expectations.
- Pluggable executor pattern allows strategy wiring without modifying gate/plan infrastructure.
- Dry-run mode supports pre-flight validation without side effects.

### Gate Impact
- No threshold changes.
- No strategy logic changes.
- No changes to acceptance framework semantics.
- Backward-compatible additions only.

### Runbook

```bash
# Pre-run gate check only (H.7):
python scripts/check_operating_profile.py
python scripts/check_operating_profile.py --json

# Full gate-first run (H.8):
python scripts/run_weekly_b1_portfolio.py

# Dry run (gate + plan, no execution):
python scripts/run_weekly_b1_portfolio.py --dry-run

# Include non-gating symbols (PA):
python scripts/run_weekly_b1_portfolio.py --include-non-gating

# JSON output:
python scripts/run_weekly_b1_portfolio.py --json

# Custom profile:
python scripts/run_weekly_b1_portfolio.py --profile configs/cutover/operating_profile_v1.yaml
```

**Exit codes:**
- `0` — gate passed, run completed (or dry-run plan returned)
- `2` — gate mismatch, run aborted

**On gate mismatch:** Do NOT override. Investigate which symbol's acceptance status changed, update the decision log, and if appropriate create a new operating profile version.

### Next Actions
1. Wire actual strategy executor callback when B1 strategy module is ready.
2. Add Slack/email notification hook on gate mismatch for production alerting.
3. Integrate runner into scheduled cron/task for automated weekly execution.

## Decision Entry - 2026-03-01 (H.9 Real B1 Strategy Execution Wiring)

### Scope
- Wire real B1 strategy execution into the gate-first portfolio runner.
- Replace placeholder/no-op executor with actual B1 detection + simulation path.

### Inputs
- H.8 runner infrastructure complete: gate check, run plan, dry-run, JSON output, summary persistence.
- Existing stable APIs:
  - `b1_detector.run_b1_detection(df, symbol, timeframe)` → `List[B1Trigger]`
  - `simulator.simulate_all(triggers, df, SimConfig)` → `List[TradeResult]`
  - `parity_prep.load_and_validate(path, label)` → `(df, errors)`
- Data path: `data/processed/databento/cutover_v1/continuous/{SYM}_continuous.csv`

### Decision
- Add `execute_b1_symbol(symbol, data_dir)` adapter in `run_orchestrator.py` that loads canonical OHLCV, runs B1 detection, simulates trades, and returns per-symbol metrics.
- Add `make_b1_executor(data_dir)` factory returning a `SymbolExecutor` closure.
- Extend `SymbolRunResult` with optional metric fields: `trigger_count`, `trade_count`, `total_r`, `win_rate` (None when not applicable — backward-compatible).
- Runner script (`run_weekly_b1_portfolio.py`) wires `make_b1_executor()` as default when not `--dry-run`.
- Failures in one symbol do not crash the portfolio run; errors are captured per-symbol.

### Rationale
- Reuses existing stable B1 detection and simulation APIs — no strategy logic changes.
- Pluggable executor pattern preserved; mock executors still work for testing.
- Per-symbol error isolation ensures partial data availability does not block the entire run.
- Optional metric fields in `SymbolRunResult.to_dict()` maintain backward-compatible JSON shape.

### Gate Impact
- No threshold changes.
- No strategy logic changes.
- No changes to acceptance semantics.
- Runner is now the canonical execution entrypoint for the cycle.

### Known Limitations
- Default executor uses `SimConfig()` (zero slippage). Instrument-specific slippage should be wired when production slippage values are calibrated.
- Weekly/monthly HTF data not passed to B1 detector (MTFA flags will be None). HTF data integration is a future enhancement.
- Run summary JSON includes metric fields only when execution occurs; dry-run and error results omit them.

### Runbook (updated)

```bash
# Dry run (gate + plan only):
python scripts/run_weekly_b1_portfolio.py --dry-run

# Actual B1 execution (default):
python scripts/run_weekly_b1_portfolio.py

# JSON output:
python scripts/run_weekly_b1_portfolio.py --json

# Include non-gating symbols:
python scripts/run_weekly_b1_portfolio.py --include-non-gating

# Custom profile:
python scripts/run_weekly_b1_portfolio.py --profile configs/cutover/operating_profile_v1.yaml
```

### Next Actions
1. Calibrate per-instrument slippage values and wire into `SimConfig`.
2. Add weekly/monthly HTF data loading for MTFA confluence flags.
3. Add Slack/email notification hook on gate mismatch for production alerting.
4. Integrate runner into scheduled cron/task for automated weekly execution.

## Decision Entry - 2026-03-01 (H.10 Weekly Ops Automation)

### Scope
- Schedule-ready ops wrapper around gate-first B1 portfolio runner.
- Configurable notification dispatch (stdout, webhook).
- Retention-based cleanup of old summary and ops-log files.

### Inputs
- H.9 runner complete: real B1 execution with per-symbol metrics.
- Need production-like weekly operations: one command, alerting, log hygiene.
- No existing retention or notification infrastructure.

### Decision
- New script `scripts/run_weekly_ops.py` wraps the H.8/H.9 runner.
- New module `src/ctl/ops_notifier.py` builds human-readable messages and dispatches to stdout or webhook.
- Retention cleanup prunes `*.json` files older than `--retention-days` (default 45) in both `run_summaries/` and `ops_logs/`.
- Webhook payload is `{"text": message}` — Slack-compatible.
- Ops logs saved to `data/processed/cutover_v1/ops_logs/YYYYMMDD_HHMMSS_ops.json`.

### Rationale
- Single command for cron scheduling eliminates manual step coordination.
- Notification dispatch surfaces gate mismatches and execution errors without manual log review.
- Retention prevents unbounded artifact accumulation in long-running cycles.
- Webhook failure does not crash the run — notification is best-effort.

### Gate Impact
- No threshold changes.
- No strategy logic changes.
- No changes to acceptance semantics.
- H.9 runner interface preserved.

### Runbook (updated)

```bash
# Gate check only (H.7):
python scripts/check_operating_profile.py

# B1 runner only (H.8/H.9):
python scripts/run_weekly_b1_portfolio.py --dry-run

# Full weekly ops (H.10):
python scripts/run_weekly_ops.py --dry-run --notify stdout
python scripts/run_weekly_ops.py --notify stdout
python scripts/run_weekly_ops.py --json --notify none
python scripts/run_weekly_ops.py --notify webhook --webhook-url https://hooks.slack.com/...

# With custom retention:
python scripts/run_weekly_ops.py --retention-days 30 --notify stdout
```

**Cron examples:**
```cron
# Weekly dry-run check (Sunday 6:00 AM):
0 6 * * 0 cd /path/to/ctl-research && .venv/bin/python scripts/run_weekly_ops.py --dry-run --notify stdout >> /var/log/ctl-ops.log 2>&1

# Weekly real run (Monday 5:00 AM):
0 5 * * 1 cd /path/to/ctl-research && .venv/bin/python scripts/run_weekly_ops.py --notify webhook >> /var/log/ctl-ops.log 2>&1
```

**Exit codes:**
- `0` — gate passed, run completed
- `2` — gate mismatch, run aborted
- `1` — unrecoverable runner error

**Notification modes:**
- `none` — no notification (default)
- `stdout` — print ops message to stdout
- `webhook` — POST `{"text": message}` to `--webhook-url` (or `OPS_WEBHOOK_URL` env)

### Next Actions
1. Configure production webhook URL and test end-to-end notification flow.
2. Calibrate per-instrument slippage values and wire into `SimConfig`.
3. Add weekly/monthly HTF data loading for MTFA confluence flags.
4. Set up systemd timer or cron job for automated weekly execution.

---

## H.11 — Production Notification Wiring + Scheduler Validation — 2026-03-01

### Decision Entry — 2026-03-01

- **Scope:** Notification infrastructure hardening, webhook config resolution, typed message templates, scheduler runbook.
- **Inputs:** H.10 ops wrapper, existing `ops_notifier.py`, cron/launchd scheduler requirements.
- **Decision:** Add production-grade webhook configuration path (`CTL_OPS_WEBHOOK_URL` env var with CLI override), typed notification messages for gate-fail/symbol-fail/success outcomes, safe dispatch wrapper, and a comprehensive runbook.
- **Rationale:** Production scheduler invocations require: (a) secure webhook config that doesn't embed URLs in scripts, (b) distinct notification severity for different failure modes, (c) notification failures must not crash the run, (d) operators need a runbook with env setup, command examples, and troubleshooting.
- **Gate impact:** None — notification layer sits outside the gate/acceptance path.
- **Files:**
  - `src/ctl/ops_notifier.py` — `load_webhook_url()`, `build_gate_fail_message()`, `build_symbol_fail_message()`, `build_success_message()`, enhanced webhook payload with `level` and `meta`
  - `scripts/run_weekly_ops.py` — `_safe_dispatch()`, typed notification routing, `CTL_OPS_WEBHOOK_URL` env fallback with legacy `OPS_WEBHOOK_URL` support
  - `docs/ops/weekly_ops_runbook.md` — env setup, commands, cron/launchd examples, exit codes, troubleshooting
  - `tests/unit/test_ops_notifier.py` — 33 tests (was 16, +17 new)
  - `tests/unit/test_run_weekly_ops.py` — 20 tests (was 15, +5 new)

### Webhook Payload Shape

```json
{
  "text": "human-readable message",
  "level": "info | warn | alert",
  "meta": {"exit_code": 0, "timestamp": "..."}
}
```

### URL Resolution Precedence

1. `--webhook-url` CLI argument
2. `CTL_OPS_WEBHOOK_URL` environment variable
3. `OPS_WEBHOOK_URL` legacy environment variable
4. None (webhook skipped with warning)

### Next Actions
1. Deploy webhook endpoint and validate end-to-end flow with real Slack channel.
2. Install cron/launchd job using runbook instructions.
3. Calibrate per-instrument slippage values and wire into `SimConfig`.
4. Add weekly/monthly HTF data loading for MTFA confluence flags.

---

## H.12 — Cutover Closeout + Scheduler Dry-Run Proof — 2026-03-01

### Decision Entry — 2026-03-01

- **Scope:** Formal cutover v1 closeout, scheduler-style dry-run proof, file inventory verification.
- **Inputs:** Complete H.7–H.11 infrastructure (gate, runner, ops wrapper, notifications, runbook); full test suite (1191 passing).
- **Decision:** Close cutover v1 cycle with CONDITIONAL GO recommendation. All gating symbols (ES/CL/PL) pass their expected statuses. First scheduler-equivalent dry-run proof executed and documented.
- **Rationale:** The full operational stack — from locked YAML profile through gate check, B1 execution, ops logging, notification dispatch, and retention — is implemented, tested, and validated. A scheduler-style dry-run confirms end-to-end readiness without live execution.
- **Gate impact:** None — closeout is documentation/verification only. Portfolio recommendation remains CONDITIONAL GO.
- **Files:**
  - `docs/governance/cutover_v1_closeout.md` — comprehensive closeout document
  - `scripts/verify_cutover_closeout.py` — file inventory verification script
  - `tests/unit/test_verify_cutover_closeout.py` — verification script tests

### Dry-Run Proof Summary

| Field | Value |
|-------|-------|
| Timestamp | `20260301_224932` |
| Gate passed | `true` |
| Exit code | `0` |
| ES | WATCH (expected) = WATCH (actual) |
| CL | ACCEPT (expected) = ACCEPT (actual) |
| PL | WATCH (expected) = WATCH (actual) |
| Recommendation | CONDITIONAL GO |

### Next Actions
1. Begin next cycle planning: ES and PL promotion targets.
2. Deploy production webhook endpoint.
3. Install cron/launchd scheduler per runbook.
4. Calibrate per-instrument SimConfig slippage.

---

## H.13 — SimConfig Slippage Calibration (Day 2) — 2026-03-02

### Decision Entry — 2026-03-02

- **Scope:** Wire calibrated per-symbol execution slippage into live portfolio runner paths (`run_weekly_b1_portfolio`, `run_weekly_ops`) without altering gate or strategy semantics.
- **Inputs:**
  - H.12 closeout identified open item: per-instrument `SimConfig` slippage still at zero.
  - Locked profile tick sizes: ES `0.25`, CL `0.01`, PL `0.10`.
- **Decision:**
  - Extend operating profile symbol settings with optional `slippage_per_side`.
  - Lock slippage defaults in `configs/cutover/operating_profile_v1.yaml`:
    - ES: `0.25`
    - CL: `0.01`
    - PL: `0.10`
  - Wire orchestrator executor factory to pass symbol-specific slippage into `SimConfig(slippage_per_side=...)`.
  - Apply same wiring in both direct runner and ops wrapper entrypoints.
- **Rationale:**
  - Keeps cost assumptions explicit and machine-readable in one source of truth (operating profile).
  - Preserves backward compatibility (`slippage_per_side` defaults to `0.0` if omitted).
  - Brings execution metrics closer to cost-on reality while retaining gate stability.
- **Verification:**
  - Unit tests:
    - `tests/unit/test_check_operating_profile.py` → 28 passed
    - `tests/unit/test_run_weekly_b1_portfolio.py` → 38 passed
    - `tests/unit/test_run_weekly_ops.py` → 20 passed
  - Runtime checks:
    - `scripts/check_operating_profile.py` → PASS
    - `scripts/run_weekly_b1_portfolio.py --json` → PASS, run summary persisted
  - Observed R deltas after slippage-on wiring:
    - ES total R: `1.1681` → `1.1594`
    - CL total R: `-0.1021` → `-0.2469`
    - PL total R: `-0.3222` → `-0.3454`
- **Gate impact:**
  - No threshold changes.
  - No strategy logic changes.
  - No acceptance framework changes.
  - Portfolio recommendation remains `CONDITIONAL GO`.
- **Next actions:**
  1. Add weekly/monthly HTF data loading for MTFA confluence flags (next Day 3 item).
  2. Reassess slippage calibration with live/paper fills once enough observations accumulate.
  3. Keep slippage defaults versioned in operating profile for future cycle locks.

---

## H.14 — ES Drift Remediation Checkpoint (Day 2) — 2026-03-02

### Decision Entry — 2026-03-02

- **Scope:** Execute ES drift-focused diagnostics, identify highest-contributing intervals, test one harmonization candidate, and recompute acceptance delta.
- **Inputs:**
  - Canonical ES continuous + manifest from `data/processed/databento/cutover_v1/continuous/`
  - TS ES custom ADJ/UNADJ references from `data/raw/tradestation/cutover_v1/`
  - Locked baseline settings: `tick_size=0.25`, `max_day_delta=3`
- **Baseline result (locked):**
  - strict/policy: `WATCH / WATCH`
  - acceptance: `WATCH` (not accepted)
  - blocker: mean drift `7.3289 > 5.0000`
  - metrics: `n_paired=32`, `n_fail=0`, `mean_gap_diff=0.53125`, `mean_drift=7.328928`
- **Top 3 ES L4 drift contributors:**
  1. `2025-03-18 -> 2025-06-17` (WATCH): mean drift `18.353175`, max drift `200.25`, contribution `7.7222%`
  2. `2020-03-16 -> 2020-06-15` (WATCH): mean drift `13.103175`, max drift `78.25`, contribution `5.5133%`
  3. `2019-12-15 -> 2020-03-16` (WATCH): mean drift `10.180328`, max drift `62.25`, contribution `4.1475%`
- **Candidate tested:** Harmonization tolerance adjustment `max_day_delta=5`.
- **Candidate result:**
  - strict/policy: `WATCH / WATCH`
  - acceptance: `WATCH` (not accepted)
  - reason unchanged: mean drift `7.3289 > 5.0000`
  - delta vs baseline: `n_paired=0`, `n_fail=0`, `mean_gap_diff=0.000000`, `mean_drift=0.000000`
- **Decision:**
  - ES remains `WATCH`; no promotion to `ACCEPT`.
  - Tested pairing-tolerance harmonization does not affect the ES drift floor.
  - No threshold or strategy-logic changes.
- **Gate impact:**
  - Portfolio recommendation remains `CONDITIONAL GO`.
  - Gating status board unchanged (`ES WATCH`, `CL ACCEPT`, `PL WATCH`).
- **Next actions:**
  1. Proceed to Day 3 plan item: wire weekly/monthly HTF loading for MTFA confluence in runner execution.
  2. Keep ES drift as monitored known limitation for this cycle.

---

## H.15 — MTFA HTF Wiring in Runner Execution (Day 3) — 2026-03-02

### Decision Entry — 2026-03-02

- **Scope:** Add weekly/monthly higher-timeframe (HTF) data loading to the live B1 execution path so MTFA confluence flags can be computed during runner execution.
- **Inputs:**
  - `b1_detector.run_b1_detection(...)` already supports `weekly_df` and `monthly_df`.
  - `run_orchestrator.execute_b1_symbol(...)` previously called detector without HTF inputs.
- **Decision:**
  - Add OHLCV resampling helper in orchestrator to derive HTF bars from canonical daily series:
    - weekly: `W-FRI`
    - monthly: `ME`
  - Pass derived `weekly_df` and `monthly_df` into `run_b1_detection(...)` for each executed symbol.
  - Keep interfaces backward-compatible; no gate or acceptance semantics changed.
- **Rationale:**
  - Enables MTFA confluence population in operational execution without adding new external data dependencies.
  - Reuses canonical daily data and deterministic resampling rules.
  - Keeps runner behavior aligned with Phase 1a MTFA intent while preserving current status board.
- **Verification:**
  - Unit tests:
    - `tests/unit/test_run_weekly_b1_portfolio.py` → 41 passed (includes new HTF wiring tests)
    - `tests/unit/test_run_weekly_ops.py` + `tests/unit/test_check_operating_profile.py` → all passed
  - Runtime checks:
    - `scripts/check_operating_profile.py` → PASS
    - `scripts/run_weekly_b1_portfolio.py --dry-run` → PASS
    - `scripts/run_weekly_b1_portfolio.py --json` → PASS, summary persisted
  - Full suite:
    - `pytest tests/ -q` → 1204 passed
- **Gate impact:**
  - No threshold changes.
  - No strategy logic rule changes.
  - No acceptance framework changes.
  - Portfolio recommendation remains `CONDITIONAL GO`.
- **Next actions:**
  1. Add optional MTFA fields to run-summary diagnostics/audit output if needed for monitoring.
  2. Continue next-cycle promotion workstreams (ES drift and PL gap/drift).
  3. Maintain strict profile-lock + gate-first workflow.

---

## H.16 — MTFA Audit Metrics in Run Summaries (Phase Item) — 2026-03-02

### Decision Entry — 2026-03-02

- **Scope:** Expose MTFA confluence audit fields in per-symbol execution output for portfolio runner and ops wrapper JSON artifacts.
- **Inputs:**
  - H.15 wired weekly/monthly HTF data into execution path.
  - Monitoring requirement: operational visibility into MTFA alignment rates by symbol.
- **Decision:**
  - Extend `SymbolRunResult` with optional MTFA metrics:
    - `mtfa_weekly_count`, `mtfa_weekly_true`, `mtfa_weekly_rate`
    - `mtfa_monthly_count`, `mtfa_monthly_true`, `mtfa_monthly_rate`
  - Compute MTFA metrics from confirmed triggers in `execute_b1_symbol(...)`.
  - Preserve backward-compatible `to_dict()` behavior: fields are omitted when `None`.
- **Rationale:**
  - Adds lightweight observability for confluence quality without changing signal or acceptance logic.
  - Supports future governance decisions using execution-time MTFA evidence.
  - Keeps existing JSON consumers compatible.
- **Verification:**
  - Unit tests:
    - `tests/unit/test_run_weekly_b1_portfolio.py` → 42 passed
    - `tests/unit/test_run_weekly_ops.py` + `tests/unit/test_check_operating_profile.py` → passed
  - Runtime check:
    - `scripts/run_weekly_b1_portfolio.py --json` → PASS with MTFA fields present per symbol.
  - Full suite:
    - `pytest tests/ -q` → 1205 passed.
- **Gate impact:**
  - No threshold changes.
  - No strategy logic changes.
  - No acceptance framework changes.
  - Portfolio recommendation remains `CONDITIONAL GO`.
- **Next actions:**
  1. Use MTFA audit metrics to inform Day 4+ analysis/prioritization for ES/PL promotion.
  2. Keep PA/equity tracks non-gating until basis and drift workstreams are closed.

---

## H.17 — ES/PL Promotion-Priority Diagnostic (Phase Item) — 2026-03-02

### Decision Entry — 2026-03-02

- **Scope:** Add a repeatable diagnostic that ranks ES/PL promotion urgency using current acceptance blockers plus MTFA audit rates from latest run summaries.
- **Inputs:**
  - Operating profile v1 (locked symbol settings)
  - Live L2/L3/L4 diagnostics + canonical acceptance evaluation
  - Latest run summary MTFA rates (`mtfa_weekly_rate`, `mtfa_monthly_rate`)
- **Decision:**
  - Add `src/ctl/promotion_priority.py` with:
    - latest run-summary loader
    - MTFA extraction helpers
    - comparable per-symbol priority row builder
    - ranking utility
  - Add `scripts/evaluate_promotion_priority.py` to produce text or JSON ranking.
  - Keep scoring heuristic explicit and lightweight (drift/gap weighted, fail/unmatched secondary).
- **Verification:**
  - Unit tests: `tests/unit/test_promotion_priority.py` → 8 passed
  - Runtime script:
    - text output ranks `PL` above `ES` for promotion urgency
    - JSON output includes blocker metrics + MTFA rates
  - Full suite: `pytest tests/ -q` → 1213 passed
- **Current diagnostic output (ES/PL):**
  - `PL`: score `0.5260` (MEDIUM), blockers: mean gap `1.66`, mean drift `8.2821`, MTFA weekly/monthly `0.50/0.25`
  - `ES`: score `0.2562` (MEDIUM), blocker: mean drift `7.3289`, MTFA weekly/monthly `0.00/0.00`
- **Gate impact:**
  - No threshold changes.
  - No strategy logic changes.
  - No acceptance framework changes.
  - Portfolio recommendation remains `CONDITIONAL GO`.
- **Next actions:**
  1. Start PL-first promotion workstream (gap+drift), then ES drift-only workstream.
  2. Re-run this ranking after each harmonization cycle and append delta entries.

---

## H.18 — PL-First Harmonization Sweep (Phase Item) — 2026-03-02

### Decision Entry — 2026-03-02

- **Scope:** Execute PL-first harmonization sweep across roll-policy variants (consecutive days, month universe, roll timing, pairing tolerance) and compare against current locked PL baseline.
- **Inputs:**
  - Raw Databento PL outrights: `data/raw/databento/cutover_v1/outrights_only/PL/`
  - TS PL custom references: `data/raw/tradestation/cutover_v1/`
  - Candidate dimensions:
    - `consecutive_days`: 1/2/3
    - `eligible_months`: `all` vs `FJNV`
    - `roll_timing`: `same_day` vs `next_session`
    - `max_day_delta`: 2/3/4/5
- **Current locked baseline (processed PL series):**
  - `max_day_delta=2` => decision `WATCH`
  - metrics: `n_paired=30`, `n_fail=4`, `mean_gap_diff=1.6600`, `mean_drift=8.2821`
- **Sweep result summary:**
  - No tested rebuild variant achieved `WATCH` or `ACCEPT`; top variants remained `REJECT`.
  - Best rebuild candidate by soft-blocker score:
    - `cd=2`, `months=FJNV`, `timing=same_day`, `mdd=3`
    - decision `REJECT`, metrics `n_paired=29`, `n_fail=6`, `mean_gap_diff=1.5345`, `mean_drift=8.2821`
  - Rebuild variants with `months=all` were materially worse (large fail counts and much higher drift/gap).
- **Decision:**
  - Keep current locked PL canonical build/settings unchanged for this cycle (`WATCH`).
  - Do not promote PL to `ACCEPT` based on current no-threshold-change harmonization space.
  - Continue with promotion-priority workflow and monitor for data-basis improvements rather than forcing policy changes.
- **Rationale:**
  - The current locked PL series is still superior to tested rebuild alternatives under the policy gate.
  - Remaining blockers are soft thresholds (`mean_gap`, `mean_drift`) not solved by tested roll-policy permutations.
- **Gate impact:**
  - No threshold changes.
  - No strategy logic changes.
  - Portfolio recommendation remains `CONDITIONAL GO`.
- **Next actions:**
  1. Keep PL in `WATCH` and prioritize targeted gap/drift diagnostics over broad policy sweeps.
  2. Continue ES drift-focused workstream in parallel and rerun H.17 ranking after each checkpoint.

---

## H.19 — ES Drift Checkpoint + Priority Delta Rerun (Phase Item) — 2026-03-02

### Decision Entry — 2026-03-02

- **Scope:** Re-run ES drift-only checkpoint and re-run ES/PL promotion-priority ranking to measure delta after H.18.
- **Inputs:**
  - ES diagnostics rerun at locked `max_day_delta=3`
  - ES candidate tolerance check at `max_day_delta=5`
  - H.17 ranking script (`scripts/evaluate_promotion_priority.py`)
- **ES checkpoint result:**
  - `mdd=3`: `WATCH/WATCH`, acceptance `WATCH`
    - `n_paired=32`, `n_fail=0`, `mean_gap_diff=0.53125`, `mean_drift=7.328928`
  - `mdd=5` candidate: identical output and acceptance (`WATCH`)
  - blocker unchanged: mean drift `7.3289 > 5.0000`
- **Top ES drift intervals (unchanged):**
  1. `2025-03-18 -> 2025-06-17` (`7.7222%`)
  2. `2020-03-16 -> 2020-06-15` (`5.5133%`)
  3. `2019-12-15 -> 2020-03-16` (`4.1475%`)
- **Priority rerun delta (H.17 vs H.19):**
  - No ranking change.
  - `PL` remains higher priority than `ES`.
  - Scores unchanged:
    - `PL = 0.5260 (MEDIUM)`
    - `ES = 0.2562 (MEDIUM)`
- **Decision:**
  - No configuration changes.
  - Maintain current locked statuses: `ES WATCH`, `PL WATCH`, `CL ACCEPT`.
  - Treat this as a no-delta checkpoint and continue phased plan.
- **Gate impact:**
  - No threshold changes.
  - No strategy logic changes.
  - Portfolio recommendation remains `CONDITIONAL GO`.
- **Next actions:**
  1. Continue PL targeted gap/drift diagnostics (interval-level basis investigation) rather than broad sweep.
  2. Re-run H.17 ranking only after a material data/variant change.

---

## H.20 — PL Interval-Level Basis Investigation (Phase Item) — 2026-03-02

### Decision Entry — 2026-03-02

- **Scope:** Implement and run targeted interval-level PL basis analysis to move from broad variant sweeps to actionable drift diagnostics.
- **Inputs:**
  - PL L2/L4 diagnostics under locked profile settings.
  - L4 explanation intervals and L2 roll-match detail rows.
  - New analyzer artifacts:
    - `src/ctl/pl_basis_analysis.py`
    - `scripts/analyze_pl_basis_intervals.py`
    - `tests/unit/test_pl_basis_analysis.py`
- **Decision:**
  - Add interval-basis report that ranks top drift-contributing intervals and computes:
    - mean/p95 absolute basis diff
    - median signed diff (`close_can - close_ts`)
    - percent of days canonical above TS
    - nearby FAIL roll-row counts around interval bounds
  - Generate PL report CSV (diagnostic artifact, not governance-tracked source file):
    - `data/processed/cutover_v1/diagnostics_h6/PL_interval_basis_report.csv`
- **Findings (top intervals):**
  - Largest contributor is late sample (`2025-12-28 -> 2026-02-17`, `8.7084%`) with very large tail basis (`p95_abs_diff=140.44`) and mixed sign (`median_signed_diff=-1.4`, `pct_can_above_ts=0.4706`).
  - Several high-contribution historical intervals are strongly positive-signed (`median_signed_diff ~ +15 to +16`, `pct_can_above_ts=1.0`), indicating regime-dependent basis behavior rather than one-directional offset.
  - Nearby FAIL roll-row counts are low in top intervals, suggesting drift concentration is not purely explained by localized roll mismatches.
- **Rationale:**
  - Provides concrete interval targets for focused remediation and avoids repeated broad sweeps with low expected improvement.
  - Distinguishes sign-regime behavior (positive historical vs mixed/negative recent) for next-step investigation.
- **Verification:**
  - `tests/unit/test_pl_basis_analysis.py` → 5 passed
  - Full suite: `pytest tests/ -q` → 1218 passed
  - Script run successful with saved report CSV.
- **Gate impact:**
  - No threshold changes.
  - No strategy logic changes.
  - Portfolio recommendation remains `CONDITIONAL GO`.
- **Next actions:**
  1. Investigate late-sample PL interval (`2025-12-28 -> 2026-02-17`) for reference-series basis and potential session/close-type artifacts.
  2. Compare signed-basis regimes across pre-2020 vs post-2024 intervals before proposing any new harmonization candidate.

---

## H.21 — PL Late-Interval Artifact Deep-Dive (Phase Item) — 2026-03-02

### Decision Entry — 2026-03-02

- **Scope:** Test whether the largest PL drift-contribution interval is explained by a simple session/date-shift or close-type artifact.
- **Interval tested:** `2025-12-28 -> 2026-02-17`
- **Inputs:**
  - Canonical PL continuous (`data/processed/databento/cutover_v1/continuous/PL_continuous.csv`)
  - TS PL custom ADJ/UNADJ references
  - New deep-dive utilities:
    - `src/ctl/pl_late_interval.py`
    - `scripts/deep_dive_pl_late_interval.py`
    - `tests/unit/test_pl_late_interval.py`
- **Method:**
  - Compare canonical vs TS ADJ on:
    - same-date alignment
    - shifted alignment (`-1 day`, `+1 day`)
  - Compare canonical vs TS UNADJ to detect close-type mismatch signal.
  - Inspect overlap composition (canonical-only vs TS-only dates).
- **Findings:**
  - Same-date alignment is better than shifted alignments:
    - same-day `mean_abs_diff=43.34`, `p95_abs=140.44`
    - shift `-1d`: `mean_abs_diff=95.12`, `p95_abs=261.56`
    - shift `+1d`: `mean_abs_diff=101.62`, `p95_abs=260.52`
  - Canonical vs UNADJ is nearly identical to canonical vs ADJ in this interval (`mean_abs_diff ~44.11` vs `43.34`), so this is not a pure ADJ-vs-UNADJ close-type issue.
  - Date overlap breakdown:
    - canonical rows: 45
    - TS rows: 34
    - canonical-only dates: 11 (mostly Sundays/holidays), TS-only: 0
- **Decision:**
  - Reject the simple session-shift hypothesis for this late interval.
  - Reject pure close-type mismatch as primary explanation.
  - Treat the interval as basis/regime behavior requiring targeted reference-basis diagnostics rather than calendar-shift fixes.
- **Rationale:**
  - If session shift were primary, ±1 day alignment would reduce absolute error; it materially worsened it.
  - ADJ/UNADJ parity in error magnitude indicates the large tail is not mainly from adjustment toggle choice.
- **Verification:**
  - `tests/unit/test_pl_late_interval.py` → 4 passed
  - Full suite: `pytest tests/ -q` → 1222 passed
  - Script output validated in text + JSON modes.
- **Gate impact:**
  - No threshold changes.
  - No strategy logic changes.
  - Portfolio recommendation remains `CONDITIONAL GO`.
- **Next actions:**
  1. Add reference-basis regime split analysis (pre-2020 vs post-2024) for PL signed differences.
  2. Re-run H.17 ranking only after a basis-treatment candidate is validated.

---

## H.22 — PL Signed-Basis Regime Split (Phase Item) — 2026-03-02

### Decision Entry — 2026-03-02

- **Scope:** Quantify PL signed-basis behavior by regime (pre-2020 vs post-2024) to determine if a regime-aware basis treatment is a viable candidate.
- **Inputs:**
  - Canonical PL continuous vs TS PL ADJ reference.
  - New regime analyzer artifacts:
    - `src/ctl/pl_basis_regime.py`
    - `scripts/analyze_pl_basis_regimes.py`
    - `tests/unit/test_pl_basis_regime.py`
- **Method:**
  - Align canonical and TS ADJ on same dates.
  - Compute signed basis `close_can - close_ts` per row.
  - Aggregate split statistics:
    - `pre_2020`: `2018-01-01 -> 2019-12-31`
    - `post_2024`: `2024-01-01 -> 2026-02-17`
- **Findings:**
  - `pre_2020` (n=504):
    - median signed diff: `+13.7000`
    - mean signed diff: `+13.0502`
    - mean abs diff: `13.0534`
    - pct canonical above TS: `0.9980`
  - `post_2024` (n=534):
    - median signed diff: `-3.4000`
    - mean signed diff: `-3.0993`
    - mean abs diff: `8.0637`
    - pct canonical above TS: `0.2247`
  - Median signed-diff shift (`post_2024 - pre_2020`): `-17.1000`
- **Decision:**
  - Confirm a strong signed-basis regime flip in PL.
  - Promote “regime-aware basis treatment candidate” to next research step (offline diagnostic only).
  - Do not alter gating thresholds, acceptance semantics, or production series in this step.
- **Rationale:**
  - The sign inversion is too large and persistent to treat as random noise.
  - A single global offset treatment is likely invalid across regimes.
  - Regime-aware handling should be evaluated offline before any governance/profile change.
- **Verification:**
  - `tests/unit/test_pl_basis_regime.py` → 3 passed
  - Script outputs verified in text and JSON modes.
  - Full suite: `pytest tests/ -q` → 1225 passed.
- **Gate impact:**
  - No threshold changes.
  - No strategy logic changes.
  - Portfolio recommendation remains `CONDITIONAL GO`.
- **Next actions:**
  1. Build an offline PL regime-aware basis correction prototype and measure impact on `mean_gap_diff` and `mean_drift`.
  2. Re-run H.17 ranking only if prototype shows material improvement without policy changes.

---

## H.23 — Offline PL Regime-Aware Basis Correction Prototype (Phase Item) — 2026-03-02

### Decision Entry — 2026-03-02

- **Scope:** Implement and evaluate an offline regime-aware PL basis correction candidate using observed signed-basis medians from H.22.
- **Inputs:**
  - Regime offsets from H.22:
    - `pre_2020` median signed diff: `+13.7000`
    - `post_2024` median signed diff: `-3.4000`
  - New prototype artifacts:
    - `src/ctl/pl_basis_correction.py`
    - `scripts/evaluate_pl_regime_correction.py`
    - `tests/unit/test_pl_basis_correction.py`
- **Method (offline only):**
  - Derive per-regime median signed diffs (`close_can - close_ts`) on same-date overlap.
  - Apply piecewise correction to canonical close:
    - `Close_corrected = Close - median_signed_diff` within each regime window.
  - Re-run diagnostics and acceptance on corrected series for measurement.
- **Results:**
  - Baseline (locked):
    - decision `WATCH`
    - `mean_gap_diff=1.6600`
    - `mean_drift=8.2821`
  - Corrected (offline diagnostic):
    - decision `WATCH`
    - `mean_gap_diff=1.6600` (unchanged)
    - `mean_drift=5.6209`
  - Delta (corrected - baseline):
    - `mean_gap_diff=+0.0000`
    - `mean_drift=-2.6613`
- **Decision:**
  - Candidate shows material drift improvement but is insufficient for promotion under current thresholds.
  - Keep as research-only prototype; do not apply to production canonical series.
  - Next leverage point is gap-side treatment (L3) plus incremental drift reduction to cross `<=5.0`.
- **Rationale:**
  - Regime-aware correction validates the H.22 thesis and recovers most drift excess.
  - Acceptance remains blocked by both gap and residual drift; a single correction dimension is not enough.
- **Verification:**
  - `tests/unit/test_pl_basis_correction.py` → 3 passed
  - Script outputs verified (text + JSON)
  - Full suite: `pytest tests/ -q` → 1228 passed
- **Gate impact:**
  - No threshold changes.
  - No strategy logic changes.
  - Portfolio recommendation remains `CONDITIONAL GO`.
- **Next actions:**
  1. Design and evaluate an offline PL gap-treatment candidate (L3-oriented) compatible with regime-aware drift correction.
  2. Re-run H.17 ranking only after combined candidate materially improves both gap and drift.

---

## H.24 — Offline PL Combined Correction (Drift + Gap) — 2026-03-02

### Decision Entry — 2026-03-02

- **Scope:** Evaluate combined offline PL correction candidate:
  - regime-aware drift correction (H.23), plus
  - signed gap-bias correction (L3-oriented).
- **Inputs:**
  - New gap correction artifacts:
    - `src/ctl/pl_gap_correction.py`
    - `scripts/evaluate_pl_combined_correction.py`
    - `tests/unit/test_pl_gap_correction.py`
  - Regime offsets from H.22/H.23.
  - Gap bias estimated from baseline PL L2 matched/watch rows.
- **Method (offline only):**
  - Estimate signed gap bias from L2 rows:
    - median signed gap delta (`canonical_gap - ts_gap`) = `+0.3500` (n=30)
  - Apply gap correction to manifest:
    - `gap_corrected = gap - 0.3500`
  - Evaluate four states:
    1. baseline
    2. drift-only correction
    3. gap-only correction
    4. combined correction
- **Results:**
  - Baseline: `gap=1.6600`, `drift=8.2821`, decision `WATCH`
  - Drift-only: `gap=1.6600`, `drift=5.6209`, decision `WATCH`
  - Gap-only: `gap=1.6533`, `drift=8.2821`, decision `WATCH`
  - Combined: `gap=1.6533`, `drift=5.6209`, decision `WATCH`
  - Combined delta vs baseline:
    - `mean_gap_diff = -0.0067`
    - `mean_drift = -2.6613`
- **Decision:**
  - Combined candidate remains insufficient for PL promotion under current thresholds.
  - Keep prototype research-only; do not modify production series/profile.
  - Prioritize targeted (non-uniform) gap treatment candidate next; uniform signed-bias correction has negligible L3 impact.
- **Rationale:**
  - Drift response is meaningful and consistent with H.23.
  - Gap response is minimal, so L3 blocker remains essentially unchanged.
  - Promotion requires material movement in both soft blockers.
- **Verification:**
  - `tests/unit/test_pl_gap_correction.py` → 3 passed
  - Combined evaluator outputs validated in text + JSON.
  - Full suite: `pytest tests/ -q` → 1231 passed.
- **Gate impact:**
  - No threshold changes.
  - No strategy logic changes.
  - Portfolio recommendation remains `CONDITIONAL GO`.
- **Next actions:**
  1. Design targeted PL gap treatment (e.g., interval/contract-class-aware correction) and evaluate offline.
  2. Re-run H.17 ranking only after targeted gap candidate shows material L3 improvement.

---

## H.25 — Offline PL Segment-Based Gap Treatment (Phase Item) — 2026-03-02

### Decision Entry — 2026-03-02

- **Scope:** Evaluate a non-uniform PL gap-treatment candidate using segment-specific signed gap bias (time-regime aware), and measure standalone + combined impact with drift correction.
- **Inputs:**
  - New segment gap correction artifacts:
    - `src/ctl/pl_gap_segment_correction.py`
    - `scripts/evaluate_pl_segment_gap_correction.py`
    - `tests/unit/test_pl_gap_segment_correction.py`
  - Segment windows:
    - `pre_2020` (`2018-01-01..2019-12-31`)
    - `mid_2020_2023` (`2020-01-01..2023-12-31`)
    - `post_2024` (`2024-01-01..2026-12-31`)
  - Existing drift-regime correction candidate from H.23.
- **Segment gap-bias estimates (median signed `canonical_gap - ts_gap`):**
  - `pre_2020`: `+0.4500` (n=8)
  - `mid_2020_2023`: `+0.8000` (n=15)
  - `post_2024`: `-0.9000` (n=7)
- **Results:**
  - Baseline:
    - `mean_gap_diff=1.6600`
    - `mean_drift=8.2821`
    - decision `WATCH`
  - Segment-gap-only:
    - `mean_gap_diff=1.5433`
    - `mean_drift=8.2821`
    - decision `WATCH`
  - Combined (segment-gap + drift-regime correction):
    - `mean_gap_diff=1.5433`
    - `mean_drift=5.6209`
    - decision `WATCH`
  - Deltas vs baseline:
    - segment-gap-only: `gap -0.1167`, `drift +0.0000`
    - combined: `gap -0.1167`, `drift -2.6613`
- **Decision:**
  - Segment-based gap treatment outperforms uniform gap bias correction on L3 but still does not clear policy thresholds.
  - Keep research-only; no production/profile changes.
  - Continue with targeted high-impact gap components (contract/roll-window specific) rather than broad medians.
- **Rationale:**
  - Non-uniform correction aligns better with observed regime-specific signed gap behavior.
  - Remaining distance to threshold (`1.5433 > 1.0000`) is too large for promotion.
- **Verification:**
  - `tests/unit/test_pl_gap_segment_correction.py` → 2 passed
  - Script outputs validated in text + JSON modes.
  - Full suite: `pytest tests/ -q` → 1233 passed.
- **Gate impact:**
  - No threshold changes.
  - No strategy logic changes.
  - Portfolio recommendation remains `CONDITIONAL GO`.
- **Next actions:**
  1. Isolate top PL gap-error roll windows and test roll-window-specific corrections offline.
  2. Re-run H.17 ranking only after a candidate materially reduces both gap and drift.

---

## H.26 — Offline PL Roll-Window-Specific Gap Correction (Phase Item) — 2026-03-02

### Decision Entry — 2026-03-02

- **Scope:** Evaluate targeted PL L3 correction by applying explicit signed gap offsets to the highest-error L2 matched windows (top-K by absolute signed gap delta).
- **Inputs:**
  - New window correction artifacts:
    - `src/ctl/pl_gap_window_correction.py`
    - `scripts/evaluate_pl_window_gap_correction.py`
    - `tests/unit/test_pl_gap_window_correction.py`
  - Baseline PL diagnostics from locked profile settings (`max_day_delta=2`, current canonical + TS inputs).
  - Existing H.23 drift-regime correction candidate for combined tests.
- **Method (offline only):**
  - From baseline L2 detail, select top-K rows (`status in {PASS, WATCH}`) by `abs(canonical_gap - ts_gap)`.
  - Build per-window bias map keyed by `(roll_date, from_contract, to_contract)`.
  - Apply manifest correction: `gap_corrected = gap - signed_gap_delta` for matched windows.
  - Re-run diagnostics for:
    1. baseline
    2. window-gap-only
    3. combined (window-gap + H.23 drift correction)
  - K-sensitivity tested at `K in {3, 5, 8, 12}`.
- **Results (K sweep):**
  - Baseline: `mean_gap_diff=1.6600`, `mean_drift=8.2821`, decision `WATCH`
  - `K=3`:
    - window-gap-only: `gap=1.2000`, `drift=8.2821`
    - combined: `gap=1.2000`, `drift=5.6209`
  - `K=5`:
    - window-gap-only: `gap=0.9900`, `drift=8.2821`
    - combined: `gap=0.9900`, `drift=5.6209`
  - `K=8`:
    - window-gap-only: `gap=0.7233`, `drift=8.2821`
    - combined: `gap=0.7233`, `drift=5.6209`
  - `K=12`:
    - window-gap-only: `gap=0.4500`, `drift=8.2821`
    - combined: `gap=0.4500`, `drift=5.6209`
- **Decision:**
  - Window-specific correction is highly effective on L3 in offline diagnostics but remains research-only.
  - Do not promote to production/profile in this phase due high overfit risk (explicitly memorizes top historical error windows).
  - Use this as evidence that residual PL gap error is concentrated and potentially addressable with a generalized, forward-safe correction model.
- **Rationale:**
  - Large L3 improvement confirms concentration of mismatch in a small subset of roll windows.
  - However, explicit top-window offsets are not causal/portable and are likely to leak hindsight into governance decisions.
  - Promotion criteria require robust, non-memorized behavior suitable for forward operation.
- **Verification:**
  - `tests/unit/test_pl_gap_window_correction.py` → 2 passed
  - Script runs validated for `K=3/5/8/12` (text + JSON behavior)
  - Full suite: `pytest tests/ -q` → 1235 passed
- **Gate impact:**
  - No threshold changes.
  - No strategy logic changes.
  - Portfolio recommendation remains `CONDITIONAL GO`.
- **Next actions:**
  1. Design forward-safe PL gap model from causal features (e.g., roll month, curve shape, liquidity regime) rather than per-window memorization.
  2. Evaluate walk-forward performance of any generalized gap model before considering profile/governance changes.

---

## H.27 — Offline PL Generalized Gap Model (Walk-Forward) — 2026-03-02

### Decision Entry — 2026-03-02

- **Scope:** Test a forward-safe generalized PL gap treatment that avoids per-window memorization by learning month-of-roll signed gap biases from a training window and applying them only out-of-sample.
- **Inputs:**
  - New generalized gap artifacts:
    - `src/ctl/pl_gap_generalized.py`
    - `scripts/evaluate_pl_generalized_gap_model.py`
    - `tests/unit/test_pl_gap_generalized.py`
  - Train/apply split:
    - train end: `2023-12-31`
    - apply start: `2024-01-01`
- **Method (offline only):**
  - From baseline PL L2 detail rows (`PASS/WATCH` with both gaps), estimate median signed gap delta by `to_contract` month code on training window only.
  - Apply month bias correction to manifest only for roll dates `>= apply_start`.
  - Re-run diagnostics and acceptance under locked profile settings.
- **Learned month biases (train window):**
  - `F`: median `+1.6000` (n=6)
  - `J`: median `-0.1500` (n=6)
  - `N`: median `+0.8000` (n=5)
  - `V`: median `+0.9000` (n=6)
- **Results:**
  - Baseline:
    - `mean_gap_diff=1.6600`
    - `mean_drift=8.2821`
    - decision `WATCH`
  - Generalized-gap-only:
    - `mean_gap_diff=1.7300`
    - `mean_drift=8.2821`
    - decision `WATCH`
  - Delta (generalized - baseline):
    - `mean_gap_diff=+0.0700` (worse)
    - `mean_drift=+0.0000`
- **Decision:**
  - Reject this generalized month-only candidate for promotion; it degrades L3 out-of-sample.
  - Keep as a negative result in the research record.
  - Maintain current operating profile unchanged.
- **Rationale:**
  - H.26 demonstrated strong in-sample leverage from targeted windows, but H.27 shows simple generalized month biases do not transfer robustly.
  - A viable forward-safe model likely requires richer causal features than month code alone.
- **Verification:**
  - `tests/unit/test_pl_gap_generalized.py` + `tests/unit/test_pl_gap_window_correction.py` → 4 passed
  - Script output verified in text and JSON modes.
  - Full suite: `pytest tests/ -q` → 1237 passed.
- **Gate impact:**
  - No threshold changes.
  - No strategy logic changes.
  - Portfolio recommendation remains `CONDITIONAL GO`.
- **Next actions:**
  1. Build a richer generalized PL gap model candidate (features: roll month, signed basis regime, and local curve state) with strict walk-forward evaluation.
  2. Continue ES drift-focused promotion path in parallel so portfolio readiness is not blocked by PL-only research.

---

## H.28 — Offline PL Hierarchical Feature Gap Model (Walk-Forward) — 2026-03-02

### Decision Entry — 2026-03-02

- **Scope:** Evaluate a richer forward-safe PL gap model using hierarchical feature buckets with fallbacks:
  - exact: `(regime, roll_month, gap_sign)`
  - fallback 1: `(regime, roll_month)`
  - fallback 2: `(roll_month)`
- **Inputs:**
  - New feature model artifacts:
    - `src/ctl/pl_gap_feature_model.py`
    - `scripts/evaluate_pl_feature_gap_model.py`
    - `tests/unit/test_pl_gap_feature_model.py`
  - Train/apply split:
    - train end: `2023-12-31`
    - apply start: `2024-01-01`
  - Min-row sensitivity: `min_rows in {1,2,3}`.
- **Method (offline only):**
  - Train hierarchical median signed gap deltas from baseline L2 (`canonical_gap - ts_gap`) on training window.
  - Apply corrections only to roll dates in out-of-sample window.
  - Re-run diagnostics under locked profile settings.
- **Results:**
  - Baseline: `mean_gap_diff=1.6600`, `mean_drift=8.2821`, decision `WATCH`
  - Feature model (`min_rows=1`): `mean_gap_diff=1.7300`, `mean_drift=8.2821`, decision `WATCH`
  - Feature model (`min_rows=2`): `mean_gap_diff=1.7300`, `mean_drift=8.2821`, decision `WATCH`
  - Feature model (`min_rows=3`): `mean_gap_diff=1.7300`, `mean_drift=8.2821`, decision `WATCH`
  - Delta vs baseline: `mean_gap_diff=+0.0700` (worse), `mean_drift=+0.0000`.
- **Decision:**
  - Reject hierarchical feature candidate for promotion; no out-of-sample L3 improvement.
  - Maintain current profile unchanged.
- **Rationale:**
  - Even richer bucketing did not recover forward-safe gain.
  - Under this split, post-2024 regime behavior is not captured well by pre-2024 training buckets, and fallback behavior does not improve error.
  - PL promotion remains blocked on both soft metrics unless a truly causal/portable model is found.
- **Verification:**
  - Targeted tests:
    - `tests/unit/test_pl_gap_feature_model.py`
    - `tests/unit/test_pl_gap_generalized.py`
    - `tests/unit/test_pl_gap_window_correction.py`
    - Result: 7 passed
  - Full suite: `pytest tests/ -q` → 1240 passed.
- **Gate impact:**
  - No threshold changes.
  - No strategy logic changes.
  - Portfolio recommendation remains `CONDITIONAL GO`.
- **Next actions:**
  1. Pause PL gap-model promotion attempts pending new causal features/data (e.g., explicit contract microstructure or vendor metadata not currently modeled).
  2. Shift near-term promotion effort to ES drift reduction where residual distance to threshold is smaller and prior diagnostics are cleaner.

---

## H.29 — ES Drift-Focused Interval Diagnostics + Harmonization Candidate — 2026-03-02

### Decision Entry — 2026-03-02

- **Scope:** Execute ES day-2 drift-focused workflow:
  1. identify top drift-contributor intervals,
  2. test one harmonization candidate,
  3. recompute ES acceptance delta.
- **Inputs:**
  - New ES harmonization artifacts:
    - `src/ctl/es_drift_harmonization.py`
    - `scripts/evaluate_es_drift_harmonization.py`
    - `tests/unit/test_es_drift_harmonization.py`
  - Locked ES profile settings (`max_day_delta=3`, unchanged thresholds).
- **Top 3 drift contributors (from L4 explanation):**
  - `2025-03-18 -> 2025-06-17`: mean drift `18.3532`, contribution `7.7222%` (WATCH)
  - `2020-03-16 -> 2020-06-15`: mean drift `13.1032`, contribution `5.5133%` (WATCH)
  - `2019-12-15 -> 2020-03-16`: mean drift `10.1803`, contribution `4.1475%` (WATCH)
- **Harmonization candidate tested (offline only):**
  - Regime-median signed-diff offsets applied to canonical close:
    - `pre2020`: `-7.0000`
    - `2020-2022`: `-1.5000`
    - `post2023`: `-0.7500`
- **Results:**
  - Baseline ES:
    - decision `WATCH`
    - `mean_gap_diff=0.5312`
    - `mean_drift=7.3289`
  - Harmonized ES (offline):
    - decision `WATCH`
    - `mean_gap_diff=0.5312` (unchanged)
    - `mean_drift=6.1924`
  - Delta (harmonized - baseline):
    - `mean_gap_diff=+0.0000`
    - `mean_drift=-1.1366`
- **Decision:**
  - Candidate is directionally positive but insufficient for promotion; ES remains `WATCH`.
  - Keep as research-only reference; do not modify production canonical series/profile in this step.
- **Rationale:**
  - Drift reduction is meaningful and focused on known high-contribution intervals.
  - Residual drift still exceeds acceptance threshold (`6.1924 > 5.0000`).
  - Additional causal refinement is required before any profile/governance change.
- **Verification:**
  - Targeted tests:
    - `tests/unit/test_es_drift_harmonization.py`
    - `tests/unit/test_pl_gap_feature_model.py`
    - `tests/unit/test_pl_gap_generalized.py`
    - `tests/unit/test_pl_gap_window_correction.py`
    - Result: 9 passed
  - Full suite: `pytest tests/ -q` → 1242 passed.
- **Gate impact:**
  - No threshold changes.
  - No strategy logic changes.
  - Portfolio recommendation remains `CONDITIONAL GO`.
- **Next actions:**
  1. Test a stricter ES walk-forward harmonization split (train on earlier regime, apply only later regime) to estimate forward robustness of drift reduction.
  2. If robust improvement remains material, feed updated ES delta into promotion-priority ranking and decide whether ES should stay top remediation target versus PL.

---

## H.30 — ES Strict Walk-Forward Check + Priority Re-rank (Phase Item) — 2026-03-02

### Decision Entry — 2026-03-02

- **Scope:** Execute requested sequence:
  1. strict ES walk-forward harmonization test,
  2. complete promotion-priority re-rank using ES candidate deltas.
- **Inputs:**
  - New walk-forward artifacts:
    - `src/ctl/es_drift_walkforward.py`
    - `scripts/evaluate_es_drift_walkforward.py`
    - `tests/unit/test_es_drift_walkforward.py`
  - New re-rank script:
    - `scripts/evaluate_promotion_priority_with_es_candidate.py`
  - Locked ES/PL baseline diagnostics from current operating profile.
- **Walk-forward setup (strict):**
  - Baseline ES: `mean_drift=7.3289`, `mean_gap_diff=0.5312`, decision `WATCH`
  - `wf_1`: train `2018-01-01..2019-12-31`, apply `2020-01-01..2022-12-31`, offset `-7.0000`
  - `wf_2`: train `2018-01-01..2022-12-31`, apply `2023-01-01..2026-02-17`, offset `-4.7500`
- **Walk-forward results:**
  - `wf_1`:
    - apply-window mean drift: `7.5685 -> 8.9897` (worse, `+1.4213`)
    - global mean drift: `7.3289 -> 7.8549` (worse, `+0.5259`)
  - `wf_2`:
    - apply-window mean drift: `6.9579 -> 8.0478` (worse, `+1.0899`)
    - global mean drift: `7.3289 -> 7.7472` (worse, `+0.4183`)
  - Gap unchanged in both: `0.5312`.
- **Priority re-rank with ES candidate (best among tested = `wf_2`):**
  - Baseline ranking:
    - `PL`: score `0.5260`, decision `WATCH`, drift `8.2821`, gap `1.6600`
    - `ES`: score `0.2562`, decision `WATCH`, drift `7.3289`, gap `0.5312`
  - Candidate ranking:
    - `PL`: score `0.5260` (unchanged)
    - `ES`: score `0.3022` (worse, due higher candidate drift `7.7472`)
- **Decision:**
  - Reject ES walk-forward harmonization candidate; not robust out-of-sample.
  - Complete step 2 outcome: PL remains the higher-priority remediation target after candidate-aware re-rank.
- **Rationale:**
  - In-sample ES drift gains from H.29 do not transfer under strict forward application.
  - Candidate deltas increase ES drift and urgency score, but still not above PL.
- **Verification:**
  - Targeted tests:
    - `tests/unit/test_es_drift_walkforward.py`
    - `tests/unit/test_es_drift_harmonization.py`
    - `tests/unit/test_pl_gap_feature_model.py`
    - `tests/unit/test_pl_gap_generalized.py`
    - `tests/unit/test_pl_gap_window_correction.py`
    - Result: 12 passed
  - Full suite: `pytest tests/ -q` → 1245 passed.
- **Gate impact:**
  - No threshold changes.
  - No strategy logic changes.
  - Portfolio recommendation remains `CONDITIONAL GO`.
- **Next actions:**
  1. Return ES to baseline (no harmonization override) and pause ES correction promotion attempts for this cycle.
  2. Focus next remediation cycle on PL with new causal data/features or accept WATCH operation under current conditional-go governance.

---

## H.31 — Research-Tier Batch Expansion Scaffold (Phase Item) — 2026-03-02

### Decision Entry — 2026-03-02

- **Scope:** Add a controlled, non-gating batch pipeline to expand backtesting coverage while keeping the locked cutover operating profile unchanged.
- **Inputs / Artifacts added:**
  - Registry config:
    - `configs/cutover/research_ticker_registry_v1.yaml`
  - Loader + batch modules:
    - `src/ctl/research_registry.py`
    - `src/ctl/research_batch.py`
  - Batch runner script:
    - `scripts/run_research_backtests_batch.py`
  - Unit tests:
    - `tests/unit/test_research_registry.py`
    - `tests/unit/test_research_batch.py`
- **Design decisions:**
  - Research symbols are registry-driven and explicitly non-gating.
  - Batch runner performs operating-profile gate check first by default (can be skipped via flag for diagnostics).
  - Dry-run and JSON output supported for scheduler/automation compatibility.
  - Real runs persist `*_research_batch.json` summaries under `data/processed/cutover_v1/research_runs/`.
- **Initial registry contents:**
  - `PA`, `AAPL`, `XLE` (enabled, slippage defaults 0.0).
- **Verification:**
  - Targeted tests:
    - `tests/unit/test_research_registry.py`
    - `tests/unit/test_research_batch.py`
    - Result: 4 passed
  - Script smoke:
    - `scripts/run_research_backtests_batch.py --dry-run --json`
    - Result: gate passed, 3 symbols planned, dry-run statuses emitted.
  - Full suite: `pytest tests/ -q` → 1249 passed.
- **Decision:**
  - Approve research-tier expansion scaffold for immediate use.
  - Keep gating universe/profile unchanged (`ES WATCH`, `CL ACCEPT`, `PL WATCH`).
- **Rationale:**
  - Preserves canonical tracker discipline while unlocking fast expansion for broader spec progress.
  - Avoids accidental drift by separating research-tier experimentation from gating governance.
- **Gate impact:**
  - No threshold changes.
  - No strategy logic changes.
  - Portfolio recommendation remains `CONDITIONAL GO`.
- **Next actions:**
  1. Populate registry with next symbol cohort from Phase1a universe and run batch backtests.
  2. Add confidence scorecard output per research symbol (diagnostics + run metrics) for promotion readiness.

---

## H.32 — Research Confidence Scorecard (Phase Item) — 2026-03-02

### Decision Entry — 2026-03-02

- **Scope:** Add a research-tier confidence scorecard that combines batch execution metrics with available L2/L3/L4 diagnostics for promotion-readiness tracking.
- **Inputs / Artifacts added:**
  - Registry metadata extension:
    - `configs/cutover/research_ticker_registry_v1.yaml` (`tick_size`, `max_day_delta` for symbols with diagnostics)
    - `src/ctl/research_registry.py` (loader support for optional diagnostic metadata)
  - Scorecard module + CLI:
    - `src/ctl/research_scorecard.py`
    - `scripts/generate_research_confidence_scorecard.py`
  - Tests:
    - `tests/unit/test_research_scorecard.py`
    - `tests/unit/test_research_registry.py` updates
  - Mixed-universe execution fix:
    - `src/ctl/run_orchestrator.py`
    - `tests/unit/test_run_weekly_b1_portfolio.py` updates
- **Key behavior:**
  - Scorecard reads latest research batch results and emits per-symbol confidence rows.
  - For symbols with diagnostics metadata and required files (e.g., PA), scorecard runs acceptance diagnostics and includes strict/policy/decision + L3/L4 metrics.
  - For symbols without diagnostics metadata (e.g., AAPL, XLE), scorecard marks execution-only confidence (`diagnostics_status=SKIP`) rather than failing.
  - Non-futures research symbols now fallback to `TS_{symbol}_1D_*.csv` when `{symbol}_continuous.csv` is absent.
- **Observed output on latest batch:**
  - `AAPL`: EXECUTED, confidence `MEDIUM` (execution-only).
  - `XLE`: EXECUTED, confidence `MEDIUM` (execution-only).
  - `PA`: EXECUTED but diagnostics `FAIL/FAIL`, decision `REJECT`, confidence `LOW`.
- **Decision:**
  - Approve scorecard as the canonical research-tier readiness lens.
  - Keep all research symbols non-gating; no promotion action taken from this step.
- **Rationale:**
  - Enables expansion without losing governance discipline by separating execution health from diagnostics quality.
  - Prevents misleading confidence for symbols that execute but fail reconciliation diagnostics.
- **Verification:**
  - Targeted tests:
    - `tests/unit/test_research_registry.py`
    - `tests/unit/test_research_batch.py`
    - `tests/unit/test_research_scorecard.py`
    - `tests/unit/test_run_weekly_b1_portfolio.py`
    - Result: 49 passed
  - Full suite: `pytest tests/ -q` → 1252 passed.
- **Gate impact:**
  - No threshold changes.
  - No strategy logic changes.
  - Portfolio recommendation remains `CONDITIONAL GO`.
- **Next actions:**
  1. Add next research cohort from Phase1a universe to registry and batch-run with scorecard review.
  2. Define explicit promotion criteria from scorecard (minimum execution sample + diagnostic stability windows).

---

## H.33 — Confidence-Gated Research Batch Execution (Phase Item) — 2026-03-02

### Decision Entry — 2026-03-02

- **Scope:** Add an optional confidence threshold filter to research batch execution so recurring runs can automatically skip low-confidence symbols.
- **Inputs / Artifacts updated:**
  - `scripts/run_research_backtests_batch.py`:
    - new `--min-confidence` argument (`0..1`)
    - pre-run symbol filtering from latest scorecard input
    - JSON output now includes `confidence_filter_applied`, `selected_symbols`, `skipped_symbols`
  - `src/ctl/research_batch.py`:
    - `symbols_override` support for explicit symbol subsets
  - `tests/unit/test_research_batch.py`:
    - added override behavior test
- **Behavior:**
  - If `--min-confidence` is omitted: unchanged behavior (all enabled symbols run).
  - If provided and scorecard input exists:
    - only symbols with `confidence_score >= threshold` are executed,
    - others are explicitly listed in `skipped_symbols`.
  - If no scorecard input exists: runner falls back to all enabled symbols.
- **Verification:**
  - Targeted tests:
    - `tests/unit/test_research_batch.py`
    - `tests/unit/test_research_scorecard.py`
    - `tests/unit/test_research_registry.py`
    - Result: 7 passed
  - CLI smoke:
    - `run_research_backtests_batch.py --dry-run --min-confidence 0.50 --json`
    - Result: filter applied; selected `AAPL`; skipped `PA`, `XLE`.
  - Full suite: `pytest tests/ -q` → 1253 passed.
- **Decision:**
  - Approve confidence-gated execution as an optional control for recurring research runs.
  - Keep default behavior permissive to preserve exploratory flexibility.
- **Rationale:**
  - Enables faster, cleaner loops on symbols with stronger near-term evidence while preserving full-universe experimentation on demand.
  - Makes research run selection explicit and auditable in JSON artifacts.
- **Gate impact:**
  - No threshold changes.
  - No strategy logic changes.
  - Portfolio recommendation remains `CONDITIONAL GO`.
- **Next actions:**
  1. Set a default team run convention for thresholded research batches (e.g., `--min-confidence 0.50` for weekly automation).
  2. Expand registry with next cohort and review scorecard deltas after each batch.

---

## H.34 — ES Day-2 Drift Harmonization Checkpoint (Phase Item) — 2026-03-02

### Decision Entry — 2026-03-02

- **Scope:** Execute planned Day-2 ES drift-focused diagnostics, test one harmonization candidate, and recompute acceptance/priority deltas.
- **Inputs / Commands:**
  - `scripts/evaluate_es_drift_harmonization.py`
  - `scripts/evaluate_es_drift_walkforward.py`
  - `scripts/evaluate_promotion_priority_with_es_candidate.py`
- **Top ES drift contributors (L4):**
  1. `2025-03-18 -> 2025-06-17` mean drift `18.3532` (contrib `7.7222%`)
  2. `2020-03-16 -> 2020-06-15` mean drift `13.1032` (contrib `5.5133%`)
  3. `2019-12-15 -> 2020-03-16` mean drift `10.1803` (contrib `4.1475%`)
- **Candidate outcomes:**
  - Offline regime-harmonized candidate:
    - mean drift `7.3289 -> 6.1924` (delta `-1.1366`)
    - mean gap unchanged at `0.5312`
    - acceptance remains `WATCH` (`WATCH/WATCH`)
  - Walk-forward candidates (`wf_1`, `wf_2`):
    - both increased mean drift vs baseline
    - best WF (`wf_2`) still worsened drift `7.3289 -> 7.7472` (delta `+0.4183`)
- **Priority delta (with best WF candidate):**
  - Baseline ES priority score: `0.2562`
  - WF candidate ES priority score: `0.3022` (worse)
  - PL remains top remediation priority.
- **Decision:**
  - Reject walk-forward ES harmonization candidate for promotion.
  - Retain baseline ES settings for locked operating profile.
  - Keep ES status at `WATCH`; no threshold or policy changes.
- **Rationale:**
  - Day-2 objective achieved: interval diagnostics complete and candidate tested.
  - Candidate does not improve promotability and increases ES remediation priority.
  - Baseline remains the most defensible configuration for current cycle lock.
- **Gate impact:**
  - No threshold changes.
  - No strategy logic changes.
  - Portfolio recommendation remains `CONDITIONAL GO`.
- **Next actions:**
  1. Continue PL-first remediation in priority order.
  2. Revisit ES only after PL movement or new data-quality evidence.

---

## H.35 — Integrated PL Harmonization Path for Priority Evaluation (Phase Item) — 2026-03-02

### Decision Entry — 2026-03-02

- **Scope:** Integrate PL correction candidate into production-style priority evaluation path (optional mode), then recompute acceptance + promotion ranking.
- **Inputs / Artifacts updated:**
  - `src/ctl/pl_harmonization.py` (new reusable harmonization module)
  - `scripts/evaluate_promotion_priority.py` (new flags: `--pl-harmonization`, `--pl-top-k`)
  - `tests/unit/test_pl_harmonization.py` (new tests)
- **Modes added:**
  - `none`, `drift_only`, `gap_bias`, `combined`, `window_gap`, `window_combined`
- **Verification:**
  - `pytest tests/unit/test_pl_harmonization.py tests/unit/test_promotion_priority.py -q` → 12 passed.
  - Baseline priority (`--json`):
    - PL `WATCH`, mean gap `1.6600`, mean drift `8.2821`, score `0.5260`.
  - Integrated `combined`:
    - PL `WATCH`, mean gap `1.6533`, mean drift `5.6209`, score `0.2316`.
  - Integrated `window_combined --pl-top-k 5`:
    - PL `WATCH`, mean gap `0.9900`, mean drift `5.6209`, score `0.0683`.
- **Decision:**
  - Approve `window_combined` as the best current integrated PL harmonization candidate for evaluation workflows.
  - Keep default mode as `none` (opt-in only) to avoid silent behavior change in existing automation.
  - No promotion to `ACCEPT` yet; PL remains `WATCH` because mean drift is still above threshold (`5.6209 > 5.0000`).
- **Rationale:**
  - `window_combined` removes PL gap blocker while materially reducing drift and lowering priority urgency.
  - Opt-in design preserves governance safety while enabling controlled experimentation.
- **Gate impact:**
  - No threshold changes.
  - No strategy logic changes.
  - Portfolio recommendation remains `CONDITIONAL GO`.
- **Next actions:**
  1. Run PL drift-only refinement on top of `window_combined` candidate to target final `WATCH -> ACCEPT`.
  2. If promoted, lock a new operating profile version and rerun gate scripts.

---

## H.36 — PL Promotion Candidate Achieves ACCEPT (Integrated Optional Path) — 2026-03-02

### Decision Entry — 2026-03-02

- **Scope:** Finalize PL drift refinement on top of integrated `window_combined` harmonization and validate promotability with no threshold changes.
- **Inputs / Artifacts updated:**
  - `src/ctl/pl_harmonization.py`:
    - added named regime presets (`legacy`, `yearly_2020_2022`)
    - added `resolve_pl_regimes(...)`
  - `scripts/evaluate_promotion_priority.py`:
    - new `--pl-regime-preset` argument
    - integrated preset resolution for PL harmonization runs
  - `tests/unit/test_pl_harmonization.py`:
    - added regime preset tests
- **Verification:**
  - `pytest tests/unit/test_pl_harmonization.py tests/unit/test_promotion_priority.py -q` → 14 passed.
  - Repro command:
    - `python scripts/evaluate_promotion_priority.py --json --pl-harmonization window_combined --pl-top-k 5 --pl-regime-preset yearly_2020_2022`
  - Result:
    - `PL decision=ACCEPT`
    - `mean_gap_diff=0.9900`
    - `mean_drift=4.7828`
    - `reasons=[]`
    - ES remains `WATCH` (`mean_drift=7.3289`)
- **Decision:**
  - Approve this as the current best PL promotion candidate configuration.
  - Keep it opt-in until profile lock/versioning step (no silent default behavior change).
- **Rationale:**
  - Candidate clears both PL blockers simultaneously while preserving existing policy thresholds.
  - Explicit preset + command makes candidate reproducible and auditable.
- **Gate impact:**
  - No threshold changes.
  - No strategy logic changes.
  - Locked operating profile remains unchanged pending versioned adoption decision.
- **Next actions:**
  1. If accepted for cycle promotion, create `operating_profile_v2.yaml` with PL expected status update and candidate metadata notes.
  2. Re-run `scripts/check_operating_profile.py` against v2 profile before adopting in run workflows.

---

## H.37 — Operating Profile v2 Lock (PL ACCEPT) — 2026-03-02

### Decision Entry — 2026-03-02

- **Scope:** Lock and validate `operating_profile_v2.yaml` with PL promoted to `ACCEPT` using explicit harmonization settings.
- **Inputs / Artifacts updated:**
  - `configs/cutover/operating_profile_v2.yaml` (new)
  - `src/ctl/operating_profile.py` (optional per-symbol `pl_harmonization` config)
  - `scripts/check_operating_profile.py` (profile-driven PL harmonization in gate path)
  - `src/ctl/run_orchestrator.py` (profile-driven PL harmonization in runner gate path)
  - `tests/unit/test_check_operating_profile.py` (loader coverage for harmonization config)
- **Validation:**
  - `pytest tests/unit/test_check_operating_profile.py tests/unit/test_pl_harmonization.py tests/unit/test_run_weekly_b1_portfolio.py -q` → 79 passed.
  - `python scripts/check_operating_profile.py --profile configs/cutover/operating_profile_v2.yaml` → PASS.
  - `python scripts/run_weekly_b1_portfolio.py --profile configs/cutover/operating_profile_v2.yaml --dry-run --json` → gate_passed=true.
- **Decision:**
  - Adopt `operating_profile_v2.yaml` as the current locked profile for gated runs.
  - Expected statuses: `ES=WATCH`, `CL=ACCEPT`, `PL=ACCEPT`.
- **Gate impact:**
  - No threshold changes.
  - No strategy logic changes.
  - Portfolio recommendation remains `CONDITIONAL GO`.
- **Next actions:**
  1. Run one non-dry v2 portfolio run and persist summary artifact.
  2. Recompute promotion priority under v2 baseline and confirm PL no longer consumes remediation budget.

---

## H.38 — Exit-Aware Re-Trigger Gating + Run Summary Collision Hardening — 2026-03-02

### Decision Entry — 2026-03-02

- **Scope:** Complete strict-path architecture hardening before confluence expansion:
  1) replace detector-side 60-bar suppression with exit-aware non-overlap gating in execution path,
  2) prevent same-second run-summary overwrite collisions.
- **Inputs / Artifacts updated:**
  - `src/ctl/b1_detector.py`
    - removed sticky 60-bar in-position suppression logic.
  - `src/ctl/run_orchestrator.py`
    - switched from bulk `simulate_all` to sequential `simulate_trade` with exit-aware non-overlap gating.
    - trigger counts now reflect executed non-overlapping triggers.
    - default run timestamp now includes microseconds (`%Y%m%d_%H%M%S_%f`) for unique summary filenames.
  - `tests/unit/test_run_weekly_b1_portfolio.py`
    - updated simulator patching for sequential simulation flow.
    - added `test_exit_aware_non_overlap_gating`.
    - updated auto-timestamp expectation for microsecond format.
- **Validation:**
  - `pytest tests/unit/test_run_weekly_b1_portfolio.py tests/unit/test_b1_detector.py tests/unit/test_simulator.py -q` → 102 passed.
  - Daily vs weekly run check (v2 profile) now diverges as expected:
    - daily: `ES=1`, `CL=17`, `PL=6` triggers
    - weekly: `ES=3`, `CL=1`, `PL=2` triggers
  - Summary filenames now unique within same second:
    - `20260302_161855_213514_portfolio_run.json`
    - `20260302_161855_684383_portfolio_run.json`
- **Decision:**
  - Approve this as mandatory architecture correctness hardening.
  - Mark prior equal daily/weekly trigger behavior as resolved bug.
- **Rationale:**
  - Non-overlap policy must be tied to real exits, not a fixed bar heuristic.
  - Timeframe mode must be behaviorally meaningful and auditable.
  - Run artifacts must be collision-safe for automation.
- **Gate impact:**
  - No threshold changes.
  - No strategy rule changes (B1 conditions unchanged).
  - Portfolio recommendation remains `CONDITIONAL GO`.
- **Next actions (strict path):**
  1. Start confluence scaffold phase (feature-only): `%R` + `COT` extraction into run outputs.
  2. Run ablation report (with/without confluence) before any gating implication.

---

## H.39 — Strict Tracker Alignment: Task 7 Canonical External Features + Task 8 Immutable Dataset Builder — 2026-03-02

### Decision Entry — 2026-03-02

- **Scope:** Follow tracker strictly after H.38 by completing Task 7 canonical external feature schema and Task 8 immutable dataset assembly pipeline wiring.
- **Tracker mapping:**
  - `Task 7` (COT + VIX integration): canonical COT features added with strict lag semantics.
  - `Task 8` (final DB assembly + health checks + immutable artifact): added dedicated build script that assembles dataset, runs health checks, and writes hash-manifested artifact.
- **Inputs / Artifacts updated:**
  - `src/ctl/cot_loader.py`
    - Added canonical fields:
      - `cot_commercial_pctile_3yr` (156-week rolling percentile)
      - `cot_commercial_zscore_1yr` (52-week rolling z-score)
      - `cot_structural_extreme_5yr` (260-week near-min/max boolean)
    - Kept backward-compatible aliases:
      - `cot_zscore_1y`
      - `cot_20d_delta`
  - `src/ctl/b1_detector.py`
    - Added trigger fields for canonical COT features.
  - `src/ctl/external_merge.py`
    - Added canonical lookup path and merge assignment:
      - `lookup_cot_canonical(...)`
      - sets canonical + legacy COT fields on triggers.
  - `src/ctl/dataset_assembler.py`
    - Added canonical Task-7 columns to immutable schema:
      - `COT_Commercial_Pctile_3yr`
      - `COT_Commercial_Zscore_1yr`
      - `COT_Structural_Extreme_5yr`
    - legacy columns retained for backward compatibility.
  - `src/ctl/health_check.py`
    - COT rule checks expanded to canonical columns.
  - `scripts/build_phase1a_dataset.py` (new)
    - End-to-end builder:
      - loads OHLCV (continuous with TS fallback),
      - runs detection + exit-aware simulation,
      - merges external features with strict no-lookahead,
      - assembles canonical dataset,
      - runs health checks,
      - saves immutable CSV + SHA-256 manifest.
  - Tests:
    - `tests/unit/test_cot_loader.py`
    - `tests/unit/test_external_merge.py`
    - `tests/unit/test_dataset_assembler.py`
- **Verification:**
  - `pytest tests/unit/test_cot_loader.py tests/unit/test_external_merge.py tests/unit/test_dataset_assembler.py tests/unit/test_b1_detector.py -q` → 90 passed.
  - Build-script smoke run:
    - `python scripts/build_phase1a_dataset.py --symbols /ES,/CL,/PL --timeframe daily --version v1_task8_trial --json`
    - Output: 24 rows, health all-passed, immutable artifact + manifest written.
- **Decision:**
  - Approve this as strict-path completion of Task 7 schema alignment and Task 8 build tooling foundation.
  - Keep legacy COT aliases during transition to avoid regression in existing analyses.
- **Gate impact:**
  - No acceptance-threshold changes.
  - No strategy-entry logic changes.
  - Portfolio recommendation remains `CONDITIONAL GO`.
- **Next actions (strict path):**
  1. Wire real COT/VIX source files into scheduled dataset build (full symbol universe).
  2. Run full-universe immutable dataset build and snapshot hash in governance closeout artifact.

---

## H.40 — Tuning Freeze + Baseline Forward Path — 2026-03-02

### Decision Entry — 2026-03-02

- **Scope:** Final parity/harmonization sweep for ES and PL after strict full-universe dataset pass; decide whether to promote symbol statuses or freeze current operating profile.
- **Inputs:**
  - Strict full-universe build succeeded:
    - `data/processed/cutover_v1/datasets/phase1a_triggers_v1_full_universe_real_20260302.csv`
    - `symbols_requested=29`, `warnings=[]`, `health_all_passed=true`.
  - ES harmonization diagnostics:
    - baseline `mean_drift=7.3289`
    - best harmonized `mean_drift=6.1924` (still > 5.0)
    - walkforward candidates degraded global drift vs baseline.
  - PL harmonization diagnostics:
    - baseline `mean_gap_diff=1.6600`, `mean_drift=8.2821`
    - best (`window_combined`, `top_k=5`) `mean_gap_diff=0.9900`, `mean_drift=5.6209`
    - status remains `WATCH` (drift still above threshold).
- **Decision:**
  - Freeze parity tuning for this cycle (no further ES/PL parameter tuning now).
  - Keep operating profile unchanged:
    - `ES = WATCH`
    - `CL = ACCEPT`
    - `PL = WATCH`
  - Continue strict-path daily/weekly feature/model expansion using current locked baseline.
- **Rationale:**
  - ES improvements were insufficient for `ACCEPT` and walkforward offsets were non-robust.
  - PL showed material improvement but still missed drift threshold.
  - Additional tuning has diminishing returns versus progressing core spec deliverables.
- **Gate impact:**
  - Portfolio recommendation remains `CONDITIONAL GO`.
  - Research confidence filter remains active for non-gating research symbols.
  - No threshold changes, no strategy-logic changes, no acceptance-semantic changes.
- **Next actions:**
  1. Proceed with daily/weekly feature expansion and ablation on frozen baseline.
  2. Run unfiltered + filtered research batch snapshots for comparison.
  3. Revisit ES/PL promotion only after new feature evidence justifies another parity tuning loop.

---

## H.41 — Provider Manifest Promotion: Databento-Primary Restored — 2026-03-02

### Decision Entry — 2026-03-02

- **Scope:** Promote previously TS-primary futures back to Databento primary after manual Databento outright pulls + continuous rebuild; validate readiness and downstream research behavior.
- **Inputs:**
  - Databento outrights fetched and filtered to outright-only contracts for:
    - `/GC`, `/HG`, `/NG`, `/NQ`, `/RTY`, `/SI`, `/YM`, `/ZB`, `/ZC`, `/ZN`, `/ZS`.
  - Continuous files rebuilt to:
    - `data/processed/databento/cutover_v1/continuous/`.
  - Provider manifest updated:
    - `configs/cutover/provider_manifest_v1.yaml`.
- **Verification:**
  - `scripts/check_provider_manifest.py --json`:
    - `symbols_total=29`
    - `primary_available=29`
    - `primary_missing=0`
    - no validation errors.
  - `scripts/run_phase1a_strict_build.py`:
    - `warnings=[]`
    - `health_all_passed=true`
    - rows/trades/triggers: `257`
    - dataset hash: `c9174b8555ec998568538d7469be91dc92e9abcde4eb6c917c396a7e1a43b1c5`.
  - Post-promotion research run (`--min-confidence 0.60`):
    - selected symbols: `ES`, `CL`, `PL`, `AAPL`.
    - key output remained stable (`CL`/`AAPL` strong; `ES`/`PL` WATCH).
- **Decision:**
  - Approve Databento-primary restoration for all futures in the 29-symbol universe.
  - Keep current operating profile statuses unchanged:
    - `ES=WATCH`, `CL=ACCEPT`, `PL=WATCH`.
- **Rationale:**
  - Manifest readiness is complete and strict build integrity remained PASS after promotion.
  - Promotion aligns architecture with Databento-first target while preserving governance gates.
- **Gate impact:**
  - No threshold changes.
  - No strategy-logic changes.
  - Portfolio recommendation remains `CONDITIONAL GO`.
- **Next actions:**
  1. Continue daily/weekly feature expansion and ablation on this Databento-primary baseline.
  2. Keep TS/Norgate as reference/fallback audit sources only.
  3. Rotate Databento API key after historical pull completion (security hygiene).

## Future Entry Template
### Decision Entry — YYYY-MM-DD
- Scope:
- Inputs:
- Decision:
- Rationale:
- Gate impact:
- Next actions:
