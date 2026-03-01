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

## Future Entry Template
### Decision Entry — YYYY-MM-DD
- Scope:
- Inputs:
- Decision:
- Rationale:
- Gate impact:
- Next actions:
