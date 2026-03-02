# Alternative Strategy Catalogue — Codex Edit
### Prioritized Build Plan and Integration Notes
*v0.2-codex — March 2026*

## Purpose
Convert the strategy catalogue into an execution-prioritized roadmap that fits current CTL cutover state.

## Current Ground Truth
- Current cycle operating posture: `CONDITIONAL GO`.
- Gate universe: `ES`, `CL`, `PL`.
- Non-gating tracks: `PA`, `AAPL`, `XLE`.
- Existing runner/ops stack is live (`check_operating_profile`, `run_weekly_b1_portfolio`, `run_weekly_ops`).

## What To Build Next (Highest ROI)
1. **R Engine v1 (portfolio risk governor)**
- Define `R` as % of equity from a single config source.
- Add hard daily/weekly/monthly loss caps in `R`.
- Add drawdown governor (reduce risk after drawdown threshold).
- Store `R_used`, `R_pnl`, and cap state in run summaries.

2. **Commodity Spread Sandbox (#13)**
- Build spread calculator and z-score monitor for 2-3 pilot spreads.
- Start with PA/PL and CL structure-adjacent spreads.
- Keep output as overlay/filter first, not independent execution engine.

3. **Rates Sandbox (#10)**
- Add regime features (real yields, Fed path proxy) as macro overlay.
- Use as conviction sizing modifier before standalone rates strategy execution.

## What To Defer
- Full crypto funding arb automation (#9)
- VIX surface/vol complex (#12)
- Cross-exchange basis infra (#15)
- Power/emissions (#16)

These are valid but should not slow core CTL hardening.

## Fundamental Driver Matrix Implementation Plan
### Phase A (PA only)
- Build one production-quality PA driver panel with 6-10 fields.
- Include update frequency, last refresh, and confidence score.
- Feed only into conviction/sizing metadata for now.

### Phase B (GC/CL/NG)
- Template and replicate once PA panel is stable.

### Phase C (full commodity set)
- Expand to HG/SI/ag complex only after operational stability.

## R-Multiple Engine Integration Design (v1)
- Inputs:
  - Equity
  - Base risk % per trade
  - Signal grade (`A/B/C`)
  - Conviction multipliers (fundamental/regime)
- Outputs:
  - `position_r` (capped)
  - dollar risk
  - remaining daily/weekly/monthly `R` budget
- Safety:
  - reject run when budget is exhausted
  - emit alert with reason

## Kelly Module (Future, flagged)
- Add research module for `half-kelly` and `quarter-kelly` comparisons.
- Run OOS/walk-forward evaluation.
- Keep this as research artifact first; promote only after stability proof.

## Suggested 90-Day Build Sequencing
1. Weeks 1-2: R engine v1 + guardrails + tests
2. Weeks 3-4: PA driver panel v1
3. Weeks 5-6: spread sandbox v1
4. Weeks 7-8: rates overlay v1
5. Weeks 9-12: OOS/walk-forward sizing research and promote candidates

## Governance Notes
- Strategy catalogue remains an ideas and research reference.
- Promotion to production requires:
  - deterministic tests,
  - runbook integration,
  - acceptance criteria,
  - alerting and failure behavior.

## Immediate Next Action
Create Task H.13:
- `R` engine and risk-governor wiring in ops wrapper,
- with no strategy logic changes,
- and full unit test coverage.
