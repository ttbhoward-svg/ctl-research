# Detour Backlog

Purpose: Track non-critical detours so they do not get lost, while preserving
focus on the canonical tracker path.

## Priority rubric

- `P0` Critical to current gate/run safety.
- `P1` High impact to current cycle but not blocking.
- `P2` Valuable improvement, defer without immediate risk.
- `P3` Idea parking lot / long-horizon.

## Active deferred items

| Date | Item | Why deferred | Priority | Revisit trigger |
|---|---|---|---|---|
| 2026-03-02 | PL window-specific gap correction promotion | In-sample gains did not survive generalized/walk-forward checks | P1 | New causal features or vendor metadata available |
| 2026-03-02 | ES harmonization promotion candidate | Strict walk-forward worsened drift | P1 | New robust forward candidate with holdout improvement |
| 2026-03-02 | Alternative strategy catalogue (GEX/dark pool/overlay suite) | Outside immediate Phase1a/cutover gating critical path | P2 | After research-tier onboarding pipeline is live |
| 2026-03-02 | Full R-engine expansion (Kelly/conviction sizing optimization) | Important, but not blocking current gate-first execution | P2 | After ticker onboarding + stable batch backtests |
| 2026-03-02 | COT source expansion: Disaggregated Futures-Only feature set | Need canonical Legacy Futures-Only baseline first to avoid moving-target drift | P2 | Legacy COT baseline is stable and ablation framework is ready |
| 2026-03-02 | COT source expansion: TFF Futures-Only features for rates/index futures | Valuable nuance for financial futures, but not required for current cutover gates | P2 | After Disaggregated COT features are integrated and benchmarked |
| 2026-03-02 | COT cross-report delta features (Legacy vs Disaggregated vs TFF) | Higher complexity; requires robust schema alignment + anti-lookahead controls | P3 | After single-report COT variants show incremental lift in holdout tests |

## Operating rule

Before starting a detour:
1. State reason and expected impact.
2. Assign priority (`P0`..`P3`).
3. If not `P0`, add/update this file and continue canonical path first.
