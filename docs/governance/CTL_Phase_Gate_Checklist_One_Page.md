# CTL Phase Gate Checklist (One Page)

## Governance Rules (Always On)
- Pre-register hypotheses, features, thresholds, ranges, and pass/fail criteria before running analysis.
- No tuning on OOS. OOS is for validation only.
- Use purged/embargoed time-series CV (purge >= max holding period, default 30 trading days).
- Track immutable dataset hash, code commit hash, model config, and date boundaries in every model card.
- Gates are binary: `PASS` or `ITERATE`. No overrides by discretion/enthusiasm.

## Core Validation Invariants (All Phases)
- Costs ON (slippage + commission) for all backtests.
- Signal logic reproducibility chain intact (same inputs => same outputs).
- Negative controls pass:
  - randomized labels => near-zero explanatory power
  - lag-shift test fails as expected
  - placebo feature receives ~zero weight
- Parameter stability plateau confirmed (not single-point optimum).
- Kill criteria not triggered.

## Phase Gates

### Gate 1: Phase 1a -> 1b (B1 + MTFA MVP)
Required:
- OOS trades >= 30
- Top-tercile minus bottom-tercile OOS spread >= +1.0R
- Score-to-outcome monotonicity present
- Feature cap respected (MVP frozen set)
- Model card complete and reproducible
Decision:
- PASS -> start two-stage EV tests
- FAIL -> iterate features/ranges/universe per pre-registered fallback only

### Gate 2: Phase 1b -> Phase 2 (Two-stage EV)
Required:
- EV model improves OOS monotonicity vs single-stage baseline
- OOS spread and MAR not degraded beyond tolerance
- Complexity earns its place (documented incremental value)
Decision:
- PASS -> add regime/term-structure layer
- FAIL -> keep single-stage model

### Gate 3: Ring N -> Ring N+1 (Universe Expansion)
Required:
- Expanded-universe OOS metrics within 20% degradation vs prior ring
- Per-cluster minimum trade counts met
- Feature portability mappings pre-registered and documented
- New model card generated
Decision:
- PASS -> expand to next ring
- FAIL -> investigate cluster drift before expansion

### Gate 4: Strategy Expansion (B2/F1/F2/F3)
Required:
- Prior strategy stack OOS-stable
- New strategy spec finalized + pre-registered
- Pooled model remains stable on OOS after adding new setup
Decision:
- PASS -> add strategy
- FAIL -> isolate and retest

### Gate 5: Live/Capital Allocation
Required:
- OOS trades >= 30 (prefer 50)
- Survived at least one >=8% drawdown regime cycle
- Shadow decomposition healthy:
  - Signal-only mechanical positive
  - Production mechanical acceptable MAR
  - Actual execution not materially worse than production mechanical
- Capital policy constraints satisfied
Decision:
- PASS -> allow staged capital increase
- FAIL -> no infusion

## Kill / Pause Criteria
- OOS top-tercile avg R < 0.5R
- OOS score-R correlation < 0.05
- Monotonicity failure (mid bucket > top bucket)
- Rolling score drift (20-trade corr < 0)
- Fewer than 30 OOS trades after 12 months of forward collection
Action:
- `REJECT` or `PAUSE` per criterion; document root cause before any restart.

## Reporting Cadence
- Every run: model card + data health report
- Every 50 trades: shadow portfolio decomposition review
- Quarterly: parameter drift, feature decay, and gate status review
