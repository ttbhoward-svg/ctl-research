# CTL Phase Gate Checklist (v2)
## Updated Feb 17, 2026 — Integrated Codex Round 2 (Universe Lock, Calibration, Drift Monitoring)

## Decision Metric Hierarchy (Fixed — Higher Rank Wins)
1. **MAR ratio** (return / max drawdown) — primary risk-adjusted metric
2. **OOS tercile spread** (top minus bottom tercile avg R) — scoring model predictive power
3. **OOS trade count** (statistical power and confidence)
4. **Net CAGR** (outcome measure, never overrides 1–3)

When metrics conflict, higher-ranked metric wins. CAGR never overrides MAR. Capital decisions use conservative planning base (20% net), not upside scenarios.

## Governance Rules (Always On)
- Pre-register hypotheses, features, thresholds, ranges, and pass/fail criteria before running analysis.
- No tuning on OOS. OOS is for validation only.
- Use purged/embargoed time-series CV (purge >= max holding period, default 30 trading days).
- Track immutable dataset hash, code commit hash, model config, and date boundaries in every model card.
- Gates are binary: `PASS` or `ITERATE`. No overrides by discretion/enthusiasm.
- Symbol universe: Ring 1 = 29 symbols (28 tradable + 1 research_only). Governed by Universe Lock Spec.
- **Universe lock file committed before each phase starts. No symbol changes without formal gate decision.**
- **Universe expansion and feature expansion cannot occur in same phase block.**
- **Strategy expansion cannot alter universe in same validation block.**
- Cluster taxonomy: 11 clusters (IDX_FUT, METALS_FUT, ENERGY_FUT, RATES_FUT, GRAINS_FUT, SOFTS_FUT, LIVESTOCK_FUT, FX_FUT, ETF_SECTOR, EQ_COMMODITY_LINKED, EQ_MACRO_BELLWETHER). Immutable within a phase.
- All symbols classified as `tradable` or `research_only`. Research_only symbols inform model coefficients but cannot generate live trades.
- **Research column promotion rule (per Gemini):** Research columns (currently 19) are for hypothesis generation ONLY. No research column may be promoted to a scoring feature without passing a fresh OOS validation in a subsequent phase. Promotion requires: (1) pre-registered hypothesis with expected sign, (2) fresh OOS sample not used in the exploratory analysis, (3) feature cap not exceeded. This prevents multiple-comparisons bias from turning noise into false features.

## Core Validation Invariants (All Phases)
- Costs ON (slippage + commission) for all backtests.
- Signal logic reproducibility chain intact (same inputs => same outputs).
- Negative controls pass:
  - randomized labels => near-zero explanatory power
  - lag-shift test fails as expected
  - placebo feature receives ~zero weight
- Parameter stability plateau confirmed (not single-point optimum).
- Slippage stress test: edge must survive at 2 ticks per side.
- Entry degradation test: edge must survive 30% delayed entries within tolerances.
- Kill criteria not triggered.

## Phase Gates

### Gate 1: Phase 1a → 1b (B1 + MTFA MVP)
Timeline: 6–8 weeks + 2-week contingency buffer (8–10 weeks realistic with non-coder + data plumbing)
Required (ALL 9 must pass):
1. OOS trades >= 30
2. Top-tercile minus bottom-tercile OOS spread >= +1.0R
3. Score-to-outcome monotonicity present (top > mid > bottom on avg R)
4. Feature cap respected (9 candidates + 1 cluster control, frozen)
5. Model card complete and reproducible (v2 template with all sections)
6. All three negative controls passed
7. Entry degradation test within tolerances (≤25% total R, ≤5pp win rate, ≤30% MAR) — **Phase 1a exploratory tolerances; may tighten for production gate (Gate 5)**
8. Slippage stress test: profitable at 2 ticks per side
9. Calibration: quintile scores directionally calibrated OOS (monotonically improving avg R)
Decision:
- PASS → start two-stage EV tests
- FAIL → iterate features/ranges/universe per pre-registered fallback only

### Gate 2: Phase 1b → Phase 2 (Two-stage EV)
Timeline: 2–3 weeks + 1-week buffer
Required:
- EV model improves OOS monotonicity vs single-stage baseline
- OOS spread not degraded beyond tolerance (no more than 0.2R reduction)
- MAR not degraded beyond tolerance (no more than 15% reduction)
- Complexity earns its place (documented incremental value with specific numbers)
- Cross-asset confirmation score shows monotonic OOS contribution (if implemented)
Decision:
- PASS → add regime/term-structure layer
- FAIL → keep single-stage model (this is a valid outcome, not a failure of the project)

### Gate 3: Ring N → Ring N+1 (Universe Expansion)
Timeline: 4–6 weeks per ring + 2-week buffer
Required:
- Expanded-universe OOS metrics within explicit degradation tolerances:
  - MAR: no more than 25% degradation vs prior ring
  - OOS tercile spread: no more than 20% degradation (0.2R absolute floor)
  - Profit factor: no more than 15% degradation
  - Win rate: no more than 10 percentage point degradation
- Per-cluster minimum: 15+ trades per cluster before cluster coefficients influence live sizing
- Feature portability mappings pre-registered and documented BEFORE ring expansion begins:
  - Equities: COT → institutional ownership change (13F) or short interest ratio
  - Equities: term structure → sector relative strength
  - Currencies: term structure → carry (interest rate differential)
- New model card generated with expanded dataset hash
- Theme concentration caps enforced in backtest and forward test (30% max per theme)
- All new symbols tagged `tradable` or `research_only` with justification
Decision:
- PASS → expand to next ring
- FAIL → investigate cluster drift before expansion

### Gate 4: Strategy Expansion (B2/F1/F2/F3)
Timeline: 3–4 weeks per strategy + 1-week buffer
Required:
- Prior strategy stack OOS-stable (no kill criteria triggered in last 50 trades)
- New strategy spec finalized + pre-registered (same rigor as B1 spec)
- Pooled model remains stable on OOS after adding new setup:
  - Tercile spread on existing strategies not degraded >15%
  - New strategy shows positive tercile spread independently
Decision:
- PASS → add strategy to pooled model
- FAIL → isolate and retest; do not contaminate working model

### Gate 5: Live / Capital Allocation
Required:
- OOS trades >= 30 (prefer 50)
- Survived at least one >= 8% drawdown regime cycle
- Shadow decomposition healthy:
  - Signal-only mechanical: positive cumulative R
  - Production mechanical: MAR >= 1.5
  - Actual execution: not more than 20% worse than production mechanical on cumulative R
- Capital policy constraints satisfied (per two-phase risk framework)
- Dashboard reproducibility test passes (all displayed values traceable to logged source data)
- Event de-risk protocol reduces tail-loss incidents vs baseline
Decision:
- PASS → allow staged capital increase per pre-registered schedule
- FAIL → no infusion; continue collecting OOS trades

## Symbol Promotion / Demotion Rules

### Promote `research_only` → `tradable` if ALL true:
1. OOS sample for symbol/cluster is sufficient (pre-registered minimum met)
2. Edge survives slippage stress test (symbol-specific realistic assumptions)
3. Score monotonicity is not broken by inclusion
4. No kill criteria triggered for symbol over validation window

### Demote `tradable` → `research_only` if ANY true:
1. OOS score-outcome relationship collapses (near-zero correlation)
2. Edge fails under realistic slippage stress
3. Parameter sensitivity is cliff-like (fragile)
4. Repeated data integrity concerns

All promotions/demotions logged with timestamp + reason code.

## Kill / Pause Criteria
| Condition | Action |
|-----------|--------|
| OOS top-tercile avg R < 0.5R | REJECT: absolute R too low after costs |
| OOS score-R correlation < 0.05 | REJECT: scoring model has no predictive power |
| Monotonicity failure (mid > top on OOS) | PAUSE: score ordering broken, investigate features |
| Rolling score drift (20-trade corr < 0) for 20 consecutive trades | PAUSE: model may need refit |
| < 30 OOS trades after 12 months forward collection | REJECT: insufficient evidence |
| > 60% of IS top-tercile trades from single cluster | REJECT: not generalizable |
| Negative controls fail | REJECT: data leakage or overfitting confirmed |
| Parameter sensitivity spike-shaped for ≥2 parameters | REJECT: likely curve-fit |
| Slippage stress: edge evaporates at 2 ticks | REJECT: execution-dependent edge |
| Entry degradation: total R collapses >50% | REJECT: timing-dependent edge |
| Two or more feature coefficients flip sign between model fits | PAUSE: possible regime shift |

Action: `REJECT` or `PAUSE` per criterion; document root cause before any restart. PAUSE requires investigation report before resuming. REJECT requires full respecification.

## Reporting Cadence
- Every regression run: model card + data health report + negative control results + calibration
- Every 50 trades: shadow portfolio decomposition review + drift diagnostics
- Quarterly: parameter drift, feature decay, coefficient stability, gate status review, metric hierarchy check, SAN cap review
- Annually: full system review — is the edge persisting? Should capital policy change?

## Minimum Trade Count Rules
- If total triggers < 80: results labeled `exploratory_only`
- Per-cluster minimum before coefficient affects sizing: 15+ OOS trades/cluster (pre-registered)
- Symbols below threshold remain in model as informational but cannot drive live sizing
