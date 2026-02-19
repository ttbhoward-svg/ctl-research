# CTL Final Lock Summary

## Lock Metadata
- Lock date: **February 17, 2026**
- Locked by: **Tim + Claude (research/PM)**
- Reviewed by: **Codex (methodology), Gemini (statistics)**
- Effective phase start: **Phase 1a**
- Notes: All 10 reconciliation checklist items resolved. Gemini passed with no structural objections. Codex reconciliation completed. Ready for build.

## Canonical Parameters (Authoritative)

### Universe
- Phase 1a Ring 1 count: **29**
- Tradable count: **28**
- Research-only count: **1 ($SBSW)**
- Cluster taxonomy version: **11 clusters** (IDX_FUT, METALS_FUT, ENERGY_FUT, RATES_FUT, GRAINS_FUT, SOFTS_FUT, LIVESTOCK_FUT, FX_FUT, ETF_SECTOR, EQ_COMMODITY_LINKED, EQ_MACRO_BELLWETHER)

### Feature Scope (Phase 1a)
- Feature cap statement: **9 candidates + 1 cluster control (frozen)**
- Model type: **Elastic Net (ElasticNetCV with StandardScaler pipeline)**
- Research-column governance statement: **19 research columns logged for hypothesis generation ONLY. No promotion to scoring feature without fresh pre-registered OOS validation in Phase 2+.**

### Validation/Gate Thresholds
- OOS minimum trades: **≥ 30**
- OOS top-bottom tercile spread: **≥ +1.0R**
- Monotonicity requirement: **Top > Mid > Bottom avg R**
- Slippage stress requirement: **Profitable at 2 ticks per side**
- Entry degradation tolerances: **≤25% total R, ≤5pp win rate, ≤30% MAR** (Phase 1a exploratory; may tighten for Gate 5)
- Purge gap: **30 trading days** (or max holding period, whichever is larger)

### Risk Framework
- Phase A base risk: **1.5-2.0% per trade**
- SAN cap: **$11,000 fixed, reviewed quarterly (next: May 17, 2026)**
- Phase A DD brakes: **10% / 15% / 20%**
- Phase B trigger: **Account > $2M for 30 consecutive days**
- Phase B base risk: **1.0% per trade**
- Phase B DD brakes: **5% / 10% / 15%**

### Execution Convention
- Stop evaluation rule: **Close-based. Stop breaches when Close[bar] < StopPrice.**
- Stop fill rule: **Next-bar open + slippage. ExitPrice = Open[bar+1] - SlippagePerSide.**
- TP fill rule: **At TP level exactly (resting limit order assumption). TP hit when High[bar] >= TP_level.**
- Collision precedence: **TP wins (default). Configurable via CollisionRule parameter.**

## Reconciliation Completion
- Reconciliation checklist path: `PreBuild_Reconciliation_Checklist.md`
- All checklist items complete: **YES**
- Outstanding decision points: **None blocking. Entry degradation tolerances marked as Phase 1a exploratory (Item 8 — decision: keep current, tighten at Gate 5 if warranted).**

### Checklist Resolution Log
1. ✅ SAN cap: $11,000 / DD brakes 10/15/20% — all docs aligned
2. ✅ OOS spread: +1.0R — standardized across all docs including capital infusion
3. ✅ Feature cap: "9 candidates + 1 cluster control (frozen)" — all variant phrasing corrected
4. ✅ Stop/exit convention: Explicit Execution Conventions paragraph added to B1 Logic Spec
5. ✅ Purge gap: 30 trading days — no stale 44-day references found
6. ✅ Universe lock: 29 symbols (28T+1R) — all stale "20 symbols" / "6 symbols" / "$RKLB" references corrected
7. ✅ Research column governance: Rule added to Phase Gate Checklist, Tracker, and Synthesis
8. ✅ Entry degradation: Marked as Phase 1a exploratory tolerance (decision logged)
9. ✅ Model naming: All forward-looking "Lasso" references changed to "Elastic Net"
10. ✅ Final Lock Summary generated with SHA-256 hashes

## Document Hash Manifest (Post-Lock)

| Document | Version | SHA-256 |
|---|---|---|
| B1 Strategy Logic Spec | v2 | `a76e6b1333bfc5514c8e6d61c431ca2bda2c8b75496364594f8f543eaacebde6` |
| Phase 1a Project Tracker | v3 | `3dbfb3e4ee17c798923500d5d7faeb7f72f28693f5b0fe27fe665ce066f1040e` |
| Phase Gate Checklist | v2 | `904ba2b06c66c619f6a6f8c0d361f1f60334e9a8a765f91f33a1656468d8ec3c` |
| Research Infrastructure Synthesis | v2 | `719c7ff12ca0905e77a10e7c1983cba757ee90d98483c7ed6f4a547a6011357c` |
| Final Pre-Build Status | v1 | `992bb3ce9377ec31fbc8114004accd92e656c23c269ce8cc4daa324bf93bafc3` |

## Build Authorization
- Coding authorized: **YES**
- Authorized scope: **Phase 1a only**
- First build task ID: **Task 1 (Data Ingest via Databento)**
- Change-freeze rules during build: **No spec changes without formal pause + reason code. Research discoveries logged in Task 12 notes, NOT integrated into spec until Phase 1a complete. Any critical bug fix to spec requires re-hash and lock summary amendment.**

---

## Lock Amendment Log

### Amendment 1 — Data Source Migration (Feb 18, 2026)

**Type:** Infrastructure only. No strategy logic, thresholds, features, or risk framework changed.

**What changed:**
- B1 Logic Spec v2 → v2.1: Added Section 0 (Data Source Convention) — provider hierarchy, canonical schema, session definitions, roll policy, close type, symbol map versioning, data health gating
- Phase 1a Tracker v3 → v3.1: Task 1 rewritten for Databento API ingest + provider abstraction. Task 1.5 expanded to include ReconciliationEngine. Task 4 rewritten for trigger-parity validation (Databento vs TS archive). Architecture section updated.
- Phase Gate Checklist v2 → v2.1: Added Data Source Governance subsection (trigger-parity gate, health gating, model card data source metadata). Added 3 data-health kill/pause criteria.
- New document: CTL_Data_Cutover_Checklist_v1.md

**What did NOT change:**
- B1 trigger conditions (C1-C4), entry logic, stop logic, TP logic, fib calculations
- Feature set (9 candidates + 1 control), research columns (19)
- Gate thresholds (OOS ≥30, tercile ≥1.0R, monotonicity, slippage, entry degradation)
- Risk framework ($11,000 SAN, 10/15/20% DD brakes)
- Universe (29 symbols, 11 clusters)
- Purge gap (30 days), CV method, negative controls

**Why:** Remove manual CSV export bottleneck. Enable automated daily ETL, health checks, and remote operation. Improve reproducibility with deterministic API pulls + manifest hashes.

**Validation gate:** Layered reconciliation gate (L1-L5) must PASS before Databento becomes primary. Exact TS continuous-price parity is not required; futures divergences must be fully explainable via documented roll schedule and gap differences, with real mismatches = 0 per cutover checklist.

### Amendment 2 — Close Type + Continuous Contract Reconciliation (Feb 18, 2026)

**Type:** Infrastructure + data convention. No strategy logic, thresholds, features, or risk framework changed.

**What changed:**
- B1 Logic Spec Section 0.5: Close type for futures changed from "last trade" to "settlement price"
  - Rationale: settlement is the exchange's official daily close, used by TS/Bloomberg/Norgate/all institutional systems. Using last-trade caused persistent small divergences compounding through EMA, producing false trigger mismatches.
  - Equities/ETFs unchanged (last trade = closing auction)
- Data Cutover Checklist: Signal Equivalence section rewritten for layered reconciliation (L1-L5) and functional parity instead of exact TS matching
- Phase 1a Tracker Task 4: Rewritten for layered reconciliation approach
- New document: CTL_Continuous_Contract_Reconciliation_Spec_v1.md

**What did NOT change:**
- B1 trigger conditions, entry/stop/TP logic, features, gate thresholds, risk framework, universe
- The "close" field still evaluates stops, still feeds EMA — only which price fills that field changed for futures

**Canonical interpretation note:**
- TradeStation is a bridge diagnostic reference only and is not the long-term canonical truth source.
- Exact continuous-price parity with TS is not required for cutover approval.
- Cutover approval requires layered reconciliation pass (L1-L5), with all futures-series divergences fully explainable via documented roll schedule and gap differences.

### Updated Document Hashes (Post-Amendment 2)

| Document | Version | Post-Amend 1 Hash | Post-Amend 2 Hash |
|---|---|---|---|
| B1 Logic Spec | v2.1 | `4b356282...` | `1180954360bfc4bebd2c40585b2f32e22106fae63059a6b0b5510d089d7700e9` |
| Phase 1a Tracker | v3.1 | `82bd399f...` | `6ee30d7341b60f95e000c144215080372a4638288e0003c33439e31d3b905630` |
| Phase Gate Checklist | v2.1 | `872eaa61...` | *(unchanged from Amend 1)* |
| Synthesis | v2 | `719c7ff1...` | *(unchanged)* |
| Final Pre-Build Status | v1 | `992bb3ce...` | *(unchanged)* |
| Data Cutover Checklist | v1 | `e89d15dd...` | `d1bbd8c22205b359c4089a7ea8cb726c8e46805f285c6bf64195ed5dab5bb98e` |
| Continuous Contract Reconciliation Spec | v1 | — | `1bbe70b8dd080a7a67e5635fbbcc5300619520b41956436d861515e31aa2571d` |

---

## Signoff
- PM signoff: **Claude** ✅
- Methodology signoff: **Codex** ✅ (reconciliation checklist passed)
- Statistical signoff: **Gemini** ✅ (final review passed, no structural objections)
- Amendment 1 signoff: **Claude** ✅ (infrastructure only, methodology preserved)
- Amendment 2 signoff: **Claude + Gemini + Codex** ✅ (close type convention, layered reconciliation)
- Operator signoff: **Tim** _____________________ (date: _____________)
