# CTL Data Source Cutover Checklist — v1
## Feb 18, 2026

## Purpose
Validate that Databento API data is equivalent to TradeStation archived data for B1 strategy purposes. All criteria must PASS before Databento becomes the primary canonical data source. This is an infrastructure change — no strategy logic, thresholds, features, or risk framework are modified.

---

## Migration Summary

| Role | Before | After |
|------|--------|-------|
| Primary (canonical) | TradeStation CSV exports | **Databento API** |
| Secondary (reconciliation) | *(none)* | **Norgate EOD** |
| Tertiary (spot-check) | *(none)* | **TradeStation archived CSVs** |
| Execution | IBKR (Phase 4+) | IBKR (Phase 4+) — unchanged |

---

## Pre-Cutover Checklist (ALL must pass)

### Infrastructure Build

| # | Criterion | Pass/Fail | Notes |
|---|-----------|-----------|-------|
| I-1 | `DataProvider` interface implemented with `get_ohlcv()` returning canonical schema | ☐ | |
| I-2 | `databento_provider.py` implemented and tested | ☐ | |
| I-3 | `norgate_provider.py` implemented and tested | ☐ | |
| I-4 | `tradestation_provider.py` reads archived CSVs, outputs canonical schema | ☐ | |
| I-5 | `symbol_map_v1.yaml` created with all 29 symbols mapped across DB/Norgate/TS/IBKR | ☐ | |
| I-6 | Symbol map SHA-256 hash recorded | ☐ Hash: `________________` |
| I-7 | Config file stores session definitions per asset class (not hardcoded) | ☐ | |
| I-8 | Config file stores roll method, close type, and provider priority | ☐ | |
| I-9 | Provider swap via config change only — no code changes required | ☐ | |
| I-10 | Data health artifact generator produces `data_health_YYYYMMDD.json` | ☐ | |

### Data Equivalence

| # | Criterion | Threshold | Pass/Fail | Artifact Path |
|---|-----------|-----------|-----------|---------------|
| D-1 | **Roll schedule reconciliation** — compare Databento canonical roll dates vs TS archive roll dates for /PA, /ES, /CL, /GC, /ZB (2018-2024). Differences allowed if fully documented with per-roll gap impact. | 100% of roll-date differences logged with per-roll gap impact; zero unexplained step changes | ☐ | `CTL_Phase1a_Data/validation/L2_roll_schedule_comparison.csv` |
| D-2 | **Daily OHLCV parity** — 50 random daily bars per symbol class (metal, energy, index, bond, grain, ETF, equity) | Close within 1 tick, Volume within 5% | ☐ | `CTL_Phase1a_Data/validation/ohlcv_parity_daily.csv` |
| D-3 | **H4 bar timestamp alignment** — 20 random H4 bars per asset class | Anchoring matches convention (futures=top-of-hour, equities=09:30) | ☐ | `CTL_Phase1a_Data/validation/h4_timestamp_check.csv` |
| D-4 | **Weekly bar boundary** — 10 random weekly bars per symbol class | Last completed exchange week boundary matches (per exchange calendar), weekly OHLC consistent | ☐ | `CTL_Phase1a_Data/validation/weekly_boundary_check.csv` |
| D-5 | **Historical depth** — all 29 symbols have data from 2015-01-01 or earlier | No gaps > 1 month in any symbol | ☐ | `CTL_Phase1a_Data/validation/historical_depth.csv` |
| D-6 | **Corporate actions** — compare 7 equity/ETF symbols on all known split dates since 2015 | Price adjustment matches within 0.1% | ☐ | `CTL_Phase1a_Data/validation/corporate_actions.csv` |
| D-7 | **Close type verification** — compare /CL and /ZB daily close (Databento settlement vs Norgate/TS) on 30 random dates. Confirms futures use settlement close convention; equities/ETFs use last-trade convention. | Max divergence: 1 tick on sampled futures closes vs reference settlement-compatible source | ☐ | `CTL_Phase1a_Data/validation/close_type_check.csv` |

### Signal Equivalence (THE CRITICAL GATE — Layered, per Reconciliation Spec)

**Important:** Do not target exact TS price parity. Target functional parity — same triggers
fire on the same dates, with divergences traceable to documented roll date differences.
See CTL_Continuous_Contract_Reconciliation_Spec_v1 for full methodology.

| # | Criterion | Threshold | Pass/Fail | Artifact Path |
|---|-----------|-----------|-----------|---------------|
| S-1 | **L1: Raw contract bar parity** — compare unadjusted outright contracts (ESH24, CLJ24, etc.) between Databento and Norgate | OHLC within 1 tick, Volume within 0.5% | ☐ | `CTL_Phase1a_Data/validation/L1_raw_contract_parity.csv` |
| S-2 | **L2: Roll schedule documented** — our canonical roll dates vs TS roll dates enumerated, each difference explainable | All differences logged with gap impact | ☐ | `CTL_Phase1a_Data/validation/L2_roll_schedule_comparison.csv` |
| S-3 | **L3: Roll gaps validated** — gap delta per roll event within tolerance | Within 2 ticks per roll event | ☐ | `CTL_Phase1a_Data/validation/L3_roll_gap_comparison.csv` |
| S-4 | **L4: Cumulative drift explainable** — all drift between canonical and TS traceable to documented roll differences | Unexplained drift <0.1% of price | ☐ | `CTL_Phase1a_Data/validation/L4_adjusted_series_drift.csv` |
| S-5 | **L5: Trigger parity — /PA daily** | Matching or borderline (<0.1% EMA-price gap) | ☐ | `CTL_Phase1a_Data/validation/L5_trigger_parity_PA.csv` |
| S-6 | **L5: Trigger parity — /ES daily** | Matching or borderline | ☐ | `CTL_Phase1a_Data/validation/L5_trigger_parity_ES.csv` |
| S-7 | **L5: Trigger parity — /CL daily** | Matching or borderline | ☐ | `CTL_Phase1a_Data/validation/L5_trigger_parity_CL.csv` |
| S-8 | **L5: Trigger parity — /GC daily** | Matching or borderline | ☐ | `CTL_Phase1a_Data/validation/L5_trigger_parity_GC.csv` |
| S-9 | **L5: Trigger parity — /ZB daily** | Matching or borderline | ☐ | `CTL_Phase1a_Data/validation/L5_trigger_parity_ZB.csv` |
| S-10 | **L5: Trigger parity — XLE daily** (ETF, no roll) | Identical trigger dates (zero mismatch) | ☐ | `CTL_Phase1a_Data/validation/L5_trigger_parity_XLE.csv` |
| S-11 | **L5: Trigger parity — $XOM daily** (equity, no roll) | Identical trigger dates (zero mismatch) | ☐ | `CTL_Phase1a_Data/validation/L5_trigger_parity_XOM.csv` |
| S-12 | **L5: Trade outcome parity** — matched triggers from /PA, /ES, /CL | R-multiples within ±0.1R | ☐ | `CTL_Phase1a_Data/validation/L5_trade_parity.csv` |
| S-13 | **Roll manifest complete** — every roll event for all 15 futures symbols logged | 100% coverage, all fields populated | ☐ | `CTL_Phase1a_Data/manifests/` |
| S-14 | **Borderline triggers < 5%** of total triggers across all futures symbols | <5% | ☐ | Computed from S-5 through S-9 |
| S-15 | **Real mismatches = 0** after accounting for roll differences (EMA gap >0.1% of price) | Zero | ☐ | Computed from S-5 through S-9 |

**ETFs and equities (S-10, S-11) have NO roll logic and must match exactly.** If these fail,
the problem is session definition or corporate actions, not roll methodology.

**Futures (S-5 through S-9) use functional parity.** Borderline triggers (EMA within 0.1%
of price on trigger date, trigger fires in one source but not the other) are acceptable
up to 5% of total. Real mismatches (EMA clearly on wrong side) must be zero.

### Reconciliation Engine

| # | Criterion | Pass/Fail | Artifact Path |
|---|-----------|-----------|---------------|
| R-1 | Automated reconciliation runs on every ingest (no manual trigger) | ☐ | |
| R-2 | Bar count parity check: Databento vs Norgate (tolerance: ±2 bars/year) | ☐ | |
| R-3 | OHLCV tolerance check: Close within 1 tick, Vol within 5% | ☐ | |
| R-4 | Roll date parity: within ±1 trading day per contract per year | ☐ | |
| R-5 | Missing/duplicate bar detection | ☐ | |
| R-6 | Status output: OK / WATCH / ALERT per symbol | ☐ | |
| R-7 | ALERT status blocks downstream processing (scoring, signal generation) | ☐ | |
| R-8 | Reconciliation report persisted as artifact | ☐ | `CTL_Phase1a_Data/health/reconciliation_YYYYMMDD.json` |

### Data Health Governance

| # | Criterion | Pass/Fail | Artifact Path |
|---|-----------|-----------|---------------|
| H-1 | Daily health artifact generated with: source metadata, row counts, checksums, reconciliation stats, final status | ☐ | `CTL_Phase1a_Data/health/data_health_YYYYMMDD.json` |
| H-2 | Health status propagates to morning brief (when Phase 2 brief is live) | ☐ | *(Phase 2)* |
| H-3 | Health history queryable (can look back at last 30 days of health status) | ☐ | `CTL_Phase1a_Data/health/` |

---

## Cutover Decision Matrix

| Outcome | Action |
|---------|--------|
| L1 passes + L2 documented + L4 drift explainable + L5 functional parity (borderline <5%, real mismatch = 0) + ETF/equity exact match | **GO.** Canonical series is primary. TS archive frozen. |
| All above but borderline triggers = 5-10% | **GO with WATCH.** Document all borderline triggers. Re-check after 30 days of live data. |
| L1 passes but L4 has unexplained drift >0.1% | **NO-GO.** Missing roll event or session mismatch. Run diagnostic sequence (Section 4 of Reconciliation Spec). |
| Any real mismatch (EMA >0.1% away from price, trigger fires in one but not other) | **NO-GO.** Fundamental series divergence. Investigate L2/L3. |
| L1 fails (raw contract bars diverge >1 tick) | **NO-GO.** Data source problem. Do not proceed to any higher layer. |
| ETF/equity triggers diverge (no roll logic involved) | **NO-GO.** Session definition or corporate action problem. Fix before cutover. |

---

## Post-Cutover Monitoring (First 2 Weeks)

| Day | Check | Action if Fail |
|-----|-------|----------------|
| Day 1-3 | Reconciliation report clean (all OK) | Investigate any WATCH/ALERT immediately |
| Day 3 | Spot-check 5 random historical bars against TS archive | If divergence: pause and investigate |
| Day 7 | Review all health artifacts from week 1 | If any ALERT occurred: root cause analysis |
| Day 14 | Full reconciliation report review | If clean: TS formally demoted. If issues: extend monitoring. |

---

## Rollback Plan

If cutover fails or data integrity degrades post-cutover:
1. Switch config back to `tradestation_provider.py` (reads archived CSVs)
2. No code changes required (provider abstraction)
3. Investigate root cause using validation artifacts
4. Re-attempt cutover only after root cause resolved and all S-criteria re-pass

---

## Completion Log

- [ ] All I-criteria passed (infrastructure build)
- [ ] All D-criteria passed (data equivalence)
- [ ] All S-criteria passed (signal equivalence) — **THE CRITICAL GATE**
- [ ] All R-criteria passed (reconciliation engine)
- [ ] All H-criteria passed (health governance)
- [ ] Cutover decision: `GO / NO-GO / CONDITIONAL GO`
- [ ] Decision date: _______________
- [ ] Decision by: _______________
- [ ] Post-cutover monitoring start: _______________
- [ ] TS formal demotion date (after 14-day clean monitoring): _______________

---

*Created: Feb 18, 2026*
*Type: Infrastructure amendment — no strategy logic, thresholds, or risk framework changed*
*Referenced by: Lock Summary Amendment 1*
