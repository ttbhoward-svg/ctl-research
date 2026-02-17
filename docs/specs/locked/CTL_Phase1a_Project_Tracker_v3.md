# CTL Research Infrastructure — Phase 1a Project Tracker (v3)
## Updated Feb 17, 2026 — Integrated Codex Round 2 (Universe Lock, Dashboard, Statistical Additions)

## Architecture: Python-First

Signal detection, trade simulation, data pipeline, and regression all run in Python. TradeStation is used for two things only: (1) exporting historical continuous futures OHLCV data, and (2) live order execution (Phase 4+).

**Why:** Python scales from 29 symbols to 75+ in a loop. No EasyLanguage debugging. One language for everything. AI (Claude Code) is dramatically better at Python than EasyLanguage. The migration bottleneck that would hit at Phase 2 is eliminated.

**Risk mitigation:** The Python simulator is cross-validated against TradeStation's native backtester on 2–3 symbols before any results are trusted.

**Universe governance:** Core + Satellite structure per Codex Universe Lock Spec. All Phase 1a symbols are `tradable` except $SBSW (`research_only` — data informs model but cannot generate live trades). No symbol changes without formal phase-gate decision. Universe lock file committed before build starts.

---

## The Only Rule: Phase 1a Is The Only Thing That Exists

Everything else is safely documented in the spec (v2.2) and synthesis. It will be there when you earn the right to open it by completing Phase 1a.

---

## Scope (Frozen)

- **Strategy:** B1 only
- **Symbols (29 — 28 tradable + 1 research_only):**
  - IDX_FUT: /ES, /NQ, /YM, /RTY (4)
  - METALS_FUT: /PA, /GC, /SI, /HG, /PL (5)
  - ENERGY_FUT: /CL, /NG (2)
  - RATES_FUT: /ZB, /ZN (2)
  - GRAINS_FUT: /ZC, /ZS (2)
  - ETF_SECTOR: XLE, XME, GDX, XLF, XLI, XLK, XLY (7)
  - EQ_COMMODITY_LINKED: $XOM, $FCX, $NEM, $CAT (tradable), $SBSW (research_only) (5)
  - EQ_MACRO_BELLWETHER: $GS, $MU (2)
- **Timeframes:** Daily + Weekly (58 configurations)
- **Features:** 9 candidates + 1 cluster control:
  - Technical: BarsOfAir, Slope_20, CleanPullback, %R_Divergence
  - MTFA: WeeklyTrendAligned, MonthlyTrendAligned
  - External: COT_Commercial_Pctile_3yr, COT_Commercial_Zscore_1yr, VIX_Regime
  - Control: AssetCluster (11-cluster taxonomy — categorical fixed effect)
  - COT features are NULL for ETF and equity clusters (no imputation)
- **Model:** Elastic Net regression (L1+L2) with StandardScaler pipeline + cluster fixed effects on Theoretical R
  - Pre-regression: correlation heatmap — drop any feature pair >90% correlated
  - Fallback A: logistic Elastic Net on win/loss (reached TP1?)
  - Fallback B: Elastic Net on MFE_R
- **R-Multiple tracking:** Actual R + Theoretical R + Day 1 Fail flag
- **Cross-validation:** TimeSeriesSplit, 30-day purge gap (≥ max holding period per Codex)
- **Shadow portfolio: THREE curves** (per Codex):
  1. Signal-only mechanical (flat 1%, every qualifying trigger)
  2. Production mechanical (MAR 2.0 module applied)
  3. Actual execution (what you really did)
- **Sizing:** Two-Phase Risk Framework. Phase A (Accumulation, $550K-$2M): 1.5-2.0% base risk, 10%/15%/20% DD brakes. Phase B (Preservation, $2M+): 1.0% base risk, 5%/10%/15% DD brakes. Transition at $2M for 30 consecutive days. SAN cap: $11,000 fixed, reviewed quarterly (next review: May 17, 2026).
- **Pass/fail:** Top tercile ≥ 1.0R above bottom tercile on 30–50 OOS trades. Quintile calibration directionally monotonic.
- **Kill criteria (pre-registered):**
  - Reject if: OOS top tercile avg R < 0.5R (absolute R too low even if spread exists)
  - Reject if: fewer than 30 OOS trades after 12 months of forward testing
  - Reject if: parameter sensitivity spike-shaped for ≥2 core parameters
  - Reject if: >60% of IS top-tercile trades from single cluster (not generalizable)
  - Reject if: slippage stress — edge gone at 2 ticks
  - Reject if: entry degradation — total R collapses >50%
- **Pre-registered sample rule:** With 29 symbols, expect 840–1110 triggers. If < 80, expand further.

---

## Weeks 1–2: Data Acquisition & Simulator Build

Goal: get clean OHLCV data for all 29 symbols (daily/weekly/monthly/H4), build a Python trade simulator, and validate it against TradeStation. (Timeline: 6–8 weeks + 2-week contingency = 8–10 weeks total for Phase 1a. Per Codex: aggressive with non-coder + data plumbing, buffer is essential.)

### Task 1: Export Historical Data from TradeStation
- [ ] Export daily OHLCV for 28 tradable symbols:
  - Futures (continuous back-adjusted, 15): @PA, @GC, @SI, @HG, @PL, @CL, @NG, @ZC, @ZS, @ES, @NQ, @YM, @RTY, @ZB, @ZN
  - ETFs (7): XLE, XME, GDX, XLF, XLI, XLK, XLY
  - Equities (6): $XOM, $FCX, $NEM, $CAT, $GS, $MU
- [ ] Export weekly OHLCV for same 28 symbols
- [ ] Export monthly OHLCV for same 28 symbols
- [ ] Export H4 OHLCV for same 28 symbols (context data, not signal detection timeframe)
  - **Bar anchoring note:** Futures H4 bars anchor to top of hour (clock-aligned). Equities/ETFs anchor to market open (9:30 AM ET). TradeStation handles this natively — export as-is. Do NOT cross-reference H4 timestamps between asset classes.
  - H4 needed for: 200 SMA context, EMA10 trend alignment, partial entry research
- [ ] Download $SBSW daily/weekly/monthly OHLCV via `yfinance` Python (research_only — not in TradeStation). H4 not available for SBSW — set H4 columns to NULL.
- [ ] Date range: 2015-01-01 to present (extra history for EMA warmup)
- [ ] Save to `CTL_Phase1a_Data/raw/` as CSV
- [ ] Spot-check: does /PA daily close on 2024-01-02 match chart? Check 3 dates per symbol class (1 metal, 1 energy, 1 ETF, 1 equity)
- [ ] **Note on back-adjustment:** Use R-multiples and % distances for analysis, not absolute price levels. Panama Canal adjustment shifts historical prices, making absolute fib levels from 5+ years ago unreliable.
- [ ] **Time estimate:** 3–4 hours (manual TradeStation export work — batch if possible, more symbols than v2)

### Task 1.5: Data Sanitizer
- [ ] **Prompt for Claude Code:** "Build a Python script that validates OHLCV CSV files for data integrity. Check for: missing trading days (gaps > 3 calendar days excluding weekends/holidays), bars where High < Low or Open/Close outside High-Low range, zero or negative volume bars, duplicate dates, and sudden price jumps > 15% day-over-day (flag as potential bad ticks or roll artifacts). For H4 data: verify bar timestamps match expected anchoring (futures = top of hour, equities/ETFs = 9:30 AM start). Output a report per symbol listing all issues found."
- [ ] Run on all 29 symbols (including SBSW) across all timeframes (daily/weekly/monthly/H4)
- [ ] Fix or flag any issues before proceeding
- [ ] **Time estimate:** 1 hour

### Task 2: Build B1 Signal Detector in Python
- [ ] **Input to Claude Code:** B1 Strategy Logic Specification + sample CSV files
- [ ] **Prompt:** "Build a Python class that implements the B1 signal detector exactly per this specification [paste B1_Strategy_Logic.md]. The class should: (1) take a daily OHLCV DataFrame and weekly/monthly DataFrames as input, (2) compute all indicators (EMA10 using ewm(span=10, adjust=False), Slope_20, BarsOfAir, WilliamsR), (3) identify trigger bars where all four conditions are met, (4) track the confirmation window and entry timing per the indexing rules, (5) output a DataFrame of triggers with all columns from Section 10 of the spec. Handle holiday-shortened weeks by using 'last available bar where Date <= Friday' for weekly alignment."
- [ ] Run on /PA daily as smoke test — does it produce triggers?
- [ ] **Time estimate:** 2–3 hours

### Task 3: Build Trade Simulator + Chart Inspector in Python
- [ ] **Prompt for Claude Code:** "Build a Python trade simulator that takes the B1 trigger DataFrame and daily OHLCV data, and walks forward bar-by-bar to simulate each trade. Implement: entry at next-bar open + slippage, close-based stop evaluation, TP detection (High >= TP level), partial exit with fractional rounding (Section 5.3 of B1 spec), same-bar TP/stop collision (TP wins), BOTH Actual R-multiple (account-constrained) and Theoretical R-multiple (unlimited splitting, all three partials execute). Track MFE_R and MAE_R. Apply slippage per instrument class. Also check: if Low of entry bar < StopPrice, flag as 'Day 1 Fail' (caution flag, trade still runs but logged)."
- [ ] **Also build:** "Add a Chart Inspector function using Plotly that, for any trade, generates an interactive chart showing: price bars, 10 EMA, 21 SMA, entry marker, stop level, TP1-TP3 levels, exit marker. This is for visual validation during chart study sessions."
- [ ] Run simulator on /PA daily — does it produce complete trade records?
- [ ] Use Chart Inspector to visually verify 3 trades — do the entry/exit markers make sense?
- [ ] **Time estimate:** 3–4 hours

### Task 4: Cross-Validate Python vs TradeStation
- [ ] Run B1 in TradeStation on **6–8 diverse symbols** covering all clusters, 2018–2024:
  - /PA daily (precious metal — primary validation)
  - /GC daily (precious metal — confirms metal cluster)
  - /CL daily (energy)
  - /ZC daily (grain)
  - /ES daily (equity index)
  - /ZB daily (bond)
  - Optional: /NQ daily (index), $CAT daily (equity) for extra coverage
- [ ] Export TradeStation trade lists for each
- [ ] Run Python simulator on same symbols, same period
- [ ] Compare: same trigger dates? Entry prices within 1 tick? Same exit bars? R-multiples within ±0.1R?
- [ ] **Target: ≥100 matched trades total across all symbols** (per Codex)
- [ ] Acceptable per-symbol: trade count within ±2, R-multiples within ±0.1R
- [ ] If discrepancies: use Chart Inspector to examine divergent trades visually
- [ ] **IF VALIDATED (100+ matched trades across 6+ symbols): proceed.**
- [ ] **IF MISMATCH: STOP. Debug until resolved. Nothing downstream is valid.**
- [ ] **Time estimate:** 4–5 hours
- [ ] **Note:** Requires building B1 in TradeStation EasyLanguage for validation symbols. One-time effort.

### Task 4b: Slippage Stress Test (Standard Model Card Output)
- [ ] Re-run full Python backtest at 0/1/2/3 ticks slippage per side
- [ ] Per-instrument tick values: /PA=$5, /GC=$10, /SI=$25, /HG=$12.50, /PL=$5, /ES=$12.50, /NQ=$5, /YM=$5, /RTY=$5, /CL=$10, /NG=$10, /ZB=$31.25, /ZN=$15.625, /ZC=$12.50, /ZS=$12.50. Equities/ETFs=$0.01.
- [ ] Record total R, win rate, avg R, MAR at each slippage level
- [ ] **Kill criterion: if edge evaporates at 2 ticks, it was never real. If profitable at 3 ticks, edge is robust.**
- [ ] Add slippage sensitivity table to model card
- [ ] **Time estimate:** 30 minutes

---

## Weeks 3–4: MTFA, Confluence, External Data

Goal: add all 9 features to the trigger dataset and merge external data for all 29 symbols.

### Task 5: Add MTFA Flags to Signal Detector
- [ ] **Prompt for Claude Code:** "Add WeeklyTrendAligned and MonthlyTrendAligned flags to the B1 signal detector per Section 9 of the B1 Strategy Logic Spec. Use the last COMPLETED weekly/monthly bar as of the trigger date. Weekly bar alignment: last Friday before trigger date (handle holiday-shortened weeks by finding last available bar where Date <= Friday). Monthly: last trading day of prior calendar month."
- [ ] Validate: pick 3 trigger dates across different symbols, manually check weekly chart — is the flag correct?
- [ ] **Time estimate:** 1–2 hours

### Task 6: Add All Confluence Flags
- [ ] **Prompt for Claude Code:** "Add these confluence flags to the B1 signal detector per Section 8 of the B1 Strategy Logic Spec: WR_Divergence, CleanPullback, VolumeDeclining, FibConfluence, GapFillBelow, MultiYearHighs, SingleBarPullback. Each must be computed at trigger time (bar N) using only data available at or before bar N."
- [ ] Spot-check 5 triggers across different symbols: do the flags match what you see on the chart? Use Chart Inspector.
- [ ] **Time estimate:** 2–3 hours

### Task 7: Integrate COT and VIX Data
- [ ] **Prompt for Claude Code:** "Write Python scripts to: (1) Download CFTC COT data for all 15 futures symbols using the `cot_reports` library. Calculate TWO features per symbol: a 3-year rolling percentile of commercial net position (structural), AND a 1-year z-score (responsive). Add a 'structural extreme' boolean flag for 5-year highs/lows. (2) Download VIX daily close from FRED (series VIXCLS). Create binary VIX_Regime = 1 if prior day close < 20. Both datasets must use publication-date timestamps — COT released Friday but reflects Tuesday positions, so timestamp as Friday."
- [ ] Merge into trigger database by date: each trigger gets the most recent COT values published BEFORE the trigger date, and the prior day's VIX
- [ ] For ETF triggers (XLE, XME, GDX, XLF, XLI, XLK, XLY): COT not applicable — set NULL. VIX still applies.
- [ ] For equity triggers ($XOM, $FCX, $NEM, $CAT, $GS, $MU, $SBSW): COT not applicable — set NULL. VIX still applies.
- [ ] COT NULLs are structurally meaningful — no imputation. Elastic Net handles via cluster fixed effects.
- [ ] Run Data Health Report: no nulls in critical fields (except COT for non-futures), no future-dated external data
- [ ] **Time estimate:** 2–3 hours

### Task 8: Final Database Assembly, Health Check & Unit Tests
- [ ] Merge all sources: triggers + confluence + MTFA + COT (both features) + VIX
- [ ] Assign AssetCluster label to each trigger using 11-cluster taxonomy:
  - IDX_FUT, METALS_FUT, ENERGY_FUT, RATES_FUT, GRAINS_FUT, ETF_SECTOR, EQ_COMMODITY_LINKED, EQ_MACRO_BELLWETHER
  - (SOFTS_FUT, LIVESTOCK_FUT, FX_FUT reserved for Phase 2+)
- [ ] Assign TradableStatus: 'tradable' for all except SBSW ('research_only')
- [ ] Run Data Health Report:
  - [ ] All 29 symbols present
  - [ ] No null values in: EntryPrice, StopPrice, TP1, SetupType, RMultiple_Actual, TheoreticalR
  - [ ] COT values present for all 15 futures symbols; NULL for all 14 ETF/equity symbols
  - [ ] COT dates lagged correctly (publication date before trigger date)
  - [ ] VIX uses prior day close (not same day)
  - [ ] No duplicate rows
  - [ ] R-multiples consistent with Entry/Stop/Exit math
  - [ ] Theoretical R computed correctly (all three partials)
- [ ] **IMMUTABLE DATASET VERSIONING (per Codex):**
  - [ ] Save final dataset as `phase1a_triggers_v1_YYYYMMDD.db`
  - [ ] Never overwrite — new versions get new filenames
  - [ ] Generate SHA-256 hash of the database file
  - [ ] Record hash in pre-registration document
- [ ] **UNIT TESTS (per Codex):**
  - [ ] Test: entry timing (N+2 rule) on a known trigger — confirm entry bar is correct
  - [ ] Test: HTF alignment — pick a date, manually verify weekly/monthly bar used
  - [ ] Test: TP/stop collision on a constructed scenario — confirm TP wins
  - [ ] Test: retrigger rule — two triggers 2 bars apart, confirm second is ignored
  - [ ] Test: stop at next-bar open — confirm exit is bar after breach, not same bar
  - [ ] Test: fractional contract rounding on 1, 2, 3 contracts
  - [ ] All tests must pass before proceeding to Weeks 5–6
- [ ] Count total triggers: _____ (target: 840–1110 with 29 symbols)
- [ ] Count per cluster: each cluster should have 15+ triggers
- [ ] Split dataset: IS = 2018-01-01 to 2024-12-31, OOS = 2025-01-01 to present
- [ ] **Time estimate:** 2–3 hours

---

## Weeks 5–6: Regression & Analysis

Goal: pre-register hypotheses, run the Elastic Net, study the results.

### Task 9: Pre-Register Hypotheses & Lock Reproducibility Chain
- [ ] Write down expected coefficient signs for each feature:
  - BarsOfAir: positive (more air = stronger trend)
  - Slope_20: positive (steeper momentum)
  - CleanPullback: positive (orderly decline)
  - %R_Divergence: positive (hidden strength)
  - WeeklyTrendAligned: positive (with the weekly trend)
  - MonthlyTrendAligned: positive (with the monthly trend)
  - COT_Commercial_Pctile_3yr: positive (smart money long)
  - COT_Commercial_Zscore_1yr: positive (recent positioning shift bullish)
  - VIX_Regime: positive (low fear)
- [ ] Set score threshold: "Top tercile from in-sample distribution = high conviction"
- [ ] Set pass/fail: "Top tercile OOS TheoreticalR ≥ 1.0R above bottom tercile, minimum 30 OOS trades"
- [ ] Set kill criteria (pre-registered):
  - OOS top tercile avg R < 0.5R → reject
  - Fewer than 30 OOS trades after 12 months → reject
  - Parameter sensitivity spike-shaped for ≥2 core params → reject
  - >60% of IS top-tercile trades from single cluster → not generalizable
  - Slippage stress: edge gone at 2 ticks → reject (execution-dependent)
  - Entry degradation: total R collapses >50% → reject (timing-dependent)
- [ ] Set parameter stability requirement: B1 must work at EMA 9/10/11
- [ ] Set fallback research questions:
  - Fallback A: predict win rate (binary: reached TP1?) using logistic Elastic Net
  - Fallback B: predict MFE_R using Elastic Net
  - Fallback C: univariate sanity checks on each feature individually
- [ ] Set Phase A risk parameters: "Base risk 1.5-2.0%. SAN cap $11,000 FIXED, reviewed quarterly. Next review: May 17, 2026. DD brakes: 10%/15%/20%."
- [ ] Set Phase B transition trigger: "Switch to Phase B when account > $2M for 30 consecutive days"
- [ ] Set drawdown thresholds: Phase A = 10%/15%/20%, Phase B = 5%/10%/15%
- [ ] **LOCK REPRODUCIBILITY CHAIN (per Codex):**
  - [ ] Record dataset file hash (SHA-256): ____________
  - [ ] Record code commit hash (git): ____________
  - [ ] Record random seed for all stochastic operations: ____________
  - [ ] Record exact IS boundary: 2018-01-01 to 2024-12-31
  - [ ] Record exact OOS boundary: 2025-01-01 to present
  - [ ] Record threshold-selection rule: top tercile from IS score distribution
  - [ ] Record ElasticNetCV alpha selection method: TimeSeriesSplit, 5 folds, 30-day purge
- [ ] **DATE-STAMP AND SAVE THIS DOCUMENT BEFORE RUNNING ANY ANALYSIS**
- [ ] **Time estimate:** 45 minutes. The most important 45 minutes of the project.

### Task 10: Correlation Check, Elastic Net Regression & Model Card (In-Sample)
- [ ] **Step 1 — Correlation Heatmap (per Gemini):**
  - [ ] Compute pairwise Pearson correlation for all 9 features
  - [ ] If any pair > 0.90: drop the less interpretable feature (pre-register the decision)
  - [ ] Visualize as Plotly heatmap, save to results folder
- [ ] **Step 2 — Feature Standardization (CRITICAL — per Gemini):**
  - [ ] Use scikit-learn `Pipeline` with `StandardScaler` before regression
  - [ ] Without this, Lasso/ElasticNet unfairly penalizes small-range features (COT 0-1) vs large-range (BarsOfAir 3-50)
  - [ ] `Pipeline([('scaler', StandardScaler()), ('model', ElasticNetCV(...))])`
- [ ] **Step 3 — Elastic Net Regression (per Gemini — replaces pure Lasso):**
  - [ ] DV: TheoreticalR. IVs: surviving features + AssetCluster one-hot fixed effects
  - [ ] Use `ElasticNetCV` (combines Lasso selection + Ridge stability for correlated features)
  - [ ] TimeSeriesSplit (5 folds, purge gap = 30 trading days)
  - [ ] Which features survived? Which zeroed out? Signs match hypotheses?
  - [ ] Per-cluster check: each cluster with 15+ trades show same directional relationship?
- [ ] **Step 4 — If model finds little / near-zero R²:**
  - [ ] Fallback A: Logistic ElasticNet on binary win/loss (reached TP1?)
  - [ ] Fallback B: ElasticNet on MFE_R
  - [ ] Fallback C: Univariate analysis — each feature individually vs TheoreticalR
  - [ ] If ALL show nothing: edge may be mostly discretionary
- [ ] **Step 5 — Model Card (per Codex):**
  - [ ] 1-page document: features, sample size (total + per cluster), CV config, coefficients, IS metrics, dataset hash, date, code commit
  - [ ] Save as `model_card_v1_YYYYMMDD.md`
- [ ] **Time estimate:** 2–3 hours

### Task 11: Parameter Sensitivity Sweep
- [ ] **Prompt for Claude Code:** "For each B1 parameter (SlopeThreshold: 4–14, MinBarsOfAir: 3–10, EntryGraceBars: 1–5, SwingLookback: 10–30), re-run the full signal detection + trade simulation at each value in the range. Plot total R, win rate, and trade count vs parameter value. Flag any parameter where results collapse within ±1 of the default."
- [ ] Is each parameter profitable across a plateau?
- [ ] If any parameter is spike-shaped: flag as potentially curve-fit
- [ ] **Time estimate:** 2–3 hours

### Task 11b: Negative Controls (per Codex — MANDATORY before proceeding)
- [ ] **Control 1 — Randomized Labels:** Shuffle TheoreticalR values randomly. Rerun ElasticNet. Model should show near-zero R² and no significant features. If it doesn't: DATA LEAKAGE. Stop and debug.
- [ ] **Control 2 — Lag-Shift Check:** Shift all features forward by 5 bars. Rerun ElasticNet. Model should perform WORSE than unshifted. If it improves: LOOK-AHEAD BIAS. Stop and debug.
- [ ] **Control 3 — Placebo Feature:** Add random noise column. Rerun ElasticNet. Random column should be zeroed out. If not: regularization too weak or sample too small.
- [ ] All three controls must PASS before proceeding to OOS validation
- [ ] Log results in model card
- [ ] **Time estimate:** 1 hour

### Task 11c: Entry Quality Degradation Test
- [ ] **Prompt for Claude Code:** "Re-run the full B1 trade simulation, but randomly delay 30% of entries by 1 additional bar (use random seed 42 for reproducibility). Compare degraded results to baseline: total R, win rate, MAR ratio."
- [ ] Degradation tolerances:
  - Total R: not worse than 25% vs baseline
  - Win rate: not worse than 5 percentage points vs baseline
  - MAR ratio: not worse than 30% vs baseline
- [ ] Run with robustness seeds (43, 44) — results should be directionally consistent
- [ ] **If edge collapses:** strategy depends on precision timing unrealistic to achieve live
- [ ] **If survives:** robust to reality of being slow/hesitant/asleep
- [ ] Log results in model card
- [ ] **Time estimate:** 30 minutes

### Task 12: Chart Study Sessions (THE MOST IMPORTANT TASK)
- [ ] Pull up the 10 highest-R trades that scored in the TOP tercile — what do they look like?
- [ ] Pull up the 10 most negative-R trades that scored in the TOP tercile — what went wrong?
- [ ] Pull up 10 trades that scored LOW but actually worked — what did the model miss?
- [ ] Pull up 10 trades that scored HIGH and failed — false confluence?
- [ ] Write 3–5 plain English observations about what you see
- [ ] **This builds the pattern recognition that no model can replace**
- [ ] **Time estimate:** 2–3 hours (do this in front of actual charts, not just data)

---

## Weeks 7–8: Out-of-Sample Validation & Decision

Goal: apply IS weights to OOS data and make the go/no-go call.

### Task 13: Out-of-Sample Test
- [ ] Score each OOS trigger using IS-derived Elastic Net coefficients
- [ ] Split OOS trades into terciles by score
- [ ] Calculate: average R per tercile, win rate per tercile, number of trades per tercile
- [ ] **THE QUESTION:** Top tercile R ≥ 1.0R above bottom tercile?
- [ ] Is the relationship monotonic across all three terciles?
- [ ] Total OOS trades: _____ (need 30–50; if fewer, note limitation)
- [ ] **CALIBRATION (per Codex Round 2):**
  - [ ] Compute Brier score on OOS data (baseline = naive frequency)
  - [ ] Generate reliability plot: 5 quintile bins, predicted avg score vs actual avg R
  - [ ] Quintile scores must be directionally calibrated (monotonically improving avg R)
  - [ ] Add calibration section to model card
- [ ] **Time estimate:** 1–2 hours

### Task 14: Pass/Fail Decision
- [ ] **IF PASS (monotonic, ≥1.0R spread, 30+ OOS trades):**
  - [ ] Phase 1a validated ✓
  - [ ] Document the IS tercile threshold as the pre-registered "high conviction" cutoff
  - [ ] Decide next step: Phase 1b (two-stage EV) or Phase 2 (expand universe with same features)
  - [ ] Begin shadow portfolio tracking
- [ ] **IF FAIL (non-monotonic, <1.0R spread):**
  - [ ] Check for data bugs first (most common cause of unexpected failure)
  - [ ] If data clean: is sample size too small? → Expand to Ring 2, same features
  - [ ] If sample adequate: do any INDIVIDUAL features show clear value? → Simplify model
  - [ ] Do NOT add features, do NOT add complexity
  - [ ] "Failure is data, not identity"
- [ ] **IF INCONCLUSIVE (too few OOS trades, borderline spread):**
  - [ ] Expand to Ring 2 (same 9 features, 15 additional symbols per universe lock)
  - [ ] Rerun with larger sample
  - [ ] Do NOT change features or parameters

### Task 15: Archive, Drift Setup & Report
- [ ] Save all code, data, regression outputs, sensitivity sweeps, OOS results
- [ ] **Set up drift monitoring pipeline (per Codex Round 2):**
  - [ ] Rolling 20-trade score-outcome correlation calculator
  - [ ] Per-feature coefficient comparison across model card versions
  - [ ] Alert thresholds: correlation < 0.10 for 20 trades = YELLOW; < 0.00 = RED (PAUSE)
  - [ ] Coefficient sign flip = YELLOW; two or more simultaneous flips = RED (PAUSE)
- [ ] Write 1-page Phase 1a Report: what worked, what didn't, what's next
- [ ] Prepare sense-check document for Gemini and Codex
- [ ] Update the synthesis and spec with findings
- [ ] **Time estimate:** 2–3 hours

---

## What You Are NOT Doing in Phase 1a

- ❌ Term structure / backwardation
- ❌ Bridgewater Quad sizing
- ❌ Macro composite score
- ❌ Nested HTF state machine
- ❌ Two-stage EV scoring
- ❌ More than 29 symbols (28 tradable + 1 research_only)
- ❌ More than 9 features + 1 control
- ❌ Any strategy other than B1
- ❌ Live trading from the model
- ❌ RadarScreen scanner / Streamlit dashboard
- ❌ Automation / n8n / OpenClaw
- ❌ Currencies, livestock, softs (Ring 2-4)
- ❌ Options flow, warehouse stocks, news sentiment
- ❌ Writing EasyLanguage code (Python handles detection + simulation)
- ❌ HMM / regime-switching models (Sandbox — Phase 2+)
- ❌ Meta-labeling model (schema logged, model not built)
- ❌ Daily Decision Cockpit / Theme Mapping / Event Protocol (Phase 2 dashboard)
- ❌ FedLiquidityExpanding or Above200DMA features (Phase 2 pre-registered)

---

## Red Team Notes (Pre-Build)

### The regression might find nothing
If Elastic Net zeros out most features, that IS a valid research result. Before concluding, run the pre-registered fallbacks: logistic Elastic Net on win/loss, Elastic Net on MFE. A factor might not predict how far a trade runs, but it might predict which trades never go against you.

### Sample size addressed by starting with 29 symbols
With 29 symbols on daily+weekly over 7 years, expect 840–1110 B1 triggers. Strong statistical power for 9 features + 1 control.

### Track Theoretical R alongside Actual R
At $550K with 1–2 contracts, most trades exit 100% at TP1. Theoretical R measures signal quality. Actual R measures account-constrained execution. You need both.

### Shadow portfolio runs at full size during validation
Mechanical baseline at 1% risk on paper regardless of drawdown brakes. Measures pure signal quality. Actual execution uses MAR 2.0 decelerator. Two curves, two purposes.

### Python vs TradeStation is the first gate
Cross-validate on 6–8 diverse symbols (metal, energy, grain, index, bond, ETF, equity). If they don't match, STOP and debug.

### Chart Inspector is not optional
Plotly chart inspector built alongside simulator. Chart study in DataFrames instead of visual charts = research with one eye closed.

### Data integrity comes before signal detection
Data Sanitizer (Task 1.5) runs before any analysis. Missing bars, bad ticks, roll artifacts corrupt everything downstream.

### SAN cap and drawdown thresholds are pre-registered
$11,000 SAN cap (fixed, reviewed quarterly — next: May 17, 2026), 10%/15%/20% drawdown brakes, parameter stability (EMA 9/10/11). Date-stamped before analysis.

### Research columns are for hypothesis generation only
Phase 1a logs 19 research columns alongside 9 scoring features. These research columns (H4 context, fib nest, touch depth, drag-up, alternative entries/stops/swing highs) cannot be promoted to scoring features without fresh OOS validation in Phase 2+, a pre-registered hypothesis with expected sign, and the feature cap not exceeded. Exploring 19 columns in Task 12 chart study will surface patterns — but patterns must survive formal validation before influencing live sizing.

---

## Daily Practice During the Build

- Morning: trade normally using your existing discretionary process + Dean's alerts
- Work session (1–3 hrs): complete the next task. One per session.
- Evening: one sentence in the project log about what you did and what's next.

16 tasks. 4 weeks. One at a time.
