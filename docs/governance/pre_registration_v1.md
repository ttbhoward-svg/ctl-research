# Phase 1a Pre-Registration Document (v1)

**Date:** 2026-02-17
**Strategy:** B1 (10-EMA Retest)
**Status:** LOCKED — no modifications after first model run

---

## 1. Hypothesis

A linear scoring model (Elastic Net on Theoretical R) using 9 pre-registered
features plus a cluster fixed effect can rank B1 trade setups such that
top-tercile scores produce meaningfully higher average R than bottom-tercile
scores out-of-sample.

**Primary metric:** Top-tercile minus bottom-tercile OOS average Theoretical R >= +1.0R.

**Secondary metrics:**
- MAR ratio (total R / max drawdown) >= 1.5 on production mechanical track
- Score-to-outcome monotonicity (top > mid > bottom on average R)
- Quintile calibration directionally monotonic OOS

---

## 2. Feature Set (Frozen — 9 Candidates + 1 Control)

| # | Feature | Type | Expected Sign | Rationale |
|---|---------|------|---------------|-----------|
| 1 | BarsOfAir | Technical, continuous | **Positive** | More air = stronger trend separation from EMA |
| 2 | Slope_20 | Technical, continuous | **Positive** | Steeper EMA momentum = stronger trend |
| 3 | CleanPullback | Technical, boolean | **Positive** | Orderly 3-bar decline = healthy retracement |
| 4 | WR_Divergence | Technical, boolean | **Positive** | Hidden bullish strength (price lower, %R higher) |
| 5 | WeeklyTrendAligned | MTFA, boolean | **Positive** | Trade with weekly trend = higher expectancy |
| 6 | MonthlyTrendAligned | MTFA, boolean | **Positive** | Trade with monthly trend = higher expectancy |
| 7 | COT_20D_Delta | External, continuous | **Positive** | Commercials adding longs = bullish structural |
| 8 | COT_ZScore_1Y | External, continuous | **Positive** | Extreme commercial positioning = structural tailwind |
| 9 | VIX_Regime | External, boolean | **Positive** | Low-vol environment (VIX < 20) favors trend trades |
| C | AssetCluster | Control, categorical | N/A | 11-cluster fixed effect (not a predictive feature) |

**Feature cap:** 9 candidates + 1 cluster control. Frozen for Phase 1a.
No feature may be added, removed, or substituted without a new pre-registration.

**COT applicability:** Futures only (15 symbols). ETFs and equities have NULL
COT values — handled by cluster fixed effects in the Elastic Net.

---

## 3. Model Specification

| Parameter | Value |
|-----------|-------|
| Model | Elastic Net (ElasticNetCV) |
| Dependent variable | TheoreticalR |
| Standardization | StandardScaler in Pipeline (before regression) |
| Cross-validation | PurgedTimeSeriesSplit, 5 folds |
| Purge gap | 30 days (calendar; see Task 4 assumptions for trading-day TODO) |
| Alpha selection | CV-optimal via ElasticNetCV |
| l1_ratio search | scikit-learn default grid |
| Random seed | 42 (for any stochastic operations) |

**Fallback models (if primary R² ~ 0):**
- Fallback A: Logistic Elastic Net on binary win/loss (TP1 reached?)
- Fallback B: Elastic Net on MFE_R
- Fallback C: Univariate analysis — each feature vs TheoreticalR

---

## 4. Data Boundaries

| Boundary | Value |
|----------|-------|
| In-sample (IS) start | 2018-01-01 |
| In-sample (IS) end | 2024-12-31 |
| Out-of-sample (OOS) start | 2025-01-01 |
| Out-of-sample (OOS) end | Present |
| Universe | 29 symbols (28 tradable + 1 research_only) |
| Active clusters | 8 (IDX_FUT, METALS_FUT, ENERGY_FUT, RATES_FUT, GRAINS_FUT, ETF_SECTOR, EQ_COMMODITY_LINKED, EQ_MACRO_BELLWETHER) |
| Reserved clusters | 3 (SOFTS_FUT, LIVESTOCK_FUT, FX_FUT — Phase 2+) |

---

## 5. B1 Signal Parameters (Frozen Defaults + Pre-Registered Ranges)

| Parameter | Default | Range | Unit |
|-----------|---------|-------|------|
| SlopeThreshold | 8 | 4–14 | % EMA rise |
| SlopeLookback | 20 | 15–30 | bars |
| MinBarsOfAir | 6 | 3–10 | bars |
| MaxBarsOfAirLookback | 50 | 20–100 | bars |
| EntryGraceBars | 3 | 1–5 | bars |
| BreakdownBuffer_ATR | 0.5 | 0.3–0.7 | ATR multiple |
| SwingLookback_Daily | 20 | 15–25 | bars |
| SwingLookback_Weekly | 12 | 8–16 | bars |
| StopMgmtMode | 1 | 1 only | static |
| GapScanWindow | 100 | 50–200 | bars |
| ATR_Period | 14 | fixed | bars |
| WilliamsR_Period | 10 | fixed | bars |
| EMA_Period | 10 | fixed | bars |

**Parameter stability requirement:** Edge must survive at EMA 9, 10, and 11
(plateau confirmation, not single-point optimum).

---

## 6. Pass / Fail Criteria (Gate 1: Phase 1a → 1b)

All 9 items must PASS:

1. OOS trades >= 30
2. Top-tercile minus bottom-tercile OOS spread >= +1.0R
3. Score-to-outcome monotonicity present (top > mid > bottom on avg R)
4. Feature cap respected (9 candidates + 1 cluster control, frozen)
5. Model card complete and reproducible (v2 template with all sections)
6. All three negative controls passed
7. Entry degradation test within tolerances (<=25% total R, <=5pp win rate, <=30% MAR)
8. Slippage stress test: profitable at 2 ticks per side
9. Calibration: quintile scores directionally calibrated OOS

**Decision:** PASS → start two-stage EV tests. FAIL → iterate per pre-registered fallback only.

---

## 7. Kill / Pause Criteria

| Condition | Action |
|-----------|--------|
| OOS top-tercile avg R < 0.5R | REJECT |
| OOS score-R correlation < 0.05 | REJECT |
| Monotonicity failure (mid > top on OOS) | PAUSE |
| Rolling score drift (20-trade corr < 0) for 20 consecutive trades | PAUSE |
| < 30 OOS trades after 12 months forward collection | REJECT |
| > 60% of IS top-tercile trades from single cluster | REJECT |
| Negative controls fail | REJECT |
| Parameter sensitivity spike-shaped for >= 2 parameters | REJECT |
| Slippage stress: edge evaporates at 2 ticks | REJECT |
| Entry degradation: total R collapses > 50% | REJECT |
| Two or more feature coefficients flip sign between fits | PAUSE |

**Actions:** REJECT requires full respecification. PAUSE requires investigation report before resuming.

---

## 8. Negative Controls

Three controls must pass before any Gate decision:

1. **Randomized labels:** Shuffle TheoreticalR labels; model should produce near-zero R² and no tercile spread.
2. **Lag-shift test:** Shift features forward by 1 bar (use future data); model should fail or degrade significantly. If it doesn't, there is data leakage.
3. **Placebo feature:** Add a random noise column; it should receive approximately zero weight from Elastic Net.

---

## 9. Minimum Trade Count Rules

- Total triggers < 80: results labeled `exploratory_only`
- Per-cluster minimum: 15+ OOS trades before cluster coefficients influence live sizing
- Symbols below threshold: remain in model as informational but cannot drive live sizing

---

## 10. Reproducibility Chain

Every model run must record and preserve the following fields:

| Field | Source | Purpose |
|-------|--------|---------|
| `dataset_sha256` | `dataset_assembler.compute_manifest()` | Immutable dataset identity |
| `code_commit_hash` | `git rev-parse HEAD` | Code version that produced the run |
| `pre_registration_version` | This document (v1) | Locked hypotheses and thresholds |
| `model_config_hash` | SHA-256 of `configs/model.yaml` | Model hyperparameters |
| `phase_config_hash` | SHA-256 of `configs/phase1a.yaml` | Phase boundaries and caps |
| `pre_reg_config_hash` | SHA-256 of `configs/pre_registration_v1.yaml` | Machine-readable pre-reg |
| `is_start` | `2018-01-01` | In-sample start boundary |
| `is_end` | `2024-12-31` | In-sample end boundary |
| `oos_start` | `2025-01-01` | Out-of-sample start boundary |
| `random_seed` | `42` | Reproducible stochastic operations |
| `run_timestamp` | ISO-8601 UTC | When the run was executed |
| `python_version` | `sys.version` | Runtime environment |
| `sklearn_version` | `sklearn.__version__` | Regression library version |

**Chain rule:** If any link in the reproducibility chain changes (dataset, code,
config, or pre-registration), a new model card must be produced. Old model cards
are never overwritten.

---

## 11. Score Threshold Definition

- **Tercile assignment:** IS trades ranked by model predicted score, split into
  three equal groups (top, mid, bottom). Remainder assigned to top group.
- **OOS application:** Score cutoffs from IS applied to OOS trades without refitting.
- **Scoring direction:** Higher score = higher predicted Theoretical R = higher conviction.

---

## 12. Risk Parameters (For Reference — Not Part of Model)

| Parameter | Phase A Value |
|-----------|---------------|
| Base risk per trade | 1.5–2.0% |
| Single-asset notional cap | $11,000 (fixed, reviewed quarterly) |
| Next SAN review | May 17, 2026 |
| Drawdown brake 1 | 10% → reduce size 50% |
| Drawdown brake 2 | 15% → reduce size 75% |
| Drawdown brake 3 | 20% → halt new trades |
| Phase B transition | Account > $2M for 30 consecutive days |

---

## Attestation

This document was produced before any model fitting. No analysis results
influenced the choice of features, thresholds, or kill criteria. All
parameters are taken from the locked specification documents:

- `docs/specs/locked/B1_Strategy_Logic_Spec_v2.md`
- `docs/specs/locked/CTL_Phase_Gate_Checklist_v2.md`
- `docs/specs/locked/CTL_Phase1a_Project_Tracker_v3.md`

**Code commit at pre-registration:** `2e0479e494a0e8775ef1c50c85d8d16c21cf8c10`
