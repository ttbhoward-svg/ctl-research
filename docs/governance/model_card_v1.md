# Phase 1a Model Card (v1)

**Model type:** Elastic Net (ElasticNetCV)
**Strategy:** B1 (10-EMA Retest)
**Date produced:** _To be filled on first run_
**Pre-registration:** `docs/governance/pre_registration_v1.md` (v1, locked 2026-02-17)

---

## 1. Reproducibility Chain

| Field | Value |
|-------|-------|
| Dataset SHA-256 | _To be filled: `dataset_assembler.compute_manifest()["sha256"]`_ |
| Code commit hash | _To be filled: `git rev-parse HEAD` at run time_ |
| Pre-registration version | v1 |
| Pre-registration config hash | _To be filled: SHA-256 of `configs/pre_registration_v1.yaml`_ |
| Model config hash | _To be filled: SHA-256 of `configs/model.yaml`_ |
| IS start | 2018-01-01 |
| IS end | 2024-12-31 |
| OOS start | 2025-01-01 |
| Random seed | 42 |
| Python version | _To be filled: `sys.version`_ |
| scikit-learn version | _To be filled: `sklearn.__version__`_ |
| Run timestamp | _To be filled: ISO-8601 UTC_ |

---

## 2. Feature Set (Frozen — 9 + 1 Control)

| # | Feature | Type | Expected Sign | Coefficient | Actual Sign | Match |
|---|---------|------|---------------|-------------|-------------|-------|
| 1 | BarsOfAir | continuous | positive | _TBD_ | _TBD_ | _TBD_ |
| 2 | Slope_20 | continuous | positive | _TBD_ | _TBD_ | _TBD_ |
| 3 | CleanPullback | boolean | positive | _TBD_ | _TBD_ | _TBD_ |
| 4 | WR_Divergence | boolean | positive | _TBD_ | _TBD_ | _TBD_ |
| 5 | WeeklyTrendAligned | boolean | positive | _TBD_ | _TBD_ | _TBD_ |
| 6 | MonthlyTrendAligned | boolean | positive | _TBD_ | _TBD_ | _TBD_ |
| 7 | COT_20D_Delta | continuous | positive | _TBD_ | _TBD_ | _TBD_ |
| 8 | COT_ZScore_1Y | continuous | positive | _TBD_ | _TBD_ | _TBD_ |
| 9 | VIX_Regime | boolean | positive | _TBD_ | _TBD_ | _TBD_ |
| C | AssetCluster | categorical | N/A | _(cluster dummies)_ | N/A | N/A |

**Feature cap:** 9 candidates + 1 cluster control. No additions permitted.

---

## 3. Training Configuration

| Parameter | Value |
|-----------|-------|
| Dependent variable | TheoreticalR |
| Standardization | StandardScaler (on 9 candidates only; cluster dummies unscaled) |
| Alpha (regularization) | _TBD: selected by ElasticNetCV_ |
| l1_ratio | _TBD: selected from [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]_ |
| Cross-validation | PurgedTimeSeriesSplit, 5 folds, 30-day purge gap |
| NaN handling | COT/MTFA NaN filled with 0.0 (neutral after scaling; see Task10_assumptions.md) |
| Cluster encoding | One-hot with drop_first=True |

---

## 4. Sample Size

| Metric | Value |
|--------|-------|
| Total IS trades | _TBD_ |
| Per-cluster counts | _TBD_ |
| Clusters with 15+ trades | _TBD_ / 8 active |
| Exploratory-only flag | _TBD: True if < 80 total_ |

---

## 5. IS Performance Metrics

| Metric | Value |
|--------|-------|
| R² (IS) | _TBD_ |
| Intercept | _TBD_ |
| Features zeroed by regularization | _TBD_ / 9 |
| Sign matches | _TBD_ / 9 |
| Sign mismatches | _TBD_ |
| Multicollinearity flags (|r| > 0.70) | _TBD_ |

---

## 6. Anti-Leakage Statement

This model was trained exclusively on in-sample data (2018-01-01 to 2024-12-31).
No out-of-sample data (2025-01-01 onward) was used for training, feature selection,
hyperparameter tuning, or threshold calibration.

Safeguards:
- PurgedTimeSeriesSplit with 30-day gap prevents within-IS temporal leakage
- Features were pre-registered before any model fitting (pre_registration_v1.md)
- Feature set is frozen (9 + 1 control) — no data-driven feature selection
- All expected coefficient signs were pre-registered before fitting
- OOS scoring (Task 13) will use the IS-fit scaler and model without refitting
- Tercile cutoffs from IS will be applied to OOS without recalibration

---

## 7. Known Limitations

1. **Purge gap uses calendar days** (30), not trading days (~21). This under-purges
   by ~30%. A trading-day purge is documented as a TODO (Task 4 assumptions).
2. **COT NaN fill** — non-futures get neutral imputation rather than structural
   handling. Cluster fixed effects compensate, but the COT coefficient reflects
   futures-only variation.
3. **No interaction terms** — the model is additive. Feature interactions (e.g.,
   BarsOfAir * WeeklyTrendAligned) are not tested in Phase 1a.
4. **Linear model** — assumes linear relationship between features and TheoreticalR.
   Non-linear effects are not captured.
5. **IS R²** is expected to be low — financial data has high noise. A low R² does
   not necessarily mean the model is useless if tercile separation is meaningful.

---

## 8. Gate 1 Readiness Checklist

| Gate 1 Item | Status |
|-------------|--------|
| 1. OOS trades >= 30 | _Pending OOS scoring (Task 13)_ |
| 2. Tercile spread >= 1.0R | _Pending OOS scoring_ |
| 3. Score monotonicity | _Pending OOS scoring_ |
| 4. Feature cap respected | PASS (9 + 1, frozen) |
| 5. Model card complete | This document |
| 6. Negative controls | _Pending (Task 11/12)_ |
| 7. Entry degradation | _Previously tested (Task 4b)_ |
| 8. Slippage stress | _Previously tested (Task 4b)_ |
| 9. Quintile calibration | _Pending OOS scoring_ |
