# CTL Research Infrastructure: Project Synthesis (v2)

## Consolidated after three rounds of review by Codex, Gemini, and Claude — Updated Feb 17, 2026
## v2 changes: Universe restructured (29 symbols), 11-cluster taxonomy, Core+Satellite governance, dashboard modules, risk infrastructure, macro trader research integration

---

## Background

I am a discretionary futures and equities trader with a $550K account (targeting $800K starting capital after initial crypto liquidations). I have six years of experience following the CTL (C The Light) methodology, a paid service run by Dean Karrys that focuses on candlestick-based setups with multi-timeframe analysis, fibonacci targets, and Williams %R divergence. My best year (2024) returned 100%+ in my IRA and 50%+ in my brokerage account. The gap between the two is entirely attributable to discipline failures in the brokerage account (scalping, revenge trading, loss averaging) that the IRA's restricted access prevented.

I cannot code. All code generation is handled by AI tools (Claude, Codex, Gemini). I use TradeStation as my charting and execution platform, with thinkorswim as a secondary reference.

---

## Where We Started (Last Week)

The original project was a **backtester specification** for TradeStation EasyLanguage. The goal was to codify three core CTL setups into mechanical rules and backtest them to derive historical performance:

- **B1: 10 EMA Retest** — backside setup where price pulls back to a rising 10 EMA after 6+ bars of air
- **B2: Empty Out** — more aggressive B1 variant where the bar opens and closes below the 10 EMA
- **F1: Inside Doji and Up** — frontside setup with a qualifying doji followed by an inside bar breakout

We built a detailed spec (CTL Codex Spec, now at v2.1) that separates:

1. **Core definitions** — deterministic trigger conditions, exact inequalities, no room for interpretation
2. **Lever stack** — execution model choices (fill model, stop management, compounding) toggled independently for experimentation
3. **Confluence module** — boolean flags logged per signal for post-hoc analysis, never required for baseline signal firing

The spec also includes environment configuration (IOG OFF, Bar Magnifier OFF, session templates, continuous contract methods) and slippage/commission defaults per instrument class to prevent unrealistic backtests.

Codex generated EasyLanguage code from the spec. After several debugging iterations, the code compiles and runs. It was validated against known historical CTL alerts.

---

## What the Backtester Revealed

The backtester produced very few entries with tight conditions. It was net positive on daily timeframes for some tickers, net negative on higher timeframes, and produced zero triggers below daily.

This result is **correct and expected**. The core CTL setups in isolation on a single timeframe are rare events. What makes them work in practice is the multi-timeframe confluence and macro context that a discretionary trader applies — the monthly trend, the quarterly structure, the daily entry refinement, the tape reading. Stripping all of that out and testing the naked trigger on one timeframe tests a shadow of the strategy.

**Key insight: the backtester's value is not as a standalone trading system. It is a scanner and signal validator that feeds a larger decision support architecture.**

---

## Where We Are Now: The Reframe

The project has evolved from "build a backtester" to **"build a one-person prop desk research infrastructure that replicates as much institutional commodity trading analytical capability as possible using AI-assisted code generation, free/cheap data sources, and a non-coder operator."**

The system's purpose is to:

1. **Quantify the marginal R-contribution of every confluence factor** across hundreds of historical triggers, so I know which factors genuinely improve outcomes and which are noise
2. **Score live setups in real time** using a weighted scorecard derived from regression analysis, so I have a pre-trade conviction level before I apply discretionary judgment
3. **Incorporate macro regime, cross-asset, flow proxy, and fundamental data** as additional scoring factors, replicating ~60-70% of the informational edge that institutional commodity desks have
4. **Force me through the analytical work** of collating, testing, and reviewing charts with performance data, which develops pattern recognition and macro intuition as a byproduct of the process
5. **Provide psychological scaffolding for scaling** — at larger account sizes ($5M+), the system's objective scoring replaces subjective conviction, preventing the emotional deterioration that destroys discretionary traders who scale without a framework

---

## Full System Architecture

### Layer 1: Signal Detection (EasyLanguage, per-strategy)

One .eld file per setup type. Each references Data1 (entry timeframe), Data2 (one timeframe up), Data3 (two timeframes up) for multi-timeframe context. All strategies output to an identical CSV schema.

**Phase 1 strategies (spec complete):**
- B1: 10 EMA Retest
- B2: Empty Out
- F1: Inside Doji and Up

**Phase 2 strategies (specs needed):**
- F2: Red Bar Ignored (requires %R filter quantification)
- F3: Frontside Momentum (inherently multi-TF: weekly up over monthly, etc.)

**Phase 3 strategies (more complex, deferred):**
- F4: A-B-C Fib Extension (requires automated swing point identification)
- Scalp/intraday setups (require tick or 1-min data)

### Layer 2: Multi-Timeframe Alignment (computed within EasyLanguage)

At the time of every trigger, the code tags the higher-TF state using Data2/Data3:

- **WeeklyTrendAligned**: Weekly Slope_20 >= threshold AND price > weekly 10 EMA
- **MonthlyTrendAligned**: Monthly Slope_20 >= threshold AND price > monthly 10 EMA
- **QuarterlyStructureBullish**: Price above quarterly 10 EMA
- **HTF_BarsOfAir**: Bars-of-air count on the next higher timeframe
- **NestedHTF_Trigger**: TRUE when the trigger timeframe has an active setup on the next higher timeframe (e.g., daily B1 fires while weekly B1 is also active — this is hypothesized to be the highest-value flag)

### Layer 3: Unified Output Schema

Every trigger from every strategy writes one row to a standardized CSV:

```
Date, Ticker, Timeframe, SetupType, Entry, Stop, TP1-TP5,
BarsOfAir, Slope_20,
[Strategy confluence flags: %R_divergence, clean_pullback, volume_declining, 
fib_confluence, gap_fill_below, multi_year_highs, single_bar_pullback],
[MTFA flags: weekly_trend, monthly_trend, quarterly_structure, 
htf_bars_of_air, nested_htf_trigger],
[Regime flags: DXY_regime, SPX_above_10EMA, NQ_above_10EMA, VIX_level, 
yield_curve_2s10s, risk_on_composite],
[Flow proxy flags: unusual_options_flow, warehouse_stock_trend, 
COT_commercial_pctile, COT_managed_money_pctile, producer_equity_relative, 
news_sentiment_7d],
[Cross-asset flags: related_commodity_confirmation, sector_equity_confirmation],
[Fundamental: seasonality_month, COT_extreme_flag],
RMultiple_Actual, MFE_R, MAE_R, TradeOutcome
```

### Layer 4: Data Warehouse (SQLite, managed by Claude Code / automation)

All output files merged, cleaned, tagged with regime and flow data from external sources. Single source of truth for all analysis.

**External data sources (free tier):**
- CFTC COT reports (via `cot_reports` Python library or direct download)
- FRED economic data (DXY, VIX, yield curves, Fed funds rate)
- CME daily options volume and open interest reports
- COMEX/LME/SHFE warehouse stock data (daily, free)
- EIA weekly petroleum/natural gas reports
- USDA crop reports
- Producer earnings transcripts (Seeking Alpha, company IR pages)
- News sentiment (NewsAPI, GDELT, parsed by Claude)
- AIS vessel tracking (limited free tier via MarineTraffic)

**External data sources (paid, $50-500/month):**
- Unusual Whales or FlowAlgo for options flow (~$50/month)
- Kpler or Vortexa for commodity vessel tracking (higher cost, evaluate later)

### Layer 5: Statistical Analysis (Python, run by Claude Code)

**Primary analysis: multi-factor regression**

Dependent variable: R-multiple of trade outcome  
Independent variables: all confluence flags, MTFA flags, regime variables, flow proxies as binary or continuous predictors  
Strategy type (B1, B2, F1, etc.) as categorical predictor

This produces a **weighted scorecard** — the marginal R-contribution of each factor. Example output: "%R divergence adds 0.4R, clean pullback adds 0.3R, weekly trend aligned adds 0.8R, COT commercial extreme adds 0.6R, single-bar pullback subtracts 0.5R."

**Secondary analyses:**
- Segment by instrument class (futures vs equities) — do the weights differ?
- Segment by timeframe — do weekly triggers have different factor importance than daily?
- Interaction terms — does %R divergence matter more when combined with clean pullback?
- Regime conditioning — do B1 triggers in risk-on regimes produce materially different R than risk-off?
- Cross-asset correlation analysis — does related commodity confirmation improve outcomes?
- COT positioning analysis — do extreme commercial positions predict better B1 outcomes?
- Seasonality analysis — month-of-year as categorical variable
- Out-of-sample validation (critical for capital reallocation decisions)

### Layer 6: Live Scanner (TradeStation RadarScreen)

RadarScreen indicator generated by Claude Code from the regression weights. Runs across full watchlist in real time.

**Columns per ticker:**
- Active trigger type (B1/B2/F1/none)
- Confluence flag count
- Weighted setup quality score
- MTFA alignment summary
- Regime score (composite risk-on/off)
- Implied R to TP1
- Alert flag if score exceeds threshold

**Morning workflow:** Scanner produces ranked list of active triggers. I review the top-scored setups on charts, apply discretionary judgment, cross-reference with Dean's alerts if any align, and execute.

### Layer 7: Automation / Orchestration

OpenClaw, n8n, or similar handles:
- File watching (TradeStation CSV dumps → auto-parse)
- Scheduled regression re-runs (weekly/monthly)
- Morning alert digest (ranked triggers via Telegram)
- Data pipeline maintenance

This layer is convenience, not critical path. The system works without it — it just requires manual file handling.

---

## Build Phases and Timeline (Revised per Codex Final Review)

### Phase 1a: B1 + MTFA only (6–8 weeks + 2-week contingency = 8–10 weeks realistic)
- Per Codex: still aggressive with non-coder + data plumbing. Build in contingency per phase.
- B1 strategy only. **29 symbols (28 tradable + 1 research_only)** per Universe Lock Spec. IDX_FUT: /ES, /NQ, /YM, /RTY. METALS_FUT: /PA, /GC, /SI, /HG, /PL. ENERGY_FUT: /CL, /NG. RATES_FUT: /ZB, /ZN. GRAINS_FUT: /ZC, /ZS. ETF_SECTOR: XLE, XME, GDX, XLF, XLI, XLK, XLY. EQ_COMMODITY_LINKED: $XOM, $FCX, $NEM, $CAT, $SBSW (research_only). EQ_MACRO_BELLWETHER: $GS, $MU. Daily + weekly.
- MTFA via last-completed weekly/monthly bar (Python date-based joins, not Data2/Data3)
- **9 feature candidates + 1 cluster control**: BarsOfAir, Slope_20, CleanPullback, %R_Divergence, WeeklyTrendAligned, MonthlyTrendAligned, COT_Commercial_Pctile_3yr, COT_Commercial_Zscore_1yr, VIX_Regime + AssetCluster (fixed effect)
- Term structure, Quad sizing, macro composite, and nested HTF state machine are all OUT of Phase 1a
- **Elastic Net regression** with `StandardScaler` pipeline on Theoretical R (per Gemini: ElasticNet over pure Lasso for correlated feature stability; StandardScaler mandatory to prevent range-based penalization)
- Track **Theoretical R** (unlimited splitting) alongside Actual R (account-constrained)
- Cross-validation: TimeSeriesSplit with **30-day purge gap** (≥ max holding period per Codex)
- **Pre-register**: score threshold rule (top tercile from IS), pass/fail criteria, parameter ranges, parameter stability (B1 must work at EMA 9 and 11, not just 10)
- **Pass/fail**: top tercile of scored trades produces ≥1.0R more than bottom tercile on OOS window of 30–50 trades minimum
- **Fallback if model finds nothing on R-multiple**: pivot to predicting win rate (binary) or MFE (continuous) before concluding confluence has no value
- **Permutation test** (per Gemini): any new signal discovery must survive 1,000-shuffle permutation test (real correlation in top 5% of shuffled distribution)
- Output: first scorecard, parameter sensitivity sweep heat maps, Plotly chart inspector for visual validation

### Phase 1b: Add two-stage EV scoring (2 weeks)
- ONLY begins after Phase 1a passes OOS validation
- Add logistic model for P(win) alongside the magnitude Elastic Net
- Combined EV = P(win) × E(R|win) becomes the scanner score
- Re-validate on OOS: does EV produce better score-to-outcome monotonicity than the single-stage Elastic Net alone?
- If not, stay with single-stage. Complexity must earn its place.

### Phase 2: Add regime layer + expand universe (4 weeks)
- Add Bridgewater Quad sizing modifier (scalars from Section 8.4)
- Add macro composite score (Section 9.1)
- Add term structure (backwardation/contango) — computed externally, not in Data2
- Add nested HTF state machine (Section 4.6)
- Expand symbol universe to Ring 2 (~19 additional symbols per Universe Lock)
- Quad designation timestamped as-of snapshots — no hindsight labeling
- Re-run regression with expanded feature set (Phase 1a 9 candidates + Phase 2 additions, on larger sample)
- Output: regime-conditioned scorecard

### Phase 3: Expand strategies + flow proxies (4 weeks)
- Add B2, F1 strategies (specs already written)
- Add warehouse stocks, options flow, producer equity RS as candidate features
- Each new feature pre-registered, must survive Elastic Net regularization on OOS
- Expand to Ring 3 (equities) and Ring 4 (currencies)
- Output: multi-strategy, multi-asset scorecard

### Phase 4: Live scanner build (2 weeks)
- Convert scorecard weights to RadarScreen indicator
- Set up dashboard and alerts
- Begin shadow portfolio tracking (mechanical vs actual)
- Pre-register score threshold for mechanical baseline

### Phase 5: Forward validation (3–6 months)
- Score every trigger in real time before trading
- Track correlation between pre-trade score and actual R-multiple
- **Minimum 30–50 out-of-sample trades** before conclusions (raised from 15 per Codex)
- Shadow portfolio delta measured at 50-trade intervals
- This is the prerequisite for capital reallocation from other buckets

### Phase 6: Ongoing refinement (continuous)
- Log live trade outcomes back into database
- Re-run regression quarterly with residual analysis
- Update scanner weights only if new features survive OOS validation
- Quarterly "Research Director" review: which factors are decaying? Which interactions are emerging?

---

## Concentric Expansion Plan

### Symbol Expansion (Core + Satellite governance, feature set frozen per ring, retrain with governance at each ring gate)

**Core + Satellite Structure (per Codex Universe Lock Spec):** All symbols classified as `tradable` (can generate live trades) or `research_only` (informs model coefficients, cannot generate live trades). Default for new thin/complex contracts: research_only until promoted via gate.

| Ring | Timeline | Symbols | Count | Purpose |
|------|----------|---------|-------|---------|
| Ring 1 (MVP) | Weeks 1–8 | IDX_FUT: /ES, /NQ, /YM, /RTY. METALS_FUT: /PA, /GC, /SI, /HG, /PL. ENERGY_FUT: /CL, /NG. RATES_FUT: /ZB, /ZN. GRAINS_FUT: /ZC, /ZS. ETF_SECTOR: XLE, XME, GDX, XLF, XLI, XLK, XLY. EQ_COMMODITY_LINKED: $XOM, $FCX, $NEM, $CAT, $SBSW*. EQ_MACRO_BELLWETHER: $GS, $MU. | 29 (28T+1R) | Validate scorecard across 8 clusters. Target: 840-1110 triggers. |
| Ring 2 | Weeks 9–14 | ENERGY_FUT: +/HO, /RB. GRAINS_FUT: +/ZW, /KE, /ZM, /ZL. LIVESTOCK_FUT: /LE, /GF, /HE. SOFTS_FUT: /KC, /CT, /CC, /SB, /OJ*, /ZR*. IDX_FUT: +/NKD*. EQ_COMMODITY_LINKED: +$BHP*. EQ_MACRO_BELLWETHER: +$NVDA*, $AMZN*. | +19 = 48 | Expand commodity coverage + new clusters. |
| Ring 3 | Weeks 15–20 | + 15-20 research-driven equities selected based on Phase 1-2 ETF/equity performance | +~18 = ~66 | Equities with pre-registered feature portability mappings |
| Ring 4 | Weeks 21–26 | FX_FUT: /6E, /6B, /6J, /6A, /6C, /6N, /6S, /6M, /DX. Commodity-currency ETF proxies: $EZA (ZAR), $EWZ (BRL). | +11 = ~77 | Currencies + commodity-currency proxies |

*\* = research_only status*

**Cluster taxonomy (11 clusters, immutable within phase):** IDX_FUT, METALS_FUT, ENERGY_FUT, RATES_FUT, GRAINS_FUT, SOFTS_FUT, LIVESTOCK_FUT, FX_FUT, ETF_SECTOR, EQ_COMMODITY_LINKED, EQ_MACRO_BELLWETHER.

**Ring governance (per Codex):** Each ring validates before the next expands. At each ring gate: retrain model on expanded dataset, generate new model card, compare OOS metrics to prior ring. If OOS metrics degrade beyond pre-registered tolerances (MAR >25%, tercile spread >20%, profit factor >15%, win rate >10pp), investigate before proceeding. Retraining uses same ElasticNet pipeline with same pre-registered features — only the dataset grows, not the model complexity. Universe expansion and feature expansion cannot occur in the same phase block.

**Feature portability (per Codex):** Features that don't apply to all asset classes require pre-registered substitute mappings BEFORE the ring expansion:
- COT → available for all futures; for equities, use institutional ownership change (13F data, quarterly) or short interest ratio
- Term structure → available for futures with multiple expirations; for equities, use sector relative strength
- These mappings must be documented in the model card for each ring

### Strategy Expansion (same database, SetupType as categorical variable)

| Ring | Strategy | Prerequisites |
|------|----------|--------------|
| Strategy Ring 1 | B1 only | MVP validation |
| Strategy Ring 2 | + F1 (Inside Doji and Up) | B1 scorecard validated |
| Strategy Ring 3 | + B2 (Empty Out) | Pooled B1+F1 model stable |
| Strategy Ring 4 | + F2 (Red Bar Ignored), F3 (Frontside Momentum) | New specs written, pooled model stable |
| Strategy Ring 5 | Nested multi-TF setups (monthly doji → weekly confirmation → daily entry) | NestedHTF_Trigger validated as significant predictor |

### Feature Expansion (only after universe is broad enough to support)

Features 11–15 added one at a time, each pre-registered, each must survive ElasticNet regularization and permutation test:
11. Warehouse stock trend (metals/energy)
12. Unusual options flow (binary)
13. Seasonality score (month-of-year)
14. Producer equity relative performance
15. News sentiment (7-day NLP)

**Research column promotion rule (per Gemini/Codex):** Phase 1a logs 19 research columns alongside the 9 scoring features. These research columns are for hypothesis generation ONLY. No research column may be promoted to a scoring feature without: (1) a pre-registered hypothesis with expected sign, (2) fresh OOS validation in a subsequent phase (not the same OOS window used for exploration), (3) the feature cap not exceeded at the time of promotion. This prevents multiple-comparisons bias from turning noise into false features.

### Full Universe at Maturity (~77 symbols)

| Cluster | Symbols | Count |
|---------|---------|-------|
| IDX_FUT | /ES, /NQ, /YM, /RTY, /NKD | 5 |
| METALS_FUT | /PA, /GC, /SI, /HG, /PL | 5 |
| ENERGY_FUT | /CL, /NG, /HO, /RB | 4 |
| RATES_FUT | /ZB, /ZN | 2 |
| GRAINS_FUT | /ZC, /ZS, /ZW, /KE, /ZM, /ZL | 6 |
| SOFTS_FUT | /KC, /CT, /CC, /SB, /OJ, /ZR | 6 |
| LIVESTOCK_FUT | /LE, /GF, /HE | 3 |
| FX_FUT | /6E, /6B, /6J, /6A, /6C, /6N, /6S, /6M, /DX | 9 |
| ETF_SECTOR | XLE, XME, GDX, XLF, XLI, XLK, XLY | 7 |
| EQ_COMMODITY_LINKED | $XOM, $FCX, $NEM, $CAT, $SBSW, $BHP, + TBD | ~12 |
| EQ_MACRO_BELLWETHER | $GS, $MU, $NVDA, $AMZN, + TBD | ~6 |
| Commodity-currency proxies | $EZA (ZAR), $EWZ (BRL) | 2 |
| **Total** | | **~67-77** |

---

## Capital Trajectory (Conditional on Validation)

### Why the Expanded Universe Changes the CAGR Math

The gross return per trade doesn't change with more instruments. What changes is **consistency and Sharpe ratio**. With 29 symbols, you get 40–70 qualifying triggers per year — a solid base. With 77 symbols across 8 uncorrelated asset clusters, you get 100–180 triggers per year, of which 40–80 pass the score threshold. Diversification smooths the equity curve, reduces drawdowns, and makes the CAGR more *reliable* year to year.

This is why institutional systematic funds run broad universes — not for higher returns, but for higher Sharpe (return per unit of volatility). Higher Sharpe means smoother compounding, which means you can size more aggressively without the drawdowns that force de-levering at the worst time.

### CAGR Targets by System Maturity

**Planning base case: 15–30% net. Treat 35%+ as upside contingent on multi-year OOS stability and discipline.** (per Codex)

| Phase | Universe | Trades/Year | Net CAGR Range | Sharpe (est.) | Max DD (est.) |
|-------|----------|-------------|----------------|---------------|---------------|
| MVP (B1, 29 symbols) | Medium | 40–70 | 15–25% | 0.5–0.8 | 15–25% |
| Expanded (B1/B2/F1, ~48 symbols) | Broad | 50–80 | 18–30% | 0.7–1.0 | 12–20% |
| Full (all strategies, ~77 symbols + fundamentals) | Full | 70–120 | 20–35% | 0.8–1.2 | 10–18% |

**Use the lower end for financial planning (when to make infusions, lifestyle decisions). Use the upper end only for understanding upside potential.**

### Compounding Table: Planning Base Case (20% Net) vs Optimistic (30% Net)

| Year | 20% Net (from $800K) | 30% Net (from $800K) | Notes |
|------|---------------------|---------------------|-------|
| 1 | $960,000 | $1,040,000 | System validating. Phase A risk. |
| 3 | $1,382,000 | $1,757,000 | + crypto infusions if OOS validates |
| 5 | $1,990,000 | $2,965,000 | Approaching Phase B transition at 20% |
| 7 | $2,867,000 | $5,015,000 | Phase B likely active at 30% |
| 10 | $4,954,000 | $11,068,000 | 20%: solid. 30%: lifestyle funded. |
| 15 | $12,317,000 | $37,065,000 | 20%: $20M target within reach at yr 17 |
| 20 | $30,637,000 | $124,000,000 | 30% for 20 years is aspirational, not planning |

Years 2–4 include conditional crypto infusions (ETH ~$250K, DOT ~$300K, wife's capital ~$400K).

**Capital infusion policy (tightened per Codex):** No infusion before ALL of:
- Minimum 30 OOS trades completed
- Survival through at least one drawdown of 8%+ (regime-cycle survival)
- OOS top-tercile spread maintained (≥1.0R above bottom tercile)
- Shadow portfolio signal curve is positive
No infusion is made based on enthusiasm, compounding projections, or in-sample results alone.

### Planning Scenario: $1M Start, 20% Net CAGR (Conservative Planning Base)

This is the financial planning anchor — use this for lifestyle decisions, infusion timing, and expectations. Upside scenarios are tracked but not planned on.

| Year | Account | Annual Net Income | Milestone |
|------|---------|-------------------|-----------|
| 1 | $1,200,000 | $200,000 | System validating, Phase A risk |
| 5 | $2,488,000 | $415,000 | Exceeds expenses 2x. Approaching Phase B |
| 7 | $3,583,000 | $597,000 | Financial independence (3x expenses) |
| 10 | $6,192,000 | $1,032,000 | Seven-figure annual income |
| 15 | $15,407,000 | $2,568,000 | Approaching $20M target |
| 17 | $22,186,000 | $3,698,000 | $20M target achieved. Lifestyle fully funded. |

With ETH ($250K year 2) and DOT ($300K year 3) infusions (subject to capital policy above): $22M by year 15.

**Primary metric remains MAR ratio, not CAGR.** Phase A allows more aggressive compounding while account size is small enough to absorb drawdowns. Phase B preserves the compounding base.

---

## Model Governance Specification (per Codex)

### Required Documentation Per Model Run

Every regression run (Phase 1a and all subsequent rings/phases) must produce a **Model Card** containing:
- Dataset snapshot ID (SHA-256 hash of the database file)
- Code commit hash (git)
- Parameter schema version (which features, which scaling, which CV config)
- Threshold rule version (how score terciles are defined)
- Random seeds for all stochastic operations
- Exact IS/OOS date boundaries
- Sample size (total and per cluster)
- Coefficients, signs, and which features were zeroed out
- IS metrics (R², MAE, tercile spread)
- OOS metrics (same, on held-out data)
- Approval status (pass/fail per pre-registered criteria)

### Phase Gate Checklist (Go/No-Go per Phase)

| Gate | Criteria | Approver |
|------|----------|----------|
| Phase 1a → 1b | OOS top tercile ≥ 1.0R above bottom, 30+ OOS trades, parameter stability confirmed, negative controls passed, kill criteria not triggered | Pre-registered criteria (no discretion) |
| Phase 1b → Phase 2 | EV scoring improves monotonicity vs single-stage model on OOS, model card generated | Pre-registered criteria |
| Ring N → Ring N+1 | OOS metrics on expanded dataset within 20% of prior ring, model card generated, feature portability mappings documented | Pre-registered criteria |
| Capital infusion gate | 30+ OOS trades, survival through 8%+ drawdown, shadow signal curve positive, OOS tercile spread maintained | Pre-registered criteria |
| Strategy expansion gate | Prior strategy validated on OOS, pooled model stable, new strategy spec written and pre-registered | Pre-registered criteria |

**No gate can be overridden by enthusiasm, time pressure, or in-sample results. Gates are binary: pass or iterate.**

### Negative Controls (per Codex — run automatically after every regression)

Three automated sanity checks that guard against leakage and overfitting:

1. **Randomized Labels:** Shuffle the TheoreticalR values randomly across all trades. Rerun ElasticNet. If the model still shows significant coefficients or non-trivial R², you have data leakage or overfitting. Expected result: near-zero R², no significant features.

2. **Lag-Shift Check:** Shift all features FORWARD by 5 bars (making them "see the future"). Rerun ElasticNet. If the model improves, you have look-ahead bias in the feature construction. Expected result: worse or equivalent performance to unshifted.

3. **Placebo Feature:** Add a column of random noise (standard normal) to the feature set. Rerun ElasticNet. If the model assigns nonzero weight to the random column, your regularization is too weak or your sample is too small. Expected result: random column zeroed out.

**All three must pass before any model is considered valid.** Results logged in the model card.

### Kill Criteria (Pre-Registered)

The following conditions trigger strategy rejection or system pause, regardless of in-sample performance:

| Condition | Action |
|-----------|--------|
| OOS top tercile avg R < 0.5R (even with spread) | Reject: absolute R too low to trade profitably after costs |
| Fewer than 30 OOS trades after 12 months forward testing | Reject: insufficient evidence to draw conclusions |
| Parameter sensitivity spike-shaped for ≥2 core parameters | Reject: likely curve-fit at specific parameter values |
| >60% of IS top-tercile trades from single cluster | Reject: not generalizable, edge is cluster-specific |
| OOS score-R correlation < 0.05 (near zero) | Reject: scoring model has no predictive power |
| Negative controls fail (randomized labels or placebo pass) | Reject: data leakage or overfitting confirmed |
| Score drift: rolling 20-trade correlation drops below 0.0 | Pause: model may need refit. Investigate before continuing. |
| Monotonicity failure: mid-tercile outperforms top tercile on OOS | Pause: score ordering is broken. Investigate feature set. |

---

## What This Is and What It Is Not

**It IS:**
- A one-person research organization that monetizes through trading (Griffin/Citadel model at home scale)
- A decision support system that scores setups before I trade them
- A research infrastructure that quantifies which factors drive returns
- A pattern recognition training tool (the analytical work develops intuition)
- A psychological framework for scaling capital without emotional deterioration
- A scanner that ensures I don't miss setups across a broad universe
- A regime-aware positioning framework (Bridgewater quadrants + macro composite)

**It is NOT:**
- An automated trading system (I press the buttons, always)
- A replacement for Dean's alerts (it's a filter and confirmation layer on top of them)
- A standalone strategy (the discretionary overlay is where the edge compounds)
- A shortcut to skip the learning process (the building IS the learning)
- A CAGR maximization tool (it's a MAR ratio maximization tool — consistency over magnitude)

---

## Lifestyle-First Prop Desk Design

### The "Attentional Capital" Principle

The scanner stays quiet 80% of the time. Only when weighted setup quality, MTFA alignment, AND macro composite all exceed threshold does it push an alert. This means most mornings the answer is "nothing to do" — freeing time for screenwriting, family, and the creative work that provides psychological balance to the trading practice.

When a "Prestige Series" trade does fire — a nested multi-TF B1 on /PA with backwardation, COT extreme, and Quad 2 regime — you sit down, review the chart, apply judgment, and deploy. The rest of the time, you're a researcher, not a screen watcher.

### The $20M Account Target

The goal is not infinite compounding. It's reaching an account size where the annual trading income funds the lifestyle with surplus, at a risk level that doesn't require daily attention. At $20M with a 30% CAGR:

| Net Income Goal | Gross Needed | Account Size at 30% |
|----------------|-------------|---------------------|
| $2M net | $3.3M gross | $11M |
| $3.5M net | $5.8M gross | $19M |
| $5M net | $8.3M gross | $27M |

$20M is the sweet spot: it generates $3.5M net annually, funds a premium lifestyle, and compounds the surplus into other wealth streams (real estate, DRIPs, ventures) without requiring you to "work for the account."

### Decision Metric Hierarchy (Fixed — per Codex)

All decisions follow this priority order. When metrics conflict, higher-ranked metric wins:
1. **MAR ratio** (return / max drawdown) — primary risk-adjusted metric
2. **OOS tercile spread** (top minus bottom tercile avg R) — scoring model predictive power
3. **OOS trade count** (statistical power and confidence)
4. **Net CAGR** (outcome measure, never overrides 1–3)

CAGR never overrides MAR. Capital decisions use conservative planning base (20% net), not upside scenarios.

### MAR Ratio as Primary Metric

Professional: MAR 1.0. Elite: MAR 2.0+. Target: **30% CAGR with 12–15% max drawdown (MAR 2.0–2.5).**

This is a more honest target than the 45% we modeled earlier. The expanded universe with regime filtering naturally trims return tails in exchange for stability. The tradeoff is worth it: at MAR 2.0, you can compound at scale without the psychological damage of 25%+ drawdowns that cause discipline failures.

### Two-Phase Dynamic Risk Framework

The risk model scales with account size. Pre-register BOTH phases and the transition trigger before trading.

**Phase A: Accumulation ($550K–$2M)**
- Base risk per trade: 1.5–2.0% of account
- SAN cap: $11,000 at $550K (2.0%), scales linearly with account
- Drawdown brakes: 10% DD = reduce 25%, 15% DD = reduce 50%, 20% DD = circuit breaker 48hr
- Score multiplier: top-tercile 1.25x, mid 1.0x, low 0.5x, below threshold 0x
- Max single-theme exposure: 30% of portfolio heat
- Transition: account crosses $2M for 30 consecutive days → permanent switch to Phase B

**Phase B: Preservation ($2M+)**
- Base risk per trade: 1.0% of account
- SAN cap: $15K at $1M-$3M, $30K at $3M-$10M, $50K at $10M+
- Drawdown brakes: 5% DD = reduce 25%, 10% DD = reduce 50% + score lockout, 15% DD = circuit breaker
- Liquidity cap: 2% of 20-day ADV (essential at $10M+)
- Cannot revert to Phase A. Permanent once triggered.

**Both phases:** 3 max per asset class, 10% max total heat, 30% max theme exposure. SAN cap cannot increase mid-drawdown. Risk Summary log on every trade.

### Extreme Conviction Tier ("Druckenmiller Sizing")

The standard framework produces 15–25% net CAGR from the steady flow of B1/B2/F1 trades at normal size. The difference between 25% and 50%+ in a specific year comes from 1–3 trades where multiple independent analytical systems converge at maximum signal strength simultaneously — the kind of setup where Druckenmiller put 20% of Quantum on the pound and PTJ tripled his fund in Q4 1987.

**Qualification criteria (ALL must be met simultaneously — pre-registered, no discretion):**
1. Fundamental conviction score = ±5 (maximum, from the fundamental model for that instrument)
2. Technical score in top DECILE (not just top tercile) of all historical triggers
3. Three or more INDEPENDENT confirmation signals aligned:
   - COT at 3-year extreme (≥90th or ≤10th percentile)
   - Weekly AND monthly trend aligned
   - Regime quadrant favorable for instrument class
   - Miner/producer equity divergence confirming (where applicable)
   - Term structure extreme (where applicable)
   - UOA confirming (where applicable)
4. Historical analog match: ≥3 comparable setups in database with avg R ≥ 3.0
5. No drawdown brake active (cannot use extreme sizing while in drawdown recovery)

**Sizing (Phase A):** Up to 5% of account risk (vs 1.5–2.0% base). Hard ceiling: 8% of account.
**Sizing (Phase B):** Up to 3% of account risk (vs 1.0% base). Hard ceiling: 5% of account.

**Expected frequency:** 1–3 qualifying trades per year across full ~77-symbol universe. Some years: zero. This is NOT a regular sizing tier — it's a pre-registered exception for genuinely extraordinary convergence.

**Why this matters for CAGR:** One extreme-conviction trade at 5% risk producing 4R = 20% account return from a single position. Two such trades plus the base flow of 30+ standard trades can produce 50–75% gross in a year where convergence events appear. In years without qualifying setups, you do 15–25% from standard flow. The system identifies these opportunities mechanically — you don't "decide" to size up based on feel. The checklist either qualifies or it doesn't.

**Safeguards:**
- Must pass ALL 5 criteria — 4 out of 5 does NOT qualify (no partial credit)
- Cannot be used more than 3 times per calendar quarter
- Each use logged with full justification in the trade journal
- If an extreme-conviction trade loses >2R, the tier is suspended for 30 days (cooling period)
- Shadow portfolio tracks extreme-conviction trades separately to measure whether this tier adds or subtracts value over time

### Shadow Portfolio: Three Equity Curves (per Codex)

Track three equity curves to separate three distinct questions:

1. **Signal-Only Mechanical (flat 1.5% risk in Phase A, 1% in Phase B):** Every qualifying trigger above score threshold, taken at base risk, no drawdown brakes. Answers: "Does the signal have edge?"
2. **Production Mechanical (full risk framework applied):** Same trades as #1, but with drawdown brakes, SAN cap, and score-based sizing. Answers: "Does the risk engine improve or hurt returns?"
3. **Actual Execution:** What you really did — including trades skipped, trades oversized, discretionary exits. Answers: "Are you adding or subtracting value through judgment?"

The delta between #1 and #2 measures risk-engine alpha. The delta between #2 and #3 measures discretionary alpha (or drag). This three-way decomposition is the accountability mechanism for the entire project and the prerequisite for capital infusions.

## Execution Platform: Interactive Brokers (IBKR)

**Decision:** IBKR is the primary execution platform. TOS remains the charting/visual review platform. TradeStation used for initial historical data export only.

**Why IBKR:**
- 3.06% interest sweep on $550K+ idle cash (~$16,800/year passive income)
- Python-native API via `ib_insync` (historical data, order submission, position tracking)
- Commission advantage at scale ($0.85/contract futures vs $1.50 TradeStation)
- Free historical OHLCV with funded account (potentially eliminates $120/month TS data feeds)
- Institutional-grade trade logging and performance reporting (Schwab/TOS futures tracking is inadequate)
- Options chains and greeks available via API (enables future options-as-vehicle trades)

**Execution workflow:**
1. Python generates Morning Brief overnight
2. Trader reviews in browser (5 minutes)
3. Visual confirmation on TOS charts (familiar interface, Dean's setups)
4. Order execution on IBKR (better fills, better rates, API-trackable)
5. Python pulls trade confirmations from IBKR API for shadow portfolio

---

## Intelligence Layer: Visual Dashboards & Research Tools

Each tool is a Python script reading from the existing database, outputting interactive HTML. All are independent modules plugged into the same data layer.

### Phase 1a–1b (built alongside core pipeline)
- **Chart Inspector** (Plotly): per-trade candlestick charts with EMAs, entry/exit markers, TP/stop levels
- **MTFA View**: synchronized monthly/weekly/daily panels for any trigger date
- **Trade Summary Dashboard**: score tercile box plots, equity curve, feature importance, scatter

### Phase 2 (weekend projects once data layer exists)
- **Morning Intelligence Brief**: single HTML page with regime status, active triggers, developing setups, open positions, portfolio risk summary, yesterday's results
- **Developing Setup Tracker**: setups approaching trigger conditions (e.g., "bars of air at 5, need 6")
- **Portfolio Risk Summary**: total heat, correlation clusters, sector concentration, drawdown status
- **Universe Health Dashboard**: grid of all symbols colored by trend status with sparklines

### Phase 3 (additional research modules)
- **Historical Analog Finder**: nearest-neighbor search by feature vector, shows 5 most similar past trades
- **Pre-Trade Scenario Calculator**: what-if portfolio impact before committing to a trade
- **Sector Flow Dashboard**: relative performance heatmap across commodity complex
- **COT Positioning Heat Map**: commercial vs speculative positioning across all futures, historical percentile coloring
- **Cross-Asset Divergence Alerts**: rolling correlation breakdown between related pairs

### Phase 4+ (advanced intelligence)
- **Seasonality/Pattern Library**: monthly patterns, year-over-year overlays, COT cycle visualization
- **Post-Trade Autopsy Generator**: auto-generated HTML report per closed trade with chart, R-multiple, analog comparison
- **Monte Carlo Drawdown Simulator**: 10,000 randomized sequences showing drawdown probability distribution
- **Walk-Forward Performance Decay Monitor**: rolling predictive power tracking with alert on degradation
- **Implied Volatility Surface Monitor**: IV rank and term structure for options-eligible symbols
- **Earnings/Event Calendar**: auto-flag equity triggers near earnings with options-implied move
- **Warehouse Stock & Physical Market Monitor**: COMEX/LME/SHFE/EIA inventory tracking with draw/build alerts
- **News Sentiment Tracker**: daily NLP scan across symbol universe via NewsAPI/GDELT

---

## Macro Theme Dashboard: Building for 2026–2028

The system is specifically tooled to monitor and trade the dominant structural themes expected over the trading horizon. Each theme produces a score that feeds into the fundamental conviction layer.

### Theme 1: AI Capex Explosion & Downstream Effects
**Bull case signals (tracked in dashboard):**
- Hyperscaler capex guidance trend (MSFT, GOOG, AMZN, META quarterly earnings — free)
- Semiconductor equipment orders (proxy: $ASML, $LRCX, $AMAT equity basket performance)
- Data center REIT performance ($EQIX, $DLR) as construction/demand proxy
- Power utility load growth (EIA data, free)
- Copper demand forecasts (IEA annual reports, free)

**Downstream commodity exposure:**
- Power demand: /NG (natural gas), uranium ($URA, $CCJ)
- Electrical infrastructure: /HG (copper)
- Cooling/industrial: aluminum ($AA as proxy)
- Construction materials: steel equities ($NUE, $STLD)

**Bear case / deceleration signals:**
- SaaS revenue growth deceleration ($IGV or individual SaaS equity basket)
- Enterprise AI adoption rate (qualitative — tracked via news sentiment)
- Semiconductor inventory builds (proxy: $SOX index relative to earnings revisions)
- AI model improvement deceleration (qualitative — news sentiment)

**Dashboard output:** Single composite score (-5 to +5) for AI theme health. Rising = favor industrial metals + energy longs, tech equity longs. Falling = shift to precious metals, treasuries, reduce industrial exposure.

### Theme 2: Commodity Supercycle (Jeff Currie Framework)
**Structural signals:**
- Mining capex as % of revenue across major producers (lagged, from quarterly earnings)
- Baker Hughes rig count (free, weekly) for energy
- New mine development pipeline (approximated by miner equity forward P/E trends)
- IEA and World Bank commodity supply/demand forecasts (free, annual/quarterly)
- Green transition demand forecasts for copper, nickel, lithium (IEA, free)

**Dashboard output:** Composite supply deficit score across commodity complex. Wide deficits = structural bull, favor buying pullbacks to EMA. Narrowing deficits = cycle maturing, tighten stops and reduce position sizes.

### Theme 3: Theme Reversal / Volatility Regime
**What to watch for:**
- AI capex growth decelerating while commodity demand expectations remain high = dangerous divergence
- Multiple themes reversing simultaneously = regime shift, trigger defensive posture
- VIX term structure inversion (front > back) = market pricing near-term risk
- Credit spreads widening (HY OAS from FRED, free) = funding stress emerging

**Dashboard output:** "Reversal risk" score. When multiple themes are showing deceleration simultaneously, the system shifts to: reduce net exposure, favor uncorrelated positions, increase cash (earning sweep interest on IBKR), and watch for high-conviction counter-trend setups.

### Theme-Level Risk Caps
No single macro theme can represent more than 30% of total portfolio heat. If AI capex longs (/HG, /NG, $CCJ, $NVDA) are all running simultaneously, the portfolio concentration is flagged: "4 positions in AI capex theme. Combined heat: 7.2%. Theme cap: 30% (OK, but monitor)." If the theme adds another position, the system warns: "Approaching theme cap. Consider which position has weakest technical score for trim."

---

## Crypto Integration (Phase 2+)

### Instruments
- BTC (daily/weekly), ETH (daily/weekly), SOL (daily/weekly)
- IBIT (BTC ETF) for options flow via UOA scanner
- Miner basket ($MARA, $RIOT, $CLSK) for divergence model
- DOT (personal portfolio position — relative value monitoring)

### Fundamental Models for Crypto
**BTC / Global Liquidity:** Sum of Fed + ECB + BOJ + PBOC balance sheets (FRED + respective central bank sites, free). When aggregate liquidity expands, BTC rallies. When it contracts, BTC falls. 20-week rolling correlation historically >0.7.

**BTC / Real Yields:** Similar inverse relationship to gold but noisier. Use same DFII10 TIPS yield from gold model. BTC tends to rally when real yields fall, especially below zero.

**BTC / DXY Inverse:** Strong inverse correlation. Dollar weakness = crypto strength. DXY already in macro dashboard.

**IBIT Options Flow:** Same UOA scanner as equities. Large block call buying on IBIT = institutional BTC accumulation. Large put buying = institutional hedging or bearish positioning. Integrates with Unusual Whales API ($30-50/month, already planned).

**Crypto Miner Divergence:** Same logic as PGM miner model. If $MARA/$RIOT/$CLSK outperform BTC spot by >1 std dev over 20 days, equity market pricing in improved hash economics → bullish BTC.

**DOT Relative Value Monitor:** Track DOT/BTC and DOT/ETH ratios. Flag structural breakdowns (ratio making new lows) vs recovery (ratio turning). Provides quantified framework for the DOT liquidation decision — "DOT/ETH ratio has broken below 2-year support and is making new lows. Relative weakness confirmed. Consider accelerating DOT-to-trading-capital conversion."

### Technical Application
B1, F1, and weekly/monthly doji detectors work on crypto daily and weekly charts — same code, same pipeline, same scoring. Crypto gets its own AssetCluster for the regression. Crypto-specific regime overlay (halving cycle, ETF flow accumulation trend) added as conviction modifiers.

---

## Future Strategy Expansion (All use same data layer and pipeline)

Each strategy is a Python detector class with a standard interface. Adding a strategy means writing a new detector — the simulator, scoring, and visualization are reused.

| Strategy | Type | Timeframe | Status |
|----------|------|-----------|--------|
| B1 (10 EMA Retest) | Trend continuation | Daily/Weekly | Phase 1a (active) |
| B2 (Empty Out) | Trend continuation | Daily/Weekly | Phase 3 |
| F1 (Inside Doji & Up) | Reversal/continuation | Daily/Weekly/Monthly | Phase 3 |
| F2 (Red Bar Ignored) | Trend continuation | Daily/Weekly | Phase 4 |
| F3 (Frontside Momentum) | Breakout | Daily/Weekly | Phase 4 |
| Intraday Mean Reversion | Mean reversion | 1min/5min (ES/NQ) | Phase 5+ |
| Options-as-Vehicle | Directional (defined risk) | 40–60 DTE | Phase 4 |
| Weekly/Monthly Doji | HTF reversal | Weekly/Monthly | Phase 3 (high priority) |

### Intraday Mean Reversion (Placeholder — Phase 5+)
Trigger: TICK < -500, %R ticking up from oversold, breadth > 60%, low-TF doji with confirmation on ES/NQ 5-min chart. Exit: mean reversion target or time-based (close by EOD). Options variant: 2 DTE SPX calls/spreads at exhaustion point.

Requires: intraday data feed (IBKR API), separate detector class, separate feature set (TICK, breadth, intraday VIX), own Phase 1a validation cycle. Same pipeline, different research project.

### Options-as-Vehicle (Placeholder — Phase 4)
For equity and index B1/F1 triggers: MAR 2.0 module calculates equivalent options position (45 DTE, appropriate strike per delta rules). Morning Brief presents side-by-side: "Buy 150 shares at $350 (risk $1,500) OR buy 6x April 345C at $12.50 (risk $1,250, defined)." Requires: IBKR options chain API, greeks calculation, strike/DTE selection rules.

### Weekly/Monthly Doji Strategy (Placeholder — Phase 3, high priority)
Weekly and monthly dojis at key levels (10 EMA, fib retracements, prior structure) with MTFA alignment offer extraordinary R/R. The data layer built in Phase 1a contains all the weekly and monthly OHLCV needed to investigate this. Detector class: ~50 lines of Python. Same pipeline, same scoring, same validation.

---

## Fundamental Conviction Layer: "The Why Before the When"

### Core Philosophy

The architecture inverts the traditional retail flow: instead of scanning for technical patterns and hoping they work, the fundamental layer identifies WHERE structural mispricings or supply/demand imbalances exist, then the technical system provides WHEN to enter. Technical setups are timing mechanisms for trade ideas that already have fundamental justification.

This mirrors the approach of macro commodity funds (Glencore, Trafigura) and discretionary macro traders (Druckenmiller, PTJ, Bacon). The fundamental model provides conviction and magnitude estimates. The technical setup provides timing and risk-defined entry/exit. The MAR 2.0 module sizes according to combined conviction.

### Logic Flow

```
Fundamental Model → Directional Conviction Score (-5 to +5)
    ↓
Instruments with |conviction| >= 3 go on ACTIVE WATCHLIST
    ↓
Technical Detector scans WATCHLIST symbols for entry setups
    ↓
Combined Score = Technical Score + Fundamental Conviction Modifier
    ↓
MAR 2.0 sizes position based on Combined Score + Regime
```

Trades where both fundamental conviction and technical score are high receive maximum position size. Trades with strong technicals but neutral fundamentals receive standard size. Trades where technicals conflict with fundamentals are flagged as caution.

### Multifactor Precursor Discovery Workflow

The most powerful use of this architecture: data-driven hypothesis generation. For each commodity, the workflow is:

1. **Identify large moves:** Find every instance of X%+ gain (or loss) in Y trading days over 10+ years
2. **Look back:** For each instance, compute a panel of candidate fundamental variables over the prior 30–90 days
3. **Find common precursors:** Which factors were present in the majority of large moves?
4. **Build composite conviction model:** Weight the factors that appeared most frequently
5. **Pre-register and validate:** Same OOS rigor as technical models. PLUS: run a permutation test (per Gemini) — shuffle the fundamental variable randomly 1,000 times, recompute correlation each time. The real correlation must be in the top 5% of the shuffled distribution to be considered statistically significant. This guards against data mining bias in the discovery process.
6. **Deploy:** Conviction score feeds into the watchlist and scoring system

This process is conducted conversationally with AI agents (Claude, Gemini, Codex) as brainstorming partners. The Python system is the testing engine. The trader is the Managing Director who decides which findings are worth pursuing.

### Fundamental Models by Commodity (Tiered by Confidence and Data Cost)

#### TIER 1: Highest Confidence, Free Data — Build Phase 2

**1. Palladium Supply Chain Model**
Variables: PGM miner equity basket ($SBSW, $IMPUY, $ANGPY) vs /PA relative performance (12-day lead), USD/ZAR rate of change (gradual drift neutral, crisis-speed depreciation bullish /PA), Eskom load-shedding cumulative days (30/60/90-day windows, scraped free from Eskom), PA/PL substitution ratio (extremes >1.5x or <0.7x signal mean reversion), COMEX palladium warehouse stocks and PALL ETF holdings (drawing = bullish), auto equity basket (Toyota, VW, GM, Ford, Stellantis) as demand proxy.
Causal chain: 75% of supply from SA + Russia → disruption to either = price spike. 80% of demand from ICE catalytic converters → auto production drives demand. Miner equities price in production fundamentals 2-3 weeks ahead of commodity.
Data: all free (Yahoo Finance, Eskom website, COMEX).

**2. Gold / Real Yields Inverse Model**
Variables: 10-year TIPS yield (FRED: DFII10), rate of change of real yields (20-day and 60-day), level relative to zero (negative real yields = structurally bullish gold).
Causal chain: Gold is a zero-yield asset. Its opportunity cost is the real return on safe bonds. When real yields fall (especially below zero), holding gold costs nothing → demand increases.
Data: free (FRED).

**3. Crude Oil Term Structure Model**
Variables: front-month / 6th-month spread (backwardation = tight supply = bullish, contango = excess supply = bearish), degree of backwardation/contango as percentile over 5-year history, rate of change of spread (steepening backwardation = accelerating tightness).
Causal chain: Physical oil market tightness shows up in the futures curve before it shows up in spot price. Deep sustained backwardation has preceded major bull runs historically.
Data: free (CME settlements via IBKR or CME delayed data).

**4. COT Extreme Positioning (already in scoring system)**
Variables: commercial net position 3-year percentile and 1-year z-score (already built as Phase 1a features).
Enhancement: combine with speculative positioning (large spec net long at extreme = crowded trade = bearish contrarian signal). Net spec long > 90th percentile historically precedes pullbacks.
Data: free (CFTC COT reports).

**5. Gold/Silver Ratio Extremes**
Variables: /GC ÷ /SI ratio, percentile over 5 and 10-year windows, rate of change.
Causal chain: silver has both precious and industrial demand. At ratio >80, silver is historically cheap (industrial demand undervalued). At ratio <50, gold is historically cheap (safe-haven demand undervalued). Mean-reverts over 6-18 month cycles.
Data: free (existing /GC and /SI price feeds).

**6. Copper/Gold Ratio as Economic Proxy**
Variables: /HG ÷ /GC ratio, ISM Manufacturing PMI (FRED), 10-year Treasury yield.
Causal chain: copper demand is industrial, gold demand is defensive. Rising ratio = accelerating economy = favor industrial metals and equities. Falling ratio = decelerating economy = favor gold and bonds.
Data: free (existing price feeds + FRED).

#### TIER 2: High Confidence, Free Data — Build Phase 3

**7. Hog/Corn Production Cost Model (COTGUY)**
Variables: lean hog price / corn price ratio, percentile over 5 and 10-year windows.
Causal chain: corn is 60-70% of hog production costs. When hogs are expensive relative to corn, producers expand herds (cheap feed), increasing future supply, depressing future hog prices. The relationship has been documented since the 1970s.
Data: free (existing /HE and /ZC price feeds).

**8. EIA Crude/NG Inventory Model**
Variables: weekly crude inventory change (EIA, free, Wednesday 10:30 AM), deviation from consensus estimate (investing.com), rolling 4-week trend (draws accelerating or decelerating?). Natural gas: weekly storage injection/withdrawal vs 5-year seasonal average (EIA, free, Thursday 10:30 AM).
Causal chain: inventory draws = supply tighter than demand = bullish. Builds = surplus = bearish. Deviation from consensus drives short-term price reaction.
Data: free (EIA.gov).

**9. Soybean Crush Spread**
Variables: soybean price vs (soybean meal + soybean oil) combined value, crush margin as percentile.
Causal chain: wide crush margin → processors increase throughput → more soybean demand (bullish beans) + more meal/oil supply (bearish meal/oil). Narrow margin → processors slow down → less demand.
Data: free (existing futures prices for /ZS, /ZM, /ZL or derived).

**10. Cattle on Feed / Placements Model**
Variables: USDA monthly cattle on feed report (placements, marketings, total on feed), placements as % of year-ago levels, rolling 3-month trend.
Causal chain: high placements today = more cattle coming to market in 4-6 months = bearish live cattle (/LE) forward. Low placements = supply tightening = bullish forward.
Data: free (USDA NASS).

**11. Platinum-Specific Models**
Variables: European diesel vehicle share in new car sales (ACEA, free, monthly), hydrogen economy proxy (fuel cell equities basket: PLUG, BE, FCEL), China consumer confidence (as jewelry demand proxy), PA/PL substitution ratio (same as palladium model, viewed from platinum's perspective).
Data: free (ACEA, Yahoo Finance for equities).

#### TIER 3: Medium Confidence, Low-Cost Data — Phase 4+

**12. UOA (Unusual Options Activity) as Conviction Multiplier**
Source: Unusual Whales API ($30-50/month).
Scanner logic: filter for symbol universe + commodity ETFs (GLD, SLV, COPX, XLE, USO). Flag: large block trades (>$500K premium), sweep orders, unusual volume (>3x avg on specific strike/DTE).
Special scanner — OT Block Put Sale Setup: large block PUT SALES (sold to open, not bought to open) 5-21 days before earnings. Logic: institutional player selling downside risk = high-conviction bullish signal. Scanner triggers when premium > $500K, DTE aligns with earnings window.
Output: UOA events tagged with direction, size, alignment with fundamental model. Shows in Morning Brief as conviction modifier, not standalone signal.

**13. LLM-Scored News Sentiment**
Source: NewsAPI ($50/month) or free RSS feeds from Reuters/Bloomberg.
Process: Pull headlines filtered by universe keywords. Score each via Claude API: "Rate this headline's likely impact on [commodity] prices from -5 to +5. One-sentence rationale." Aggregate into rolling 3-day sentiment score per symbol.
Application: "Palladium news sentiment: +2.8. Key driver: EU relaxation of 2035 EV mandate." Shows in Morning Brief as context, not signal.
Cost: NewsAPI $50/month + Claude API ~$30-40/month.

**14. Brazil Coffee Weather Monitor**
Source: NOAA/GFS weather models (free).
Logic: Monitor overnight minimum temperatures in Minas Gerais and São Paulo coffee regions during Southern Hemisphere winter (May-August). Frost events cause 30-50% price spikes in /KC.
Application: Binary alert: "Frost risk elevated in Minas Gerais. /KC on high alert." Seasonal, operates only May-August.

**15. Warehouse Stock & Physical Market Monitor**
Source: COMEX, LME (basic data free; detailed data from Quandl/Nasdaq Data Link ~$50-100/month).
Variables: daily warehouse stock levels for metals (gold, silver, copper, palladium, platinum), weekly draw/build rate, comparison to 5-year average.
Application: "COMEX palladium stocks at 3-year lows. Drawing 2% per week. Physical market tight."

### "Is the Market Pricing This In?" Framework

For each fundamental signal, quantify the gap between the signal strength and the market's response:

```
Fundamental_Signal_Strength = abs(conviction_score)  # 0-5 scale
Market_Response = abs(price_move_since_signal / ATR_20)  # normalized move
Pricing_Gap = Fundamental_Signal_Strength - (Market_Response / 2)
```

Positive Pricing_Gap: fundamental signal stronger than market response → opportunity may exist.
Negative Pricing_Gap: market has already moved more than fundamentals justify → late or overcorrected.

For event-driven signals (WASDE, COT release, EIA report): compare pre-event private/estimated data to futures-implied value. Divergence = potential trade around the event.

### Data Budget Summary

| Tier | Phase | Monthly Cost | Models Enabled |
|------|-------|-------------|---------------|
| Free | 1a-2 | $0 | Models 1-6 (gold/yields, crude term structure, PGM supply chain, COT, Au/Ag ratio, Cu/Au ratio) |
| Free | 3 | $0 | Models 7-11 (hog/corn, EIA, crush spread, cattle, platinum) |
| Low-cost | 4+ | $80-200 | Models 12-15 (UOA, news sentiment, coffee weather, warehouse stocks) |

**Total at full implementation: under $200/month. Expected EV: if fundamental models add even 0.3R per qualifying trade across 40-60 trades/year = 12-18R additional annual expectancy. At $5,000-$15,000 per R-unit (depending on account size), that's $60K-$270K of additional returns from a $200/month data budget.**

### CAGR Impact of Fundamental Layer

The fundamental conviction layer does not increase the number of trades. It increases the QUALITY distribution of trades taken:
- Trades with high fundamental + high technical alignment: sized up (regime multiplier), historically produce 2.5-4.0R average
- Trades with strong technical but neutral fundamental: standard size, produce 1.0-2.0R average
- Trades where fundamental conflicts with technical: flagged as caution, reduced size or passed

This asymmetric quality improvement shifts the CAGR range:
- Conservative (technical-only): 20% net
- Base case (technical + fundamental alignment): 25% net
- Optimistic (10-15 high-conviction fundamental+technical trades/year at 3-5R each): 35% net

The key insight: same number of trades, better average R, better CAGR. The fundamentals don't generate trades — they identify which technical triggers deserve maximum conviction.

---

## Consolidated Reviewer Feedback (Codex + Gemini + Claude)
- Direction is strong, architecture is sound
- Primary risk is overfitting, not technical feasibility
- MVP scope discipline is mandatory (B1 first, 29 symbols, 9 features + 1 control)
- Pre-registration of hypotheses and pass/fail criteria before running analysis
- Data2/Data3 look-ahead bias is a real gotcha requiring [1] indexing
- Regime should modify sizing, not filter entries
- Out-of-sample validation required before any capital reallocation

### Key Additions from Reviews:
- **Gemini:** Two-stage EV scoring (logistic + magnitude), term structure/backwardation, state machine for nested triggers, macro as sizing modifier, Data2/Data3 look-ahead fix, data validation layer, MAR 2.0 sizing module with SAN cap and drawdown decelerator, attentional capital filter, correlation heatmap before regression (drop >90% correlated pairs), asyncio for IBKR, modular build (Data/Signal/Dashboard modules), StandardScaler mandatory for feature normalization, Elastic Net over pure Lasso for correlated feature stability, permutation test for new signal discovery (1000-shuffle, top 5%)
- **Codex:** Shadow portfolio tracking, parameter sensitivity sweep, MAR ratio as primary metric, pre-registration protocol, purged/embargoed CV, feature cap enforcement (9 candidates + 1 cluster control), OOS threshold 30–50 trades, Phase 1a/1b split, score threshold pre-registration, THREE shadow curves (signal-only/production/actual), cluster fixed effects in regression, per-cluster minimum trade counts, kill criteria pre-registration, model card per regression run, immutable dataset versioning with SHA-256 hash, unit tests for edge cases, 30-day purge gap, stop exit at next-bar open, exchange session calendar for weekly bars, COT as two features (1yr z-score + 3yr percentile), full reproducibility chain, Model Governance Spec with phase gate checklist, negative controls (randomized labels + lag-shift + placebo), score drift tolerance, monotonicity failure detection, capital infusion policy tightening (30+ OOS trades + regime survival), CAGR anchor to 15-30% net planning base case, 6-8 week realistic timeline, ring-based retraining with governance, feature portability mappings for cross-asset expansion
- **Claude:** Bridgewater quads (not Hedgeye), macro composite score, cross-asset correlation signals, COT integration, flow proxy architecture, concentric expansion plan, Griffin research-org framing, Python-first architecture decision, IBKR execution platform, intelligence layer roadmap, future strategy expansion framework, fundamental conviction layer (15 models across 3 tiers), macro theme dashboard (AI capex, commodity supercycle, reversal risk), crypto integration, two-phase dynamic risk framework (Accumulation/Preservation), multifactor precursor discovery workflow, "is the market pricing this in?" quantification framework, UOA block put sale scanner (OT setup)

---

## Architecture Decision: Python-First

After analysis, the project uses Python for all signal detection, trade simulation, data integration, and regression. TradeStation is used only for: (1) historical OHLCV data export for continuous futures contracts, and (2) live order execution.

**Rationale:** Scales from 29 to 77 symbols in a loop. Eliminates Data2/Data3 look-ahead risk entirely (Python uses explicit date-based joins). One language for everything. Avoids EasyLanguage→Python migration at Phase 2. Claude Code dramatically better at Python.

**Risk mitigation:** Python simulator cross-validated against TradeStation on 2+ symbols before results are trusted. Must match: same trigger dates, entry prices within 1 tick, R-multiples within 0.1R.

**Supporting documents:** B1 Strategy Logic Specification (language-agnostic pseudocode resolving all edge cases) serves as the "source of truth" for the Python implementation.

---

## Red Team Notes (Pre-Build)

### The regression might find nothing
If Elastic Net zeros out most features and no confluence factors meaningfully improve B1 outcomes, that is a successful research result. It means: the base setup carries the edge, confluence is storytelling, focus shifts to execution discipline and regime timing. Do not add complexity to explain away a null result.

### Sample size may be thinner than expected
B1 on 29 symbols over 7 years should produce 840–1110 triggers. With 9 features + 1 cluster control (8 clusters), the minimum reliable sample is ~100. Pre-registered rule: if triggers < 80, results are "exploratory only" — investigate whether parameters are too restrictive before expanding further.

### Drawdown decelerator and forward validation tension
During forward validation, the MAR 2.0 module may throttle trade size so severely during drawdowns that it takes 18 months to accumulate 30–50 OOS trades. Fix: the shadow portfolio mechanical baseline runs at full 1% sizing on paper regardless of drawdown brakes. This measures pure signal quality while the decelerator protects real capital.

### SAN cap and drawdown thresholds are pre-registered
$11,000 SAN cap (fixed, reviewed quarterly — next: May 17, 2026). Phase A DD brakes: 10%/15%/20%. Phase B DD brakes: 5%/10%/15%. Documented and date-stamped before any analysis or trading. Only changeable at quarterly review.

---

## Macro Regime Layer

### Quadrant Framework (Bridgewater)

Ray Dalio's economic quadrant model classifies the macro environment by the direction of change in two variables: economic growth and inflation. This produces four regimes:

- **Quad 1 — Growth Rising, Inflation Falling:** Goldilocks. Risk assets rally, equities lead. Commodity pullbacks get bought aggressively. B1 setups in this regime have maximum macro tailwind.
- **Quad 2 — Growth Rising, Inflation Rising:** Reflation/overheating. Commodities outperform, especially industrials and energy. This is the highest-conviction regime for our commodity-heavy book. B1 triggers in Quad 2 should carry maximum size.
- **Quad 3 — Growth Falling, Inflation Rising:** Stagflation. Hardest regime to trade. Commodity signals become unreliable because demand is weakening even as prices are elevated. Defensive posture, reduced sizing, tighter stops.
- **Quad 4 — Growth Falling, Inflation Falling:** Deflation. Bonds win, commodities get crushed. B1 triggers on commodities in Quad 4 are low-conviction by definition — the macro wind is directly against you. Focus shifts to bond futures (/ZB, /ZN) and defensive equities.

**Implementation:** Approximate the quadrant from public data:
- Growth direction: Atlanta Fed GDPNow (free, weekly) or ISM PMI trend
- Inflation direction: Cleveland Fed Inflation Nowcast (free) or CPI/PCE month-over-month trend
- Update monthly. The Quad designation becomes a single categorical variable in the regression.

### Macro Composite Score

Collapse multiple macro indicators into a single score (-5 to +5) for commodity-long favorability:

| Indicator | Source | +1 Condition | -1 Condition |
|-----------|--------|-------------|-------------|
| DXY 3-month rate of change | FRED | Weakening | Strengthening |
| Real yields (10Y TIPS) | FRED | Falling | Rising |
| Global liquidity (Fed balance sheet proxy) | Fed H.4.1 | Expanding | Contracting |
| Credit conditions (HY spreads) | FRED ICE BofA | Tightening | Widening |
| China credit impulse (TSF) | PBOC/FRED | Accelerating | Decelerating |

A score of +4/+5 = "full macro tailwind for commodity longs." A score of -3 or worse = "macro headwind, defensive posture." Combined with the Quad designation, these two variables capture the full macro picture in just two regression slots.

### Regime-Aware Scanner Behavior

The scanner at maturity displays the current Quad and macro composite alongside every trigger score. The morning digest reads something like:

> "Current regime: Quad 2, Macro Composite +3. Active triggers: /PA weekly B1 (score 8.7, nested monthly F1 parent), /HG daily B1 (score 6.2), $FCX weekly B2 (score 5.9). Recommended posture: aggressive on high-scored commodity triggers."

In Quad 4 with a macro composite of -3, the same scanner might show active triggers but the digest reads:

> "Current regime: Quad 4, Macro Composite -3. Active triggers: /GC daily B1 (score 7.1), /ZB weekly F1 (score 6.8). Recommended posture: defensive. Consider /ZB over /GC given regime."

---

## Organizing Principle

Ken Griffin has said that Citadel is not a trading organization but a research organization — they employ meteorologists, doctors, crop scientists, and domain specialists across every field, and trading is simply how they monetize their research output.

This project adopts that principle at home scale. The daily practice is research: maintaining the data pipelines, updating regime classifications, studying chart patterns against performance data, deepening understanding of cross-asset relationships. Trading is the occasional output when the research indicates a high-conviction opportunity. The system enforces this posture by making "nothing to do" the default state and requiring a high score threshold before deployment.

---

## Questions for Sense-Checking

*Updated after initial review by Codex and Gemini (see Appendix A for their feedback).*

1. Is the regression approach (R-multiple vs binary/continuous factor predictors across a pooled universe of triggers) the right statistical framework, or is there a better approach for this kind of multi-factor setup quality scoring? **Gemini suggests Elastic Net over pure Lasso for correlated feature stability. Codex says regularized regression first, tree models as robustness check. Current plan: Elastic Net (L1+L2) regression with StandardScaler as primary, random forest as sanity check.**

2. What are the biggest risks of overfitting when integrating this many predictor variables? What guard rails beyond out-of-sample validation should we implement? **Codex recommends: pre-registration, purged/embargoed CV for time series, feature cap (9 candidates + 1 control in Phase 1a), walk-forward only, no tuning on OOS. Current plan adopts all of these.**

3. Is the MTFA implementation via TradeStation's Data2/Data3 multi-data approach sound, or are there edge cases in bar alignment across timeframes that could produce misleading results? **Gemini flags: referencing `Close of Data2` on a mid-week daily bar can return the developing weekly close (look-ahead). Fix: always use `[1] of Data2/Data3` to reference last COMPLETED higher-TF bar.**

4. For the flow proxy data (COT, warehouse stocks, options flow) — what's the most robust way to time-align this data with trigger dates, given that some sources are daily, some weekly, and some have reporting lags? **Gemini recommends: as-of timestamps with publication lag rules. COT = regime filter updated Saturday AM. Options flow = 24hr sentiment score reset at daily open. Warehouse stocks = 30-day MA of levels.**

5. Are there additional data sources or analytical approaches commonly used in institutional commodity research that we're missing? **Gemini adds: term structure (contango vs backwardation) — now included as one of the 10 MVP features. Codex adds: parameter stability maps and capacity/slippage stress testing.**

6. Is the Phase 1 timeline realistic for a non-coder using AI code generation tools? **Both reviewers say 2 weeks is too aggressive. Revised to 4 weeks for Phase 1 MVP. Gemini notes Data2/Data3 debugging alone takes 3–4 days.**

7. What's the minimum sample size needed per factor in the regression before the coefficients are trustworthy enough to inform live sizing decisions? **Rule of thumb: 10–20 observations per predictor variable. With 9 features + 1 cluster control, need 100–200 triggers minimum. The concentric expansion plan is designed to accumulate this sample size before conclusions are drawn.**

8. Can the regression framework surface NEW relationships we haven't hypothesized? **Yes — interaction terms and residual analysis can reveal unexpected factor combinations. But this must be treated as hypothesis generation, not confirmation. Any "discovered" relationship must be pre-registered and validated on a fresh out-of-sample window before being incorporated into the live scorecard.**

9. How do we prevent the Bridgewater quadrant classification from becoming stale or lagging at regime transitions? **Update monthly using leading indicators (GDPNow, Cleveland Fed Nowcast) rather than lagging (actual GDP, backward-looking CPI). Accept that transitions will be identified with a 1–2 month lag. The macro composite score (-5 to +5) provides faster-moving context that can signal transitions before the quad officially changes.**
