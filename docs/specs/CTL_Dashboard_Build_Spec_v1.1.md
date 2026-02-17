# CTL Dashboard Build Spec (Phased, UX-First, Model-Aware) v1.1

## Objective
Build a phased, modular dashboard that evolves with the research pipeline.
Prioritize decision quality, auditability, and operator clarity over UI complexity.

## Design Principles
1. Phase-aligned build: only expose what current phase can reliably support.
2. "No Action" default: dashboard should reduce overtrading.
3. Full provenance: every metric links to snapshot timestamp, model version, and data hash.
4. Research vs Execution separation: different views, different cognitive load.
5. Modular components: each panel independently pluggable/removable.

---

## Information Architecture

## App Modes
1. `Research Mode`
- Model quality, feature diagnostics, OOS checks, drift monitoring.

2. `Execution Mode`
- Ranked candidates, risk budget, event mode, compliance constraints.

3. `Review Mode`
- Shadow portfolio decomposition, overrides, post-trade attribution.

## Core Navigation
- Home (Cockpit)
- Signals
- Trade Drilldown
- Model Diagnostics
- Risk & Exposure
- Events
- Morning Brief
- Audit Log
- Settings

---

## Data Contract (All Phases)
Every screen header must show:
- `asof_timestamp`
- `data_snapshot_id`
- `dataset_hash`
- `model_version`
- `code_commit_hash`
- `phase_tag` (1a/1b/2/3)

If any stale/missing feed, show blocking warning badge.

---

## Daily Morning Brief Module (Required)

### Purpose
A pre-market decision memo that summarizes what matters today in 2-5 minutes.

### Schedule
- Auto-generate once each trading morning (pre-open).
- Optional refresh intraday on major event/risk-mode changes.

### Delivery
- Dashboard home panel
- Optional email/Telegram summary
- Saved as immutable record (`brief_id`, timestamp, data/model hashes)

### Required Sections
1. Executive Summary (3-6 bullets)
- Market regime state
- Risk mode (normal/reduced/freeze)
- Whether there is actionable signal: yes/no

2. Risk Budget
- Starting equity
- Current DD and brake tier
- Total heat cap and available new risk
- Max new trades allowed today

3. Top Candidate Setups
- Ranked list (max 5-10)
- symbol, setup type, timeframe, score, expected R, stop distance, suggested size
- cross-asset confirmation status
- event conflict flag

4. Event Watchlist
- Tier 1/2 events in next 24h
- pre-lock/post-stabilization windows
- affected symbols/themes

5. Theme/Exposure Snapshot
- Current theme exposures vs caps
- concentration warnings
- invalidator alerts

6. Compliance Reminders
- cooldown timers
- daily loss halt threshold
- max overrides remaining

7. Action State
- `NO ACTION` (default) or `ACTIONABLE SETUPS PRESENT`
- If actionable, include exact symbols and priority order

### Output Schema (minimum)
- `brief_id`
- `asof_ts`
- `snapshot_id`
- `dataset_hash`
- `model_version`
- `risk_mode`
- `executive_summary_json`
- `top_setups_json`
- `event_watch_json`
- `theme_exposure_json`
- `compliance_json`
- `action_state`

### Governance
- Morning Brief is informational, not a signal-generation override.
- If brief says `NO ACTION`, discretionary entries require override reason code.
- Any mismatch between brief and dashboard data must trigger a stale-data warning.

### Phase Rollout
- Phase 1a: risk budget + top setups + action state
- Phase 1b: add score diagnostics summary
- Phase 2: add regime/theme/event sections
- Phase 3+: add analogs and discretionary attribution snippets

---

## Phase-by-Phase Build

## Phase 1a Dashboard (MVP, Read-Only)
### Goal
Operate and validate baseline B1 pipeline with minimal UI complexity.

### Required Panels
1. Pipeline Health
- ingestion status
- row counts
- null/duplicate checks
- latest run pass/fail

2. Signal Queue (B1 only)
- symbol, timeframe, setup, score (single-stage), expected R proxy, stop distance, status

3. Validation Snapshot
- IS/OOS split summary
- top-vs-bottom tercile spread
- trade counts
- MAR/PF/DD quick stats

4. Trade Drilldown (minimal)
- trigger details
- confluence flags
- entry/stop/TP
- MFE/MAE/R outcome
- slippage stress/degradation results

### Exclusions
- No complex customization
- No multi-strategy overlays
- No advanced theming

## Phase 1b Dashboard (Scoring Upgrade)
### Additions
1. Model Comparison Panel
- single-stage vs two-stage EV
- OOS monotonicity comparison
- calibration/reliability plot
- Brier score

2. Score Distribution Panel
- bucketed expectancy
- hit-rate by score band
- threshold sensitivity chart

### Gate Support
- explicit pass/fail ribbon for Phase 1b -> Phase 2 gate

## Phase 2 Dashboard (Regime + Theme + Cross-Asset)
### Additions
1. Regime Panel
- active regime labels
- regime transition warning
- regime alignment per candidate

2. Cross-Asset Confirmation Matrix
- component checks (+1/0/-1)
- composite score [-1,+1]
- warning state if below threshold

3. Theme Mapping Panel
- Theme ID per trade
- theme exposure % and caps
- invalidator status
- review queue

4. Event Risk Panel
- next high-impact events
- symbol/theme impact
- current risk mode (normal/reduced/freeze)

## Phase 3+ Dashboard (Advanced Review + Custom UX)
### Additions
1. Shadow Portfolio 3-Curve View
- signal-only mechanical
- production mechanical
- actual execution

2. Override & Compliance Analytics
- override frequency
- rule violations
- discretionary alpha/drag decomposition

3. Historical Analogs / Similar Trades
- nearest-neighbor setup comparisons
- outcomes and context

4. Custom Workspace Builder
- user-selected columns
- saved watchlists
- saved thresholds
- saved panel layouts

---

## Drilldown Spec (All Phases)
Click path:
`Signal Row -> Trade Detail -> Feature Contributions -> Historical Context -> Execution/Audit`

Trade Detail must include:
- setup metadata
- all active feature values
- score breakdown
- risk sizing components
- event/regime mode at decision time
- audit trail (overrides, reason codes)

---

## Customization Spec (Progressive)
## Phase 1a-1b
- column show/hide
- sort/filter
- save one default view

## Phase 2
- multiple saved views (`Research`, `Execution`, `Risk`)
- threshold sliders (read-only if not pre-registered)

## Phase 3+
- full layout customization
- user-defined alert presets
- role-based presets (if team expands)

---

## UX Guardrails (Behavioral)
1. "No Action" banner when no candidate passes threshold.
2. Risk lockout banner if DD brake/event freeze active.
3. Hard warning if stale data or model mismatch.
4. Require reason code before manual override.
5. Show trade budget remaining (daily/weekly).

---

## Technical Stack Recommendation
## MVP (Phase 1a-2)
- Streamlit frontend
- SQLite/Postgres backend
- Plotly charts
- scheduled pipeline jobs (cron/n8n)

## Later (Phase 3+ if needed)
- React + API layer for advanced customization/performance

---

## Performance Targets
- Initial load < 3s
- filter/sort interactions < 500ms
- refresh cadence:
  - research panels: 5-15 min
  - execution panels: 1-5 min
  - event panel: 1 min

---

## QA / Validation Requirements
1. Reproducibility test:
- same snapshot + model version => identical dashboard values

2. Data integrity test:
- stale feed and missing fields raise visible warnings

3. Gate alignment test:
- dashboard pass/fail tags must match model-card gate outcomes

4. Audit test:
- every override action logged with timestamp + reason code + prior value

---

## Module Rollout Plan

Use this phased module schedule to align dashboard complexity with model maturity.

## Phase 1a
- Chart Inspector
- MTFA View

## Phase 1b
- Trade Summary Dashboard (tercile distributions, equity curve, OOS diagnostics)
- Portfolio Risk Summary (heat, DD tier, SAN usage, risk budget)

## Phase 2
- Morning Brief
- Developing Setup Tracker

## Phase 3
- Historical Analog Finder
- Pre-Trade Scenario Calculator

## Phase 4
- Seasonality Library
- Universe Health Dashboard
- COT Heat Map

## Phase 5
- Post-Trade Autopsy
- Monte Carlo Simulator

Notes:
- Portfolio Risk Summary is intentionally moved up to Phase 1b as a governance-critical control.
- Each module addition must map to a phase gate and include test coverage before promotion.

---

## Color & Logic Notes (From Prior Ideation, Reconciled)

Use the tags below when integrating prior concept text into current docs:
- `ADOPT NOW` = aligned with current spec and phase plan
- `DECISION POINT` = useful, but needs explicit rule before implementation
- `DEFER/IGNORE` = valuable later, but out of current phase scope or conflicts with governance

### 1) Morning Intelligence Brief (single HTML page)
Status: `ADOPT NOW` (already aligned with this spec)
- Keep: regime status, active triggers, developing setups, open position status, portfolio risk summary, prior-day outcomes.
- Keep UX philosophy: 5-minute read, mostly \"No Action\" days.
- `DECISION POINT`: fixed generation time (6 AM) vs market-specific pre-open schedule by asset class.

### 2) Historical Analog Finder
Status: `DEFER to Phase 3` (already in module rollout)
- Keep concept: nearest-neighbor feature similarity + visual analog outcomes.
- `DECISION POINT`: similarity metric choice (cosine vs Mahalanobis) and feature scaling policy.
- Governance rule: analogs are advisory context, never direct override of pre-registered score threshold.

### 3) Pre-Trade Scenario Calculator
Status: `ADOPT in Phase 3` (high value, risk-desk function)
- Keep: what-if heat, correlation overlap, DD impact, and suggested de-risk actions.
- `DECISION POINT`: whether recommendations are hard constraints or advisory prompts.
- Required integration: MAR 2.0 brakes + theme caps + event risk mode.

### 4) Seasonality/Pattern Library (+ COT heat map)
Status: `ADOPT in Phase 4`
- Keep: monthly seasonality, year-over-year overlays, COT heat map.
- `DECISION POINT`: whether seasonality is context-only or model feature candidate.
- Governance rule: if promoted to model input, must be separately pre-registered and OOS-validated.

### 5) Post-Trade Autopsy (auto-generated one-page)
Status: `ADOPT in Phase 5`
- Keep: predicted vs realized outcome, key deltas vs analogs, chart links.
- Keep workflow: auto-generated quantitative skeleton + manual qualitative annotation.
- Governance rule: autopsy output feeds review, not retroactive model tuning without pre-registered process.

### 6) Universe Health Dashboard
Status: `ADOPT in Phase 4`
- Keep: grid/heatmap with trend status, EMA distance, MTFA alignment.
- `DECISION POINT`: color thresholds and transition-state definitions must be pre-registered per cluster.
- Governance rule: informational only unless converted into explicit model features.

### 7) Monte Carlo Drawdown Simulator
Status: `ADOPT in Phase 5`
- Keep: drawdown distribution and probability context for psychological resilience.
- `DECISION POINT`: run cadence (weekly vs monthly) and resampling method (bootstrap vs permutation).
- Governance rule: simulator informs risk policy calibration, not discretionary signal overrides.

### Terminology/Consistency Reconciliation
- Replace references to \"20+ symbols\" with current phase-specific universe counts.
- Replace any generic \"score\" language with explicit versioned model output (`single-stage` in 1a, `EV` in 1b+ only if validated).
- Any module that implies live discretionary action must respect phase gates and override logging.

---

## Deliverables Requested from Claude
1. Detailed UI wireframe spec (phase-by-phase)
2. Backend schema additions and API contracts
3. Component map for each panel
4. Implementation plan with milestones by phase
5. Test plan (reproducibility, stale data, gate consistency, audit trail)
