# Cutover v1 Closeout

**Cycle:** `cutover_v1`
**Locked date:** 2026-03-01
**Portfolio recommendation:** CONDITIONAL GO

## Final Scope

### Gating Universe

| Symbol | Expected Status | Tick Size | Max Day Delta |
|--------|----------------|-----------|---------------|
| ES | WATCH | 0.25 | 3 |
| CL | ACCEPT | 0.01 | 3 |
| PL | WATCH | 0.10 | 2 |

### Non-Gating Symbols

| Symbol | Role |
|--------|------|
| PA | Research-enabled, non-gating (Databento extract contained PA, not PL) |
| AAPL | Equity reference, non-gating, basis-alignment workstream |
| XLE | Equity reference, non-gating, basis-alignment workstream |

### Policy Constraints

- **Thresholds locked:** Yes
- **Strategy logic locked:** Yes

## Current Status Board

```
ES .... WATCH           (drift floor observed; accepted for conditional operation)
CL .... ACCEPT          (L2/L3/L4 diagnostics pass all thresholds)
PL .... WATCH           (within tolerance; monitoring for promotion)
PA .... NON-GATING      (research-only; not in gating universe)
AAPL .. NON-GATING      (equity scope; not in gating universe)
XLE ... NON-GATING      (equity scope; not in gating universe)

Portfolio: CONDITIONAL GO
```

## Key Decisions Timeline

| Task | Summary |
|------|---------|
| H.2 | Accept H.3 recommended CL variant; maintain strict NO-GO with WATCH bridge status |
| H.3 | Accept CL variant for bridge diagnostics with strict NO-GO and policy WATCH |
| H.4 | Databento system-of-record; normalization metadata; WATCH for canonical progression |
| H.5 | Canonical acceptance via L2/L3/L4 explainability; portfolio NO-GO with mixed statuses |
| H.6 | CL upgraded to ACCEPT; PL blocked (wrong commodity); ES WATCH; portfolio CONDITIONAL GO |
| H.7 | Operating profile locked in YAML; gate-check regression script |
| H.8 | Gate-first B1 portfolio runner with dry-run, JSON output, summary persistence |
| H.9 | Real B1 strategy execution wired into runner (detection + simulation) |
| H.10 | Schedule-ready ops wrapper with notification dispatch and retention cleanup |
| H.11 | Production webhook config, typed notifications, safe dispatch, runbook |
| H.12 | Cutover closeout, scheduler dry-run proof, file verification |

## Infrastructure Inventory

### Config (1 file)

| File | Lines |
|------|-------|
| `configs/cutover/operating_profile_v1.yaml` | 53 |

### Source Modules (3 files, ops-layer)

| File | Lines | Purpose |
|------|-------|---------|
| `src/ctl/operating_profile.py` | 274 | Profile loader, status comparison, gate logic |
| `src/ctl/run_orchestrator.py` | 497 | Run plan, B1 execution adapter, summary persistence |
| `src/ctl/ops_notifier.py` | 292 | Message builders, webhook dispatch, notification routing |

### Scripts (3 files)

| File | Purpose |
|------|---------|
| `scripts/check_operating_profile.py` | CLI gate check (exit 0 = pass, exit 2 = mismatch) |
| `scripts/run_weekly_b1_portfolio.py` | Gate-first B1 portfolio runner |
| `scripts/run_weekly_ops.py` | Schedule-ready ops wrapper with notifications + retention |

### Tests (4 files, ops-layer)

| File | Test Count |
|------|------------|
| `tests/unit/test_check_operating_profile.py` | 27 |
| `tests/unit/test_run_weekly_b1_portfolio.py` | 36 |
| `tests/unit/test_ops_notifier.py` | 33 |
| `tests/unit/test_run_weekly_ops.py` | 20 |
| **Total** | **116** |

### Documentation (3 files)

| File | Purpose |
|------|---------|
| `docs/governance/cutover_h2_h3_decision_log.md` | Decision log (H.2–H.12) |
| `docs/ops/weekly_ops_runbook.md` | Operational runbook with cron/launchd examples |
| `docs/governance/cutover_v1_closeout.md` | This document |

## Operational Commands

### Gate Check

```bash
.venv/bin/python scripts/check_operating_profile.py
.venv/bin/python scripts/check_operating_profile.py --json
```

### B1 Portfolio Runner (Direct)

```bash
.venv/bin/python scripts/run_weekly_b1_portfolio.py --dry-run --json
.venv/bin/python scripts/run_weekly_b1_portfolio.py --json
```

### Weekly Ops Wrapper (Scheduler-Ready)

```bash
# Dry-run with stdout notification
.venv/bin/python scripts/run_weekly_ops.py --dry-run --notify stdout

# Full run with webhook notification
CTL_OPS_WEBHOOK_URL="https://hooks.slack.com/..." \
  .venv/bin/python scripts/run_weekly_ops.py --notify webhook

# JSON output for machine parsing
.venv/bin/python scripts/run_weekly_ops.py --dry-run --json --notify none
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Gate passed, run completed |
| 1 | Unexpected runner error |
| 2 | Gate mismatch, run aborted |

## Scheduler Dry-Run Proof

Executed `scripts/run_weekly_ops.py --dry-run --notify stdout --json` on 2026-03-01.

**Key fields from proof artifact:**

| Field | Value |
|-------|-------|
| `timestamp` | `20260301_224932` |
| `gate_passed` | `true` |
| `aborted` | `false` |
| `dry_run` | `true` |
| `exit_code` | `0` |
| `has_errors` | `false` |
| `portfolio_recommendation` | `CONDITIONAL GO` |
| ES gate | expected WATCH, actual WATCH, passed |
| CL gate | expected ACCEPT, actual ACCEPT, passed |
| PL gate | expected WATCH, actual WATCH, passed |
| ES run | DRY_RUN |
| CL run | DRY_RUN |
| PL run | DRY_RUN |

Notification output: `[OK] Portfolio run completed successfully. Gate: PASS. Mode: dry-run.`

Artifact path (not committed): `data/processed/cutover_v1/ops_logs/20260301_224932_ops.json`

## Known Limitations

1. **ES WATCH status.** ES exhibits a persistent drift floor in L2 diagnostics. It passes the WATCH threshold but has not been promoted to ACCEPT. No further ES variant tuning is planned for this cycle.

2. **PL WATCH status.** PL is within tolerance but has not achieved full ACCEPT. Underlying Databento extract originally contained PA (palladium) instead of PL (platinum); PA was reclassified as non-gating research.

3. **No live execution validation.** B1 strategy execution has been tested against historical data only. The dry-run proof confirms gate and plan logic; live order routing is out of scope for cutover v1.

4. **Webhook endpoint not deployed.** The notification infrastructure is wired and tested with mocks, but no production Slack webhook has been configured yet.

5. **Retention is time-based only.** Old ops logs and summaries are pruned by age (default 45 days). No count-based or size-based retention policy exists.

6. **Single-timeframe B1 only.** The runner executes daily-timeframe B1 detection. Weekly/monthly HTF confluence is not yet wired.

## Risk Statements

- **Data drift risk.** If Databento upstream data changes schema or roll conventions, the acceptance framework will flag the change as a gate mismatch (exit code 2). This is by design.
- **Strategy parameter risk.** All thresholds and strategy logic are locked. Any tuning requires a new cycle with a new operating profile.
- **Notification failure risk.** As of H.11, notification dispatch failures are caught and logged but do not crash the run. Webhook delivery is best-effort.

## Go-Forward Plan (Next Cycle)

1. **ES promotion target.** Investigate drift floor root cause; evaluate whether an adjusted tick tolerance or variant selection can promote ES from WATCH to ACCEPT.
2. **PL promotion target.** With corrected PL (not PA) data, re-run full L2/L3/L4 diagnostics and target ACCEPT.
3. **Deploy webhook.** Configure production Slack incoming webhook; validate end-to-end notification flow.
4. **Install scheduler.** Set up cron or launchd job per `docs/ops/weekly_ops_runbook.md`.
5. **SimConfig calibration.** Calibrate per-instrument slippage values for more realistic R calculations.
6. **MTFA confluence.** Add weekly/monthly HTF data loading for multi-timeframe alignment flags.
7. **Equity workstream.** Advance AAPL/XLE basis-alignment research; evaluate for future gating inclusion.
