# Gate 1 Decision Format — Phase 1a to Phase 1b

## Purpose

This document defines the format and interpretation rules for Gate 1
decision artifacts produced by `src/ctl/gate_decision.py` (Task 14).

## Decision Verdicts

| Verdict | Meaning | Action |
|---------|---------|--------|
| **PASS** | All 9 criteria met, no kill/pause triggered | Proceed to Phase 1b (two-stage EV tests) |
| **ITERATE** | One or more criteria failed, no kill/pause | Fix per remediation, re-run, do NOT proceed |
| **PAUSE** | Monotonicity failure (mid > top on OOS) | Investigation report required before resuming |
| **REJECT** | Kill criterion triggered (top R < 0.5 or corr < 0.05) | Full respecification required |
| **INCOMPLETE** | One or more inputs not yet computed | Complete upstream tasks, re-evaluate |

Severity hierarchy: REJECT > PAUSE > ITERATE > INCOMPLETE > PASS.

## Gate 1 Criteria (9 Items)

| # | Criterion | Source | Threshold |
|---|-----------|--------|-----------|
| 1 | OOS trade count | OOS evaluation (Task 13) | >= 30 |
| 2 | OOS tercile spread | OOS evaluation (Task 13) | >= 1.0R |
| 3 | Score monotonicity | OOS evaluation (Task 13) | top > mid > bottom avg R |
| 4 | Feature cap respected | Manual attestation | 9 + 1 frozen |
| 5 | Model card complete | Manual attestation | v2 template |
| 6 | Negative controls | Negative controls (Task 11b) | All 3 passed |
| 7 | Entry degradation | Entry degradation (Task 11c) | <= 25% R, <= 5pp WR, <= 30% MAR |
| 8 | Slippage stress | Slippage stress (Task 4b) | Profitable at 2 ticks |
| 9 | Quintile calibration | OOS evaluation (Task 13) | Monotonically improving |

## Kill Criteria

| ID | Condition | Action |
|----|-----------|--------|
| K.1 | OOS top-tercile avg R < 0.5R | REJECT |
| K.2 | OOS score-R correlation < 0.05 | REJECT |
| K.3 | Monotonicity failure (mid > top) | PAUSE |

## Artifact Formats

### JSON (`gate1_decision.json`)

```json
{
  "verdict": "PASS",
  "timestamp": "2026-02-18T...",
  "dataset_hash": "sha256_...",
  "config_hash": "sha256_...",
  "code_commit_hash": "abc123",
  "n_passed": 9,
  "n_failed": 0,
  "n_incomplete": 0,
  "criteria": [
    {
      "item": 1,
      "name": "oos_trade_count",
      "status": "PASS",
      "value": "50",
      "reason": "OOS trades = 50 >= 30",
      "remediation": ""
    }
  ],
  "kill_check": {
    "top_tercile_kill": false,
    "correlation_kill": false,
    "monotonicity_pause": false,
    "details": []
  }
}
```

### Markdown (`gate1_decision.md`)

Human-readable document with:
- Verdict banner
- Provenance (dataset hash, config hash, commit hash, timestamp)
- Criteria table (9 rows)
- Remediation actions (for failed items only)
- Kill/pause flags (if triggered)

## Interpretation Rules

1. **Gates are binary.** No overrides by discretion or enthusiasm.
2. **ITERATE means fix and re-run.** Use remediation actions.
3. **PAUSE requires investigation report** documenting root cause before
   any further work.
4. **REJECT requires full respecification** — the current model/feature
   set is invalidated.
5. **INCOMPLETE is not a failure** — it means upstream work is pending.
   Complete the tasks and re-evaluate.
