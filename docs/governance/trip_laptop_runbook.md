# Trip Laptop Runbook (No TradeStation)

Window: 2026-02-19 through 2026-02-27 (return Friday)
Constraint: no local TradeStation access during travel

## Goals While Traveling
1. Keep infrastructure deterministic and auditable.
2. Strengthen reporting and review flows using existing data/artifacts.
3. Prepare a zero-friction re-entry checklist for TradeStation return.

## Daily Start (Laptop)
Run from repo root:

```bash
source .venv/bin/activate
python3 -m pytest tests/unit/ -q
```

If green, proceed with scheduled task of the day.

## Laptop-Safe Work Queue
1. Governance hardening
- tighten `docs/governance/phase1a_closeout_report.md`
- ensure cross-links between lock summary, pre-registration, and gate decision docs

2. Artifact/report utilities
- add helper scripts to render markdown summaries from JSON artifacts
- standardize output folder structure and naming conventions

3. Chart-study workflow polish
- improve templates in `src/ctl/chart_study.py`
- add concise analyst checklists for top/bottom tercile review sessions

4. Drift/report rehearsal
- run drift monitor on synthetic or prior exported data snapshots
- verify alert/status behavior and reporting format

5. Phase 2 prep (docs only, no scope jump)
- draft clear entry criteria for Phase 2 start
- freeze what must be true after first post-trip TS refresh

## Do Not Do During Trip
- no threshold changes to locked Phase 1a criteria
- no new strategy logic modules beyond approved roadmap
- no backtest interpretation from stale/non-refreshed TS exports as final evidence

## Re-entry Checklist (Desktop + TradeStation)
On return week (starting 2026-02-27):
1. Export fresh TS datasets using locked schema and naming.
2. Run ingestion + sanitization.
3. Rebuild assembled dataset and manifest hash.
4. Re-run OOS evaluation + gate decision artifacts.
5. Compare against Phase 1a snapshot for drift and consistency.

## Fast Commands
```bash
# sanity
python3 -m pytest tests/unit/ -q

# focused modules
python3 -m pytest tests/unit/test_oos_evaluation.py -v
python3 -m pytest tests/unit/test_gate_decision.py -v
python3 -m pytest tests/unit/test_drift_monitor.py -v
```

## End-of-Day Log Rule
At end of each travel workday:
1. Add one entry to `SESSION_LOG.md`
2. Add hours row in `docs/governance/project_time_log.md`
3. Commit/push docs and any tested code changes
