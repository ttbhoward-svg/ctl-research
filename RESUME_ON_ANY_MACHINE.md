# Resume On Any Machine

Use this file to restart work quickly on desktop or laptop without losing context.

## 1) Sync and bootstrap

```bash
cd ~/ctl-research
# if first time on machine:
# git clone git@github.com:ttbhoward-svg/ctl-research.git

git pull origin main
python3 -m venv .venv 2>/dev/null || true
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pytest tests/unit/ -q
```

## 2) Read these files first

1. `HANDOFF.md`
2. `SESSION_LOG.md`
3. `docs/governance/phase1a_completion_snapshot.md`
4. `docs/governance/trip_laptop_runbook.md`
5. `docs/governance/project_time_log.md`

## 3) Claude restart prompt (copy/paste)

```text
Read HANDOFF.md, SESSION_LOG.md, docs/governance/phase1a_completion_snapshot.md, and docs/governance/trip_laptop_runbook.md.
Then summarize current state in 10 bullets and identify the single highest-priority next task.
Rules:
- No scope drift.
- Show changed files before commit.
- Run targeted tests and full unit suite before commit.
- Commit one task per commit with clear message.
```

## 4) Codex restart prompt (copy/paste)

```text
Read /Users/ttbhoward/Desktop/Codex Test/ctl-research/HANDOFF.md and /Users/ttbhoward/Desktop/Codex Test/ctl-research/SESSION_LOG.md, then give me a concise status snapshot and the exact next task prompt for Claude.
Also provide the exact terminal test and git commands for that task.
```

## 5) Daily operating checklist

1. Pull latest (`git pull origin main`)
2. Confirm branch/state (`git status`)
3. Run task-scoped tests
4. Run full unit suite
5. Commit/push
6. Update `SESSION_LOG.md`
7. Update `docs/governance/project_time_log.md`

## 6) Data locations

- Databento raw cutover files:
  - `data/raw/databento/cutover_v1/`
- TradeStation reference files:
  - `data/raw/tradestation/cutover_v1/`

Do not commit large raw data files unless explicitly intended.

## 7) Non-negotiables

- No threshold changes without explicit governance update.
- No data-provider-specific logic inside strategy/regression modules.
- ALERT data health status blocks downstream scoring.
- Keep one task per commit for auditability.

## 8) End-of-day closeout

```bash
git status
git add .
git commit -m "<clear task/day summary>"
git push origin main
```

Then append a new entry in `SESSION_LOG.md` and a row in `docs/governance/project_time_log.md`.
