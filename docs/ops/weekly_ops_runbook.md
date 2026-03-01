# Weekly Ops Runbook

Operational procedures for the CTL weekly B1 portfolio runner
(`scripts/run_weekly_ops.py`).

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `CTL_OPS_WEBHOOK_URL` | Webhook endpoint for notifications (Slack-compatible) | For `--notify webhook` |
| `OPS_WEBHOOK_URL` | Legacy alias (checked if `CTL_OPS_WEBHOOK_URL` is unset) | No |

Set in your shell profile or scheduler environment:

```bash
export CTL_OPS_WEBHOOK_URL="https://hooks.slack.com/services/T.../B.../..."
```

## Commands

### Dry Run (validate gate + plan, no execution)

```bash
.venv/bin/python scripts/run_weekly_ops.py --dry-run --notify stdout
```

### Full Run with Stdout Notification

```bash
.venv/bin/python scripts/run_weekly_ops.py --notify stdout
```

### Full Run with Webhook Notification

```bash
.venv/bin/python scripts/run_weekly_ops.py --notify webhook
```

Webhook URL resolved from: `--webhook-url` CLI arg > `CTL_OPS_WEBHOOK_URL` env > `OPS_WEBHOOK_URL` env.

### JSON Output (machine-readable)

```bash
.venv/bin/python scripts/run_weekly_ops.py --json --notify none
```

### Custom Profile

```bash
.venv/bin/python scripts/run_weekly_ops.py --profile configs/cutover/operating_profile_v2.yaml --dry-run
```

## Scheduler Setup

### cron (Linux / macOS)

```cron
# Every Sunday at 18:00 UTC — dry-run with webhook alert
0 18 * * 0  cd /path/to/ctl-research && .venv/bin/python scripts/run_weekly_ops.py --dry-run --notify webhook >> /var/log/ctl_ops.log 2>&1

# Every Sunday at 20:00 UTC — full run with webhook alert
0 20 * * 0  cd /path/to/ctl-research && .venv/bin/python scripts/run_weekly_ops.py --notify webhook >> /var/log/ctl_ops.log 2>&1
```

### macOS launchd

Create `~/Library/LaunchAgents/com.ctl.weekly-ops.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.ctl.weekly-ops</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/ctl-research/.venv/bin/python</string>
        <string>/path/to/ctl-research/scripts/run_weekly_ops.py</string>
        <string>--notify</string>
        <string>webhook</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>CTL_OPS_WEBHOOK_URL</key>
        <string>https://hooks.slack.com/services/T.../B.../...</string>
    </dict>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Weekday</key>
        <integer>0</integer>
        <key>Hour</key>
        <integer>20</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>/tmp/ctl_weekly_ops.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/ctl_weekly_ops.err</string>
</dict>
</plist>
```

Load with:

```bash
launchctl load ~/Library/LaunchAgents/com.ctl.weekly-ops.plist
```

## Exit Codes

| Code | Meaning | Action |
|------|---------|--------|
| 0 | Gate passed, run completed | No action needed |
| 1 | Runner error (unexpected exception) | Check logs, investigate |
| 2 | Gate mismatch (expected vs actual status diverged) | Review acceptance status, update profile if warranted |

## Troubleshooting

**Exit code 2 — gate mismatch**
The operating profile's expected statuses no longer match what the acceptance
framework derives from current data. Check which symbol(s) diverged in the
ops log JSON, then investigate the underlying data or update the locked
profile if the status change is intentional.

**Exit code 1 — runner error**
An unexpected exception occurred during B1 detection or simulation. Check
stderr/log output for the traceback. Common causes: missing data files,
corrupted CSV, or dependency version mismatch.

**Webhook not sending**
1. Verify `CTL_OPS_WEBHOOK_URL` is set: `echo $CTL_OPS_WEBHOOK_URL`
2. Test connectivity: `curl -X POST -H 'Content-Type: application/json' -d '{"text":"test"}' "$CTL_OPS_WEBHOOK_URL"`
3. Check that `--notify webhook` is passed (default is `none`).

**Retention not pruning**
Files are only pruned if older than `--retention-days` (default 45). Verify
file timestamps with `ls -la data/processed/cutover_v1/ops_logs/`.

## Notification Payloads

Webhook payloads are JSON with the following shape:

```json
{
  "text": "human-readable message",
  "level": "info | warn | alert",
  "meta": {
    "exit_code": 0,
    "timestamp": "20260301_200000"
  }
}
```

Three notification types are sent depending on outcome:
- **Success** (`level: info`): gate passed, all symbols completed cleanly.
- **Symbol failure** (`level: warn`): gate passed but one or more symbols errored.
- **Gate failure** (`level: alert`): gate mismatch, run aborted.
