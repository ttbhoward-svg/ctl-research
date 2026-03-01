#!/usr/bin/env python3
"""Schedule-ready weekly ops wrapper (H.10).

Wraps the gate-first B1 portfolio runner with:
  - ops-log persistence
  - configurable notification dispatch (stdout, webhook)
  - retention-based cleanup of old summaries and ops logs

Designed for cron/systemd-timer invocation.

Usage
-----
    python scripts/run_weekly_ops.py --dry-run --notify stdout
    python scripts/run_weekly_ops.py --json --notify none
    python scripts/run_weekly_ops.py --notify webhook --webhook-url https://hooks.slack.com/...

Exit codes: 0 = gate pass + run ok, 2 = gate mismatch, 1 = runner error.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure src/ is importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ctl.operating_profile import load_operating_profile  # noqa: E402
from ctl.ops_notifier import (  # noqa: E402
    build_gate_fail_message,
    build_ops_message,
    build_success_message,
    build_symbol_fail_message,
    dispatch_notification,
    load_webhook_url,
)
from ctl.run_orchestrator import (  # noqa: E402
    build_run_plan,
    execute_run_plan,
    make_b1_executor,
    run_profile_gate,
    save_run_summary,
    summarize_run,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_weekly_ops")

DEFAULT_PROFILE = REPO_ROOT / "configs" / "cutover" / "operating_profile_v1.yaml"
DEFAULT_OPS_LOG_DIR = (
    REPO_ROOT / "data" / "processed" / "cutover_v1" / "ops_logs"
)
DEFAULT_SUMMARY_DIR = (
    REPO_ROOT / "data" / "processed" / "cutover_v1" / "run_summaries"
)
DEFAULT_RETENTION_DAYS = 45


# ---------------------------------------------------------------------------
# Retention cleanup
# ---------------------------------------------------------------------------

def prune_old_files(directory: Path, retention_days: int) -> list[str]:
    """Delete JSON files older than *retention_days* in *directory*.

    Parameters
    ----------
    directory : Path
        Directory to scan for ``*.json`` files.
    retention_days : int
        Files older than this many days are deleted.

    Returns
    -------
    list of str
        Names of deleted files.
    """
    if not directory.is_dir():
        return []

    cutoff = datetime.now(timezone.utc).timestamp() - (retention_days * 86400)
    deleted: list[str] = []

    for f in sorted(directory.glob("*.json")):
        if f.stat().st_mtime < cutoff:
            logger.info("Retention: deleting %s", f.name)
            f.unlink()
            deleted.append(f.name)

    return deleted


# ---------------------------------------------------------------------------
# Ops log persistence
# ---------------------------------------------------------------------------

def save_ops_log(ops_result: dict, out_dir: Path = DEFAULT_OPS_LOG_DIR) -> Path:
    """Write ops summary JSON to disk.

    Parameters
    ----------
    ops_result : dict
        The ops summary dict.
    out_dir : Path
        Output directory.

    Returns
    -------
    Path to the written JSON file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = ops_result.get("timestamp", datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    filename = f"{ts}_ops.json"
    path = out_dir / filename
    with open(path, "w") as f:
        json.dump(ops_result, f, indent=2)
    logger.info("Ops log saved to %s", path)
    return path


# ---------------------------------------------------------------------------
# Safe notification dispatch (H.11)
# ---------------------------------------------------------------------------

def _safe_dispatch(
    notify_mode: str,
    message: str,
    webhook_url: str | None,
    level: str = "info",
    meta: dict | None = None,
) -> None:
    """Dispatch notification without crashing the run on failure."""
    try:
        dispatch_notification(notify_mode, message, webhook_url,
                              level=level, meta=meta)
    except Exception:
        logger.warning("Notification dispatch failed; continuing.", exc_info=True)


# ---------------------------------------------------------------------------
# Core wrapper
# ---------------------------------------------------------------------------

def run_weekly_ops(
    profile_path: Path,
    include_non_gating: bool = False,
    dry_run: bool = False,
    notify_mode: str = "none",
    webhook_url: str | None = None,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    ops_log_dir: Path = DEFAULT_OPS_LOG_DIR,
    summary_dir: Path = DEFAULT_SUMMARY_DIR,
) -> dict:
    """Execute the full weekly ops cycle.

    Returns
    -------
    dict
        Ops result dict (JSON-serialisable).
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # --- Retention cleanup ---
    pruned_summaries = prune_old_files(summary_dir, retention_days)
    pruned_ops = prune_old_files(ops_log_dir, retention_days)

    # --- Gate check ---
    logger.info("Running operating profile gate check...")
    gate_result = run_profile_gate(profile_path)

    if not gate_result.passed:
        ops_result: dict = {
            "timestamp": timestamp,
            "gate_passed": False,
            "aborted": True,
            "dry_run": dry_run,
            "exit_code": 2,
            "gate_result": gate_result.to_dict(),
            "symbol_run_results": [],
            "retention": {
                "pruned_summaries": pruned_summaries,
                "pruned_ops": pruned_ops,
            },
        }
        save_ops_log(ops_result, ops_log_dir)
        msg = build_gate_fail_message(ops_result)
        _safe_dispatch(notify_mode, msg, webhook_url, level="alert",
                       meta={"exit_code": 2, "timestamp": timestamp})
        return ops_result

    # --- Build plan + execute ---
    logger.info("Gate passed. Building run plan...")
    profile = load_operating_profile(profile_path)
    plan = build_run_plan(
        profile,
        include_non_gating=include_non_gating,
        profile_path=str(profile_path),
    )

    executor = make_b1_executor() if not dry_run else None
    symbol_results = execute_run_plan(plan, executor=executor, dry_run=dry_run)

    # --- Summarize ---
    summary = summarize_run(
        plan, gate_result, symbol_results,
        dry_run=dry_run, timestamp=timestamp,
    )

    # --- Save run summary (skip on dry-run) ---
    summary_path = None
    if not dry_run:
        summary_path = save_run_summary(summary, out_dir=summary_dir)

    # --- Build ops result ---
    has_errors = any(r.status == "ERROR" for r in symbol_results)
    ops_result = {
        "timestamp": timestamp,
        "gate_passed": True,
        "aborted": False,
        "dry_run": dry_run,
        "exit_code": 0,
        "run_summary": summary.to_dict(),
        "summary_path": str(summary_path) if summary_path else None,
        "symbol_run_results": [r.to_dict() for r in symbol_results],
        "has_errors": has_errors,
        "retention": {
            "pruned_summaries": pruned_summaries,
            "pruned_ops": pruned_ops,
        },
    }

    save_ops_log(ops_result, ops_log_dir)

    # --- Dispatch typed notification ---
    if has_errors:
        msg = build_symbol_fail_message(ops_result)
        _safe_dispatch(notify_mode, msg, webhook_url, level="warn",
                       meta={"exit_code": 0, "has_errors": True, "timestamp": timestamp})
    else:
        msg = build_success_message(ops_result)
        _safe_dispatch(notify_mode, msg, webhook_url, level="info",
                       meta={"exit_code": 0, "timestamp": timestamp})
    return ops_result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Schedule-ready weekly ops wrapper (H.10).",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_PROFILE,
        help="Path to operating profile YAML.",
    )
    parser.add_argument(
        "--include-non-gating",
        action="store_true",
        default=False,
        help="Include non-gating symbols (e.g. PA) in the run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Gate + plan only, no strategy execution.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output ops result as JSON.",
    )
    parser.add_argument(
        "--notify",
        choices=["none", "stdout", "webhook"],
        default="none",
        help="Notification dispatch mode.",
    )
    parser.add_argument(
        "--webhook-url",
        type=str,
        default=None,
        help="Webhook URL (env: CTL_OPS_WEBHOOK_URL, legacy: OPS_WEBHOOK_URL).",
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=DEFAULT_RETENTION_DAYS,
        help="Delete ops/summary files older than N days (default 45).",
    )
    args = parser.parse_args()

    # Resolve webhook URL: CLI arg > CTL_OPS_WEBHOOK_URL > legacy OPS_WEBHOOK_URL.
    resolved_url = load_webhook_url(args.webhook_url)
    if resolved_url is None:
        resolved_url = os.environ.get("OPS_WEBHOOK_URL") or None

    ops_result = run_weekly_ops(
        profile_path=args.profile,
        include_non_gating=args.include_non_gating,
        dry_run=args.dry_run,
        notify_mode=args.notify,
        webhook_url=resolved_url,
        retention_days=args.retention_days,
    )

    if args.json_output:
        print(json.dumps(ops_result, indent=2))

    sys.exit(ops_result.get("exit_code", 1))


if __name__ == "__main__":
    main()
