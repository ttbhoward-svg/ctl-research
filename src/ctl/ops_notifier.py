"""Ops notification helpers (H.10).

Builds human-readable ops messages from run result dicts and dispatches
them to configurable backends (stdout, webhook).

The webhook payload is a simple ``{"text": ...}`` JSON body, compatible
with Slack incoming-webhook endpoints.
"""

from __future__ import annotations

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------

def build_ops_message(result: dict) -> str:
    """Build a human-readable ops message from a run result dict.

    Parameters
    ----------
    result : dict
        Ops summary dict (as produced by ``run_weekly_ops``).

    Returns
    -------
    str
    """
    lines: list[str] = []

    gate_passed = result.get("gate_passed", False)
    aborted = result.get("aborted", False)
    dry_run = result.get("dry_run", False)
    exit_code = result.get("exit_code", -1)

    if aborted:
        lines.append("[ALERT] Gate mismatch — portfolio run aborted.")
    elif exit_code != 0:
        lines.append(f"[WARN] Portfolio run exited with code {exit_code}.")
    else:
        lines.append("[OK] Portfolio run completed successfully.")

    lines.append(f"  Gate: {'PASS' if gate_passed else 'MISMATCH'}")

    if dry_run:
        lines.append("  Mode: dry-run")

    # Symbol results.
    sym_results = result.get("symbol_run_results", [])
    errors = [s for s in sym_results if s.get("status") == "ERROR"]
    if errors:
        lines.append(f"  Errors: {len(errors)} symbol(s) failed")
        for e in errors:
            lines.append(f"    - {e['symbol']}: {e.get('detail', '')}")

    executed = [s for s in sym_results if s.get("status") == "EXECUTED"]
    if executed:
        total_trades = sum(s.get("trade_count", 0) for s in executed)
        total_r = sum(s.get("total_r", 0.0) for s in executed)
        lines.append(f"  Executed: {len(executed)} symbol(s), {total_trades} trades, R={total_r:.2f}")

    ts = result.get("timestamp", "")
    if ts:
        lines.append(f"  Timestamp: {ts}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dispatch backends
# ---------------------------------------------------------------------------

def notify_stdout(message: str) -> None:
    """Print ops message to stdout."""
    print(message)


def notify_webhook(
    message: str,
    webhook_url: str,
    timeout: int = 10,
) -> bool:
    """Post ops message to a webhook endpoint.

    Sends ``{"text": message}`` as JSON.  Compatible with Slack
    incoming webhooks.

    Parameters
    ----------
    message : str
        The message body.
    webhook_url : str
        Destination URL.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    bool
        True if the request succeeded (2xx), False otherwise.
    """
    try:
        resp = requests.post(
            webhook_url,
            json={"text": message},
            timeout=timeout,
        )
        if resp.ok:
            logger.info("Webhook notification sent (%s).", resp.status_code)
            return True
        logger.warning(
            "Webhook returned %s: %s", resp.status_code, resp.text[:200],
        )
        return False
    except requests.RequestException as exc:
        logger.warning("Webhook request failed: %s", exc)
        return False


def dispatch_notification(
    mode: str,
    message: str,
    webhook_url: Optional[str] = None,
) -> None:
    """Route a notification to the selected backend.

    Parameters
    ----------
    mode : str
        One of ``"none"``, ``"stdout"``, ``"webhook"``.
    message : str
        The ops message.
    webhook_url : str or None
        Required when *mode* is ``"webhook"``.
    """
    if mode == "none":
        return
    if mode == "stdout":
        notify_stdout(message)
        return
    if mode == "webhook":
        if not webhook_url:
            logger.warning("Webhook URL not provided; skipping notification.")
            return
        notify_webhook(message, webhook_url)
        return
    logger.warning("Unknown notification mode '%s'; skipping.", mode)
