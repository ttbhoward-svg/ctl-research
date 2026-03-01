"""Ops notification helpers (H.10, H.11).

Builds human-readable ops messages from run result dicts and dispatches
them to configurable backends (stdout, webhook).

The webhook payload is ``{"text": ..., "level": ..., "meta": {...}}``
JSON body, compatible with Slack incoming-webhook endpoints.

H.11 additions: ``load_webhook_url`` for secure config resolution,
typed message builders (``build_gate_fail_message``,
``build_symbol_fail_message``, ``build_success_message``).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Environment variable for webhook URL (H.11).
CTL_OPS_WEBHOOK_ENV = "CTL_OPS_WEBHOOK_URL"


# ---------------------------------------------------------------------------
# Webhook configuration (H.11)
# ---------------------------------------------------------------------------

def load_webhook_url(cli_url: Optional[str] = None) -> Optional[str]:
    """Resolve webhook URL with precedence: CLI arg > env var > None.

    Parameters
    ----------
    cli_url : str or None
        Explicitly provided URL (e.g. from ``--webhook-url``).

    Returns
    -------
    str or None
        Resolved URL, or None if no source provides one.
    """
    if cli_url:
        return cli_url
    return os.environ.get(CTL_OPS_WEBHOOK_ENV) or None


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
# Typed message builders (H.11)
# ---------------------------------------------------------------------------

def build_gate_fail_message(result: dict) -> str:
    """Build a gate-failure notification message.

    Parameters
    ----------
    result : dict
        Ops summary dict where ``gate_passed`` is False.

    Returns
    -------
    str
    """
    lines = ["[ALERT] Gate mismatch — portfolio run aborted."]
    lines.append("  Gate: MISMATCH")

    gate_result = result.get("gate_result", {})
    for sr in gate_result.get("symbol_results", []):
        if not sr.get("passed", True):
            lines.append(
                f"    - {sr['symbol']}: expected {sr.get('expected', '?')}, "
                f"got {sr.get('actual', '?')}"
            )

    ts = result.get("timestamp", "")
    if ts:
        lines.append(f"  Timestamp: {ts}")
    return "\n".join(lines)


def build_symbol_fail_message(result: dict) -> str:
    """Build a notification message for symbol-level failures.

    Parameters
    ----------
    result : dict
        Ops summary dict with errors in ``symbol_run_results``.

    Returns
    -------
    str
    """
    sym_results = result.get("symbol_run_results", [])
    errors = [s for s in sym_results if s.get("status") == "ERROR"]

    lines = [f"[WARN] Portfolio run completed with {len(errors)} symbol failure(s)."]
    lines.append("  Gate: PASS")
    for e in errors:
        lines.append(f"    - {e['symbol']}: {e.get('detail', 'unknown error')}")

    ts = result.get("timestamp", "")
    if ts:
        lines.append(f"  Timestamp: {ts}")
    return "\n".join(lines)


def build_success_message(result: dict) -> str:
    """Build a success notification message.

    Parameters
    ----------
    result : dict
        Ops summary dict for a clean run.

    Returns
    -------
    str
    """
    lines = ["[OK] Portfolio run completed successfully."]
    lines.append("  Gate: PASS")

    dry_run = result.get("dry_run", False)
    if dry_run:
        lines.append("  Mode: dry-run")

    sym_results = result.get("symbol_run_results", [])
    executed = [s for s in sym_results if s.get("status") == "EXECUTED"]
    if executed:
        total_trades = sum(s.get("trade_count", 0) for s in executed)
        total_r = sum(s.get("total_r", 0.0) for s in executed)
        lines.append(
            f"  Executed: {len(executed)} symbol(s), {total_trades} trades, R={total_r:.2f}"
        )

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
    level: str = "info",
    meta: Optional[dict] = None,
) -> bool:
    """Post ops message to a webhook endpoint.

    Sends ``{"text": message, "level": level, "meta": {...}}`` as JSON.
    Compatible with Slack incoming webhooks (extra fields ignored).

    Parameters
    ----------
    message : str
        The message body.
    webhook_url : str
        Destination URL.
    timeout : int
        Request timeout in seconds.
    level : str
        Severity level (``"info"``, ``"warn"``, ``"alert"``).
    meta : dict or None
        Optional metadata dict included in the payload.

    Returns
    -------
    bool
        True if the request succeeded (2xx), False otherwise.
    """
    payload: dict = {"text": message, "level": level}
    if meta:
        payload["meta"] = meta
    try:
        resp = requests.post(
            webhook_url,
            json=payload,
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
    level: str = "info",
    meta: Optional[dict] = None,
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
    level : str
        Severity level forwarded to webhook backend.
    meta : dict or None
        Optional metadata forwarded to webhook backend.
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
        notify_webhook(message, webhook_url, level=level, meta=meta)
        return
    logger.warning("Unknown notification mode '%s'; skipping.", mode)
