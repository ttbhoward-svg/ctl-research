"""Unit tests for ops notification helpers (H.10, H.11)."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.ops_notifier import (
    CTL_OPS_WEBHOOK_ENV,
    build_gate_fail_message,
    build_ops_message,
    build_success_message,
    build_symbol_fail_message,
    dispatch_notification,
    load_webhook_url,
    notify_stdout,
    notify_webhook,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_success_result() -> dict:
    return {
        "timestamp": "20260301_120000",
        "gate_passed": True,
        "aborted": False,
        "dry_run": False,
        "exit_code": 0,
        "symbol_run_results": [
            {"symbol": "ES", "status": "EXECUTED", "detail": "ok",
             "trigger_count": 5, "trade_count": 3, "total_r": 2.5, "win_rate": 0.6},
            {"symbol": "CL", "status": "EXECUTED", "detail": "ok",
             "trigger_count": 10, "trade_count": 8, "total_r": 1.0, "win_rate": 0.5},
        ],
    }


def _make_abort_result() -> dict:
    return {
        "timestamp": "20260301_120000",
        "gate_passed": False,
        "aborted": True,
        "dry_run": False,
        "exit_code": 2,
        "symbol_run_results": [],
    }


def _make_error_result() -> dict:
    return {
        "timestamp": "20260301_120000",
        "gate_passed": True,
        "aborted": False,
        "dry_run": False,
        "exit_code": 0,
        "symbol_run_results": [
            {"symbol": "ES", "status": "EXECUTED", "detail": "ok",
             "trigger_count": 5, "trade_count": 3, "total_r": 2.5, "win_rate": 0.6},
            {"symbol": "CL", "status": "ERROR", "detail": "data corrupt"},
        ],
    }


# ---------------------------------------------------------------------------
# Tests: build_ops_message
# ---------------------------------------------------------------------------

class TestBuildOpsMessage:
    def test_success_message(self):
        msg = build_ops_message(_make_success_result())
        assert "[OK]" in msg
        assert "PASS" in msg
        assert "Executed: 2" in msg
        assert "11 trades" in msg

    def test_abort_message(self):
        msg = build_ops_message(_make_abort_result())
        assert "[ALERT]" in msg
        assert "aborted" in msg.lower()
        assert "MISMATCH" in msg

    def test_error_message_includes_failures(self):
        msg = build_ops_message(_make_error_result())
        assert "[OK]" in msg  # overall run completed
        assert "1 symbol(s) failed" in msg
        assert "CL" in msg
        assert "data corrupt" in msg

    def test_dry_run_noted(self):
        result = _make_success_result()
        result["dry_run"] = True
        msg = build_ops_message(result)
        assert "dry-run" in msg

    def test_timestamp_included(self):
        msg = build_ops_message(_make_success_result())
        assert "20260301_120000" in msg

    def test_nonzero_exit_code_warns(self):
        result = _make_success_result()
        result["exit_code"] = 1
        result["aborted"] = False
        msg = build_ops_message(result)
        assert "[WARN]" in msg
        assert "code 1" in msg


# ---------------------------------------------------------------------------
# Tests: notify_stdout
# ---------------------------------------------------------------------------

class TestNotifyStdout:
    def test_prints_message(self, capsys):
        notify_stdout("hello ops")
        captured = capsys.readouterr()
        assert "hello ops" in captured.out


# ---------------------------------------------------------------------------
# Tests: notify_webhook
# ---------------------------------------------------------------------------

class TestNotifyWebhook:
    @patch("ctl.ops_notifier.requests.post")
    def test_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        result = notify_webhook("test msg", "https://example.com/hook")
        assert result is True
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"]
        assert payload["text"] == "test msg"
        assert payload["level"] == "info"  # default level

    @patch("ctl.ops_notifier.requests.post")
    def test_http_error_returns_false(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_post.return_value = mock_resp

        result = notify_webhook("test", "https://example.com/hook")
        assert result is False

    @patch("ctl.ops_notifier.requests.post")
    def test_request_exception_returns_false(self, mock_post):
        import requests as req
        mock_post.side_effect = req.ConnectionError("refused")

        result = notify_webhook("test", "https://example.com/hook")
        assert result is False

    @patch("ctl.ops_notifier.requests.post")
    def test_timeout_parameter_passed(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        notify_webhook("msg", "https://example.com", timeout=5)
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["timeout"] == 5


# ---------------------------------------------------------------------------
# Tests: dispatch_notification
# ---------------------------------------------------------------------------

class TestDispatchNotification:
    def test_none_mode_does_nothing(self, capsys):
        dispatch_notification("none", "should not print")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_stdout_mode_prints(self, capsys):
        dispatch_notification("stdout", "hello from dispatch")
        captured = capsys.readouterr()
        assert "hello from dispatch" in captured.out

    @patch("ctl.ops_notifier.notify_webhook")
    def test_webhook_mode_calls_webhook(self, mock_wh):
        dispatch_notification("webhook", "msg", webhook_url="https://x.com")
        mock_wh.assert_called_once_with("msg", "https://x.com", level="info", meta=None)

    @patch("ctl.ops_notifier.notify_webhook")
    def test_webhook_mode_no_url_skips(self, mock_wh):
        dispatch_notification("webhook", "msg", webhook_url=None)
        mock_wh.assert_not_called()

    def test_unknown_mode_does_not_crash(self, capsys):
        dispatch_notification("pigeon", "msg")
        captured = capsys.readouterr()
        assert captured.out == ""

    @patch("ctl.ops_notifier.notify_webhook")
    def test_dispatch_forwards_level_and_meta(self, mock_wh):
        meta = {"exit_code": 2}
        dispatch_notification("webhook", "msg", webhook_url="https://x.com",
                              level="alert", meta=meta)
        mock_wh.assert_called_once_with("msg", "https://x.com",
                                        level="alert", meta=meta)


# ---------------------------------------------------------------------------
# Tests: load_webhook_url (H.11)
# ---------------------------------------------------------------------------

class TestLoadWebhookUrl:
    def test_cli_url_takes_precedence(self):
        with patch.dict(os.environ, {CTL_OPS_WEBHOOK_ENV: "https://env.example.com"}):
            result = load_webhook_url(cli_url="https://cli.example.com")
        assert result == "https://cli.example.com"

    def test_env_fallback(self):
        with patch.dict(os.environ, {CTL_OPS_WEBHOOK_ENV: "https://env.example.com"}):
            result = load_webhook_url()
        assert result == "https://env.example.com"

    def test_no_source_returns_none(self):
        with patch.dict(os.environ, {}, clear=True):
            result = load_webhook_url()
        assert result is None

    def test_empty_env_returns_none(self):
        with patch.dict(os.environ, {CTL_OPS_WEBHOOK_ENV: ""}):
            result = load_webhook_url()
        assert result is None

    def test_none_cli_falls_through_to_env(self):
        with patch.dict(os.environ, {CTL_OPS_WEBHOOK_ENV: "https://env.example.com"}):
            result = load_webhook_url(cli_url=None)
        assert result == "https://env.example.com"


# ---------------------------------------------------------------------------
# Tests: build_gate_fail_message (H.11)
# ---------------------------------------------------------------------------

class TestBuildGateFailMessage:
    def test_basic_gate_fail(self):
        result = _make_abort_result()
        result["gate_result"] = {
            "passed": False,
            "recommendation": "CONDITIONAL GO",
            "symbol_results": [
                {"symbol": "ES", "expected": "WATCH", "actual": "WATCH", "passed": True},
                {"symbol": "CL", "expected": "ACCEPT", "actual": "REJECT", "passed": False},
            ],
        }
        msg = build_gate_fail_message(result)
        assert "[ALERT]" in msg
        assert "MISMATCH" in msg
        assert "CL" in msg
        assert "expected ACCEPT" in msg
        assert "got REJECT" in msg

    def test_timestamp_included(self):
        result = _make_abort_result()
        msg = build_gate_fail_message(result)
        assert "20260301_120000" in msg

    def test_no_gate_result_key(self):
        result = {"timestamp": "20260301_120000", "gate_passed": False}
        msg = build_gate_fail_message(result)
        assert "[ALERT]" in msg


# ---------------------------------------------------------------------------
# Tests: build_symbol_fail_message (H.11)
# ---------------------------------------------------------------------------

class TestBuildSymbolFailMessage:
    def test_symbol_fail(self):
        result = _make_error_result()
        msg = build_symbol_fail_message(result)
        assert "[WARN]" in msg
        assert "1 symbol failure" in msg
        assert "CL" in msg
        assert "data corrupt" in msg

    def test_timestamp_included(self):
        result = _make_error_result()
        msg = build_symbol_fail_message(result)
        assert "20260301_120000" in msg


# ---------------------------------------------------------------------------
# Tests: build_success_message (H.11)
# ---------------------------------------------------------------------------

class TestBuildSuccessMessage:
    def test_success(self):
        msg = build_success_message(_make_success_result())
        assert "[OK]" in msg
        assert "PASS" in msg
        assert "Executed: 2" in msg
        assert "11 trades" in msg

    def test_dry_run_noted(self):
        result = _make_success_result()
        result["dry_run"] = True
        msg = build_success_message(result)
        assert "dry-run" in msg

    def test_timestamp_included(self):
        msg = build_success_message(_make_success_result())
        assert "20260301_120000" in msg


# ---------------------------------------------------------------------------
# Tests: webhook payload shape (H.11)
# ---------------------------------------------------------------------------

class TestWebhookPayloadShape:
    @patch("ctl.ops_notifier.requests.post")
    def test_default_payload_has_text_and_level(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        notify_webhook("msg", "https://example.com")
        payload = mock_post.call_args[1]["json"]
        assert "text" in payload
        assert "level" in payload
        assert payload["level"] == "info"
        assert "meta" not in payload  # omitted when None

    @patch("ctl.ops_notifier.requests.post")
    def test_payload_with_meta(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        notify_webhook("msg", "https://example.com", level="alert",
                       meta={"exit_code": 2})
        payload = mock_post.call_args[1]["json"]
        assert payload["text"] == "msg"
        assert payload["level"] == "alert"
        assert payload["meta"] == {"exit_code": 2}

    @patch("ctl.ops_notifier.requests.post")
    def test_warn_level_payload(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        notify_webhook("warn msg", "https://example.com", level="warn")
        payload = mock_post.call_args[1]["json"]
        assert payload["level"] == "warn"
