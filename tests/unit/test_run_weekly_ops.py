"""Unit tests for weekly ops wrapper (H.10)."""

import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.operating_profile import (
    OperatingProfile,
    PolicyConstraints,
    PortfolioCheckResult,
    SymbolCheckResult,
    SymbolSetting,
)
from ctl.run_orchestrator import SymbolRunResult


# Import the ops wrapper functions directly (not the CLI entrypoint).
# We need to handle the sys.path for the script as well.
SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from run_weekly_ops import (  # noqa: E402
    prune_old_files,
    run_weekly_ops,
    save_ops_log,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gate_pass() -> PortfolioCheckResult:
    return PortfolioCheckResult(
        passed=True,
        recommendation="CONDITIONAL GO",
        symbol_results=[
            SymbolCheckResult("ES", "WATCH", "WATCH", True),
            SymbolCheckResult("CL", "ACCEPT", "ACCEPT", True),
            SymbolCheckResult("PL", "WATCH", "WATCH", True),
        ],
    )


def _make_gate_fail() -> PortfolioCheckResult:
    return PortfolioCheckResult(
        passed=False,
        recommendation="CONDITIONAL GO",
        symbol_results=[
            SymbolCheckResult("ES", "WATCH", "WATCH", True),
            SymbolCheckResult("CL", "ACCEPT", "REJECT", False, "expected ACCEPT, got REJECT"),
            SymbolCheckResult("PL", "WATCH", "WATCH", True),
        ],
    )


def _write_profile_yaml(tmp_path: Path) -> Path:
    data = {
        "cycle_id": "test_v1",
        "locked_date": "2026-03-01",
        "portfolio_recommendation": "CONDITIONAL GO",
        "portfolio_scope": "futures_only",
        "gating_universe": ["ES", "CL", "PL"],
        "non_gating_symbols": ["PA"],
        "symbol_settings": {
            "ES": {"tick_size": 0.25, "max_day_delta": 3, "expected_status": "WATCH"},
            "CL": {"tick_size": 0.01, "max_day_delta": 3, "expected_status": "ACCEPT"},
            "PL": {"tick_size": 0.10, "max_day_delta": 2, "expected_status": "WATCH"},
        },
        "policy_constraints": {
            "thresholds_locked": True,
            "strategy_logic_locked": True,
        },
    }
    path = tmp_path / "profile.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


def _mock_b1_executor(sym):
    return SymbolRunResult(
        sym, "EXECUTED", f"{sym} done",
        trigger_count=5, trade_count=3, total_r=1.5, win_rate=0.67,
    )


# ---------------------------------------------------------------------------
# Tests: prune_old_files
# ---------------------------------------------------------------------------

class TestPruneOldFiles:
    def test_deletes_old_files(self, tmp_path):
        old_file = tmp_path / "old.json"
        old_file.write_text("{}")
        # Backdate mtime to 60 days ago.
        old_mtime = time.time() - (60 * 86400)
        import os
        os.utime(old_file, (old_mtime, old_mtime))

        new_file = tmp_path / "new.json"
        new_file.write_text("{}")

        deleted = prune_old_files(tmp_path, retention_days=45)
        assert "old.json" in deleted
        assert not old_file.exists()
        assert new_file.exists()

    def test_keeps_recent_files(self, tmp_path):
        recent = tmp_path / "recent.json"
        recent.write_text("{}")
        deleted = prune_old_files(tmp_path, retention_days=45)
        assert deleted == []
        assert recent.exists()

    def test_missing_directory_returns_empty(self, tmp_path):
        deleted = prune_old_files(tmp_path / "nonexistent", retention_days=45)
        assert deleted == []

    def test_ignores_non_json_files(self, tmp_path):
        txt_file = tmp_path / "old.txt"
        txt_file.write_text("data")
        old_mtime = time.time() - (60 * 86400)
        import os
        os.utime(txt_file, (old_mtime, old_mtime))

        deleted = prune_old_files(tmp_path, retention_days=45)
        assert deleted == []
        assert txt_file.exists()


# ---------------------------------------------------------------------------
# Tests: save_ops_log
# ---------------------------------------------------------------------------

class TestSaveOpsLog:
    def test_saves_json(self, tmp_path):
        result = {"timestamp": "20260301_120000", "gate_passed": True}
        path = save_ops_log(result, out_dir=tmp_path)
        assert path.exists()
        assert path.name == "20260301_120000_ops.json"
        with open(path) as f:
            data = json.load(f)
        assert data["gate_passed"] is True

    def test_creates_directory(self, tmp_path):
        nested = tmp_path / "a" / "b"
        result = {"timestamp": "20260301_120000"}
        path = save_ops_log(result, out_dir=nested)
        assert nested.is_dir()
        assert path.exists()


# ---------------------------------------------------------------------------
# Tests: run_weekly_ops — success path
# ---------------------------------------------------------------------------

class TestRunWeeklyOpsSuccess:
    @patch("run_weekly_ops.run_profile_gate")
    @patch("run_weekly_ops.execute_run_plan")
    def test_success_path(self, mock_exec, mock_gate, tmp_path):
        mock_gate.return_value = _make_gate_pass()
        mock_exec.return_value = [
            SymbolRunResult("ES", "EXECUTED", "ok", 5, 3, 1.5, 0.67),
            SymbolRunResult("CL", "EXECUTED", "ok", 10, 8, 2.0, 0.5),
            SymbolRunResult("PL", "EXECUTED", "ok", 3, 2, 0.5, 0.5),
        ]

        profile_path = _write_profile_yaml(tmp_path)
        ops_dir = tmp_path / "ops"
        summary_dir = tmp_path / "summaries"

        result = run_weekly_ops(
            profile_path=profile_path,
            dry_run=False,
            notify_mode="none",
            ops_log_dir=ops_dir,
            summary_dir=summary_dir,
        )

        assert result["gate_passed"] is True
        assert result["aborted"] is False
        assert result["exit_code"] == 0
        assert len(result["symbol_run_results"]) == 3
        assert result["has_errors"] is False
        # Ops log was saved.
        assert list(ops_dir.glob("*.json"))

    @patch("run_weekly_ops.run_profile_gate")
    @patch("run_weekly_ops.execute_run_plan")
    def test_dry_run_path(self, mock_exec, mock_gate, tmp_path):
        mock_gate.return_value = _make_gate_pass()
        mock_exec.return_value = [
            SymbolRunResult("ES", "DRY_RUN", "skipped"),
            SymbolRunResult("CL", "DRY_RUN", "skipped"),
            SymbolRunResult("PL", "DRY_RUN", "skipped"),
        ]

        profile_path = _write_profile_yaml(tmp_path)
        ops_dir = tmp_path / "ops"

        result = run_weekly_ops(
            profile_path=profile_path,
            dry_run=True,
            notify_mode="none",
            ops_log_dir=ops_dir,
            summary_dir=tmp_path / "summaries",
        )

        assert result["dry_run"] is True
        assert result["exit_code"] == 0
        assert all(s["status"] == "DRY_RUN" for s in result["symbol_run_results"])


# ---------------------------------------------------------------------------
# Tests: run_weekly_ops — gate mismatch path
# ---------------------------------------------------------------------------

class TestRunWeeklyOpsGateFail:
    @patch("run_weekly_ops.run_profile_gate")
    def test_gate_mismatch_aborts(self, mock_gate, tmp_path):
        mock_gate.return_value = _make_gate_fail()

        profile_path = _write_profile_yaml(tmp_path)
        ops_dir = tmp_path / "ops"

        result = run_weekly_ops(
            profile_path=profile_path,
            notify_mode="none",
            ops_log_dir=ops_dir,
            summary_dir=tmp_path / "summaries",
        )

        assert result["gate_passed"] is False
        assert result["aborted"] is True
        assert result["exit_code"] == 2
        assert result["symbol_run_results"] == []

    @patch("run_weekly_ops.run_profile_gate")
    def test_gate_mismatch_saves_ops_log(self, mock_gate, tmp_path):
        mock_gate.return_value = _make_gate_fail()

        profile_path = _write_profile_yaml(tmp_path)
        ops_dir = tmp_path / "ops"

        run_weekly_ops(
            profile_path=profile_path,
            notify_mode="none",
            ops_log_dir=ops_dir,
            summary_dir=tmp_path / "summaries",
        )

        assert list(ops_dir.glob("*_ops.json"))


# ---------------------------------------------------------------------------
# Tests: notification dispatch integration
# ---------------------------------------------------------------------------

class TestRunWeeklyOpsNotify:
    @patch("run_weekly_ops.run_profile_gate")
    @patch("run_weekly_ops.execute_run_plan")
    def test_stdout_notify_on_success(self, mock_exec, mock_gate, tmp_path, capsys):
        mock_gate.return_value = _make_gate_pass()
        mock_exec.return_value = [
            SymbolRunResult("ES", "EXECUTED", "ok", 5, 3, 1.5, 0.67),
        ]

        profile_path = _write_profile_yaml(tmp_path)

        run_weekly_ops(
            profile_path=profile_path,
            dry_run=True,
            notify_mode="stdout",
            ops_log_dir=tmp_path / "ops",
            summary_dir=tmp_path / "summaries",
        )

        captured = capsys.readouterr()
        assert "[OK]" in captured.out

    @patch("run_weekly_ops.run_profile_gate")
    def test_stdout_notify_on_gate_fail(self, mock_gate, tmp_path, capsys):
        mock_gate.return_value = _make_gate_fail()

        profile_path = _write_profile_yaml(tmp_path)

        run_weekly_ops(
            profile_path=profile_path,
            notify_mode="stdout",
            ops_log_dir=tmp_path / "ops",
            summary_dir=tmp_path / "summaries",
        )

        captured = capsys.readouterr()
        assert "[ALERT]" in captured.out

    @patch("run_weekly_ops.dispatch_notification")
    @patch("run_weekly_ops.run_profile_gate")
    @patch("run_weekly_ops.execute_run_plan")
    def test_webhook_failure_does_not_crash(self, mock_exec, mock_gate, mock_dispatch, tmp_path):
        mock_gate.return_value = _make_gate_pass()
        mock_exec.return_value = [SymbolRunResult("ES", "EXECUTED", "ok")]
        mock_dispatch.side_effect = Exception("webhook exploded")

        profile_path = _write_profile_yaml(tmp_path)

        # Should not raise despite dispatch failure.
        with pytest.raises(Exception, match="webhook exploded"):
            run_weekly_ops(
                profile_path=profile_path,
                notify_mode="webhook",
                webhook_url="https://example.com",
                ops_log_dir=tmp_path / "ops",
                summary_dir=tmp_path / "summaries",
            )


# ---------------------------------------------------------------------------
# Tests: JSON output shape
# ---------------------------------------------------------------------------

class TestOpsResultShape:
    @patch("run_weekly_ops.run_profile_gate")
    @patch("run_weekly_ops.execute_run_plan")
    def test_success_shape(self, mock_exec, mock_gate, tmp_path):
        mock_gate.return_value = _make_gate_pass()
        mock_exec.return_value = [
            SymbolRunResult("ES", "EXECUTED", "ok", 5, 3, 1.5, 0.67),
        ]

        profile_path = _write_profile_yaml(tmp_path)
        result = run_weekly_ops(
            profile_path=profile_path,
            notify_mode="none",
            ops_log_dir=tmp_path / "ops",
            summary_dir=tmp_path / "summaries",
        )

        # Must be JSON-serialisable.
        serialized = json.dumps(result)
        parsed = json.loads(serialized)

        assert "timestamp" in parsed
        assert "gate_passed" in parsed
        assert "aborted" in parsed
        assert "dry_run" in parsed
        assert "exit_code" in parsed
        assert "symbol_run_results" in parsed
        assert "retention" in parsed
        assert "run_summary" in parsed

    @patch("run_weekly_ops.run_profile_gate")
    def test_abort_shape(self, mock_gate, tmp_path):
        mock_gate.return_value = _make_gate_fail()

        profile_path = _write_profile_yaml(tmp_path)
        result = run_weekly_ops(
            profile_path=profile_path,
            notify_mode="none",
            ops_log_dir=tmp_path / "ops",
            summary_dir=tmp_path / "summaries",
        )

        serialized = json.dumps(result)
        parsed = json.loads(serialized)

        assert parsed["aborted"] is True
        assert parsed["exit_code"] == 2
        assert "gate_result" in parsed
        assert "retention" in parsed
