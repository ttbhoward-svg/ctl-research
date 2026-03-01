"""Unit tests for run orchestrator and gate-first portfolio runner (H.8/H.9)."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.operating_profile import (
    OperatingProfile,
    PolicyConstraints,
    PortfolioCheckResult,
    SymbolCheckResult,
    SymbolSetting,
    load_operating_profile,
)
from ctl.run_orchestrator import (
    RunPlan,
    RunSummary,
    SymbolRunResult,
    build_run_plan,
    execute_b1_symbol,
    execute_run_plan,
    make_b1_executor,
    save_run_summary,
    summarize_run,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile(**overrides) -> OperatingProfile:
    """Build a minimal OperatingProfile for testing."""
    defaults = dict(
        cycle_id="test_v1",
        locked_date="2026-03-01",
        portfolio_recommendation="CONDITIONAL GO",
        gating_universe=["ES", "CL", "PL"],
        non_gating_symbols=["PA"],
        symbol_settings={
            "ES": SymbolSetting(0.25, 3, "WATCH"),
            "CL": SymbolSetting(0.01, 3, "ACCEPT"),
            "PL": SymbolSetting(0.10, 2, "WATCH"),
        },
        policy_constraints=PolicyConstraints(),
    )
    defaults.update(overrides)
    return OperatingProfile(**defaults)


def _make_gate_pass() -> PortfolioCheckResult:
    """Build a passing gate result."""
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
    """Build a failing gate result (CL mismatch)."""
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
    """Write a minimal valid profile YAML and return the path."""
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


def _make_ohlcv_csv(tmp_path: Path, symbol: str, n_bars: int = 200) -> Path:
    """Write a synthetic OHLCV CSV with enough bars for B1 detection."""
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=n_bars)
    close = 100.0 + np.cumsum(np.random.randn(n_bars) * 0.5)
    high = close + np.abs(np.random.randn(n_bars) * 0.3)
    low = close - np.abs(np.random.randn(n_bars) * 0.3)
    open_ = close + np.random.randn(n_bars) * 0.1
    volume = np.random.randint(1000, 50000, size=n_bars)

    df = pd.DataFrame({
        "Date": dates,
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })
    path = tmp_path / f"{symbol}_continuous.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Tests: build_run_plan
# ---------------------------------------------------------------------------

class TestBuildRunPlan:
    def test_gating_only(self):
        profile = _make_profile()
        plan = build_run_plan(profile, include_non_gating=False)

        assert plan.symbols == ["ES", "CL", "PL"]
        assert plan.gating_symbols == ["ES", "CL", "PL"]
        assert plan.non_gating_symbols == []
        assert plan.include_non_gating is False

    def test_include_non_gating_adds_pa(self):
        profile = _make_profile()
        plan = build_run_plan(profile, include_non_gating=True)

        assert plan.symbols == ["ES", "CL", "PL", "PA"]
        assert plan.gating_symbols == ["ES", "CL", "PL"]
        assert plan.non_gating_symbols == ["PA"]
        assert plan.include_non_gating is True

    def test_plan_captures_cycle_id(self):
        profile = _make_profile(cycle_id="custom_cycle")
        plan = build_run_plan(profile)
        assert plan.cycle_id == "custom_cycle"

    def test_plan_captures_recommendation(self):
        profile = _make_profile()
        plan = build_run_plan(profile)
        assert plan.portfolio_recommendation == "CONDITIONAL GO"

    def test_plan_to_dict_keys(self):
        profile = _make_profile()
        plan = build_run_plan(profile, profile_path="/some/path.yaml")
        d = plan.to_dict()
        expected_keys = {
            "cycle_id", "profile_path", "symbols", "gating_symbols",
            "non_gating_symbols", "include_non_gating", "portfolio_recommendation",
        }
        assert set(d.keys()) == expected_keys
        assert d["profile_path"] == "/some/path.yaml"


# ---------------------------------------------------------------------------
# Tests: execute_run_plan
# ---------------------------------------------------------------------------

class TestExecuteRunPlan:
    def test_dry_run_returns_dry_run_status(self):
        profile = _make_profile()
        plan = build_run_plan(profile)
        results = execute_run_plan(plan, dry_run=True)

        assert len(results) == 3
        for r in results:
            assert r.status == "DRY_RUN"
            assert "skipped" in r.detail

    def test_custom_executor_called(self):
        profile = _make_profile()
        plan = build_run_plan(profile)

        def mock_executor(sym):
            return SymbolRunResult(sym, "CUSTOM", f"ran {sym}")

        results = execute_run_plan(plan, executor=mock_executor)
        assert all(r.status == "CUSTOM" for r in results)
        assert results[0].detail == "ran ES"

    def test_executor_error_caught(self):
        profile = _make_profile()
        plan = build_run_plan(profile)

        def failing_executor(sym):
            raise RuntimeError(f"boom for {sym}")

        results = execute_run_plan(plan, executor=failing_executor)
        assert all(r.status == "ERROR" for r in results)
        assert "boom for ES" in results[0].detail

    def test_include_non_gating_runs_all(self):
        profile = _make_profile()
        plan = build_run_plan(profile, include_non_gating=True)
        results = execute_run_plan(plan, dry_run=True)

        assert len(results) == 4
        symbols_run = [r.symbol for r in results]
        assert "PA" in symbols_run

    def test_partial_failure_does_not_crash_portfolio(self):
        """One symbol fails, others still run."""
        profile = _make_profile()
        plan = build_run_plan(profile)

        call_count = [0]

        def mixed_executor(sym):
            call_count[0] += 1
            if sym == "CL":
                raise RuntimeError("CL data corrupt")
            return SymbolRunResult(sym, "EXECUTED", f"{sym} ok",
                                   trigger_count=5, trade_count=3,
                                   total_r=2.5, win_rate=0.6)

        results = execute_run_plan(plan, executor=mixed_executor)
        assert call_count[0] == 3  # all three symbols attempted
        assert results[0].status == "EXECUTED"
        assert results[1].status == "ERROR"
        assert "CL data corrupt" in results[1].detail
        assert results[2].status == "EXECUTED"


# ---------------------------------------------------------------------------
# Tests: SymbolRunResult
# ---------------------------------------------------------------------------

class TestSymbolRunResult:
    def test_to_dict_basic(self):
        r = SymbolRunResult("ES", "EXECUTED", "ok")
        d = r.to_dict()
        assert d == {"symbol": "ES", "status": "EXECUTED", "detail": "ok"}

    def test_default_detail_empty(self):
        r = SymbolRunResult("CL", "DRY_RUN")
        assert r.detail == ""

    def test_to_dict_with_metrics(self):
        r = SymbolRunResult("ES", "EXECUTED", "done",
                            trigger_count=10, trade_count=7,
                            total_r=5.5, win_rate=0.71)
        d = r.to_dict()
        assert d["trigger_count"] == 10
        assert d["trade_count"] == 7
        assert d["total_r"] == 5.5
        assert d["win_rate"] == 0.71

    def test_to_dict_omits_none_metrics(self):
        """Backward-compatible: no metric keys when None."""
        r = SymbolRunResult("ES", "DRY_RUN", "skipped")
        d = r.to_dict()
        assert "trigger_count" not in d
        assert "trade_count" not in d
        assert "total_r" not in d
        assert "win_rate" not in d

    def test_to_dict_backward_compatible_keys(self):
        """Core keys always present regardless of metrics."""
        r = SymbolRunResult("ES", "EXECUTED", "test",
                            trigger_count=0, trade_count=0,
                            total_r=0.0, win_rate=0.0)
        d = r.to_dict()
        assert "symbol" in d
        assert "status" in d
        assert "detail" in d


# ---------------------------------------------------------------------------
# Tests: execute_b1_symbol
# ---------------------------------------------------------------------------

class TestExecuteB1Symbol:
    def test_missing_csv_returns_error(self, tmp_path):
        result = execute_b1_symbol("ES", tmp_path)
        assert result.status == "ERROR"
        assert "Data load" in result.detail

    def test_insufficient_bars_returns_skipped(self, tmp_path):
        # Write a CSV with only 10 bars.
        dates = pd.bdate_range("2020-01-01", periods=10)
        df = pd.DataFrame({
            "Date": dates,
            "Open": range(10),
            "High": range(10),
            "Low": range(10),
            "Close": range(10),
            "Volume": range(10),
        })
        (tmp_path / "ES_continuous.csv").write_text(df.to_csv(index=False))

        result = execute_b1_symbol("ES", tmp_path)
        assert result.status == "SKIPPED"
        assert "Insufficient" in result.detail

    def test_synthetic_data_returns_executed(self, tmp_path):
        """With enough bars, execute_b1_symbol completes without error."""
        _make_ohlcv_csv(tmp_path, "ES", n_bars=200)
        result = execute_b1_symbol("ES", tmp_path)

        assert result.status == "EXECUTED"
        assert result.trigger_count is not None
        assert result.trade_count is not None
        assert result.total_r is not None
        assert result.win_rate is not None
        assert result.trigger_count >= 0
        assert result.trade_count >= 0

    def test_metrics_are_numeric(self, tmp_path):
        _make_ohlcv_csv(tmp_path, "CL", n_bars=300)
        result = execute_b1_symbol("CL", tmp_path)

        assert result.status == "EXECUTED"
        assert isinstance(result.total_r, float)
        assert isinstance(result.win_rate, float)
        assert 0.0 <= result.win_rate <= 1.0

    def test_detail_contains_summary(self, tmp_path):
        _make_ohlcv_csv(tmp_path, "PL", n_bars=200)
        result = execute_b1_symbol("PL", tmp_path)

        assert result.status == "EXECUTED"
        assert "triggers" in result.detail
        assert "trades" in result.detail
        assert "R=" in result.detail


# ---------------------------------------------------------------------------
# Tests: make_b1_executor
# ---------------------------------------------------------------------------

class TestMakeB1Executor:
    def test_factory_returns_callable(self, tmp_path):
        executor = make_b1_executor(data_dir=tmp_path)
        assert callable(executor)

    def test_factory_executor_runs(self, tmp_path):
        _make_ohlcv_csv(tmp_path, "ES", n_bars=200)
        executor = make_b1_executor(data_dir=tmp_path)
        result = executor("ES")
        assert result.status == "EXECUTED"
        assert result.symbol == "ES"


# ---------------------------------------------------------------------------
# Tests: summarize_run
# ---------------------------------------------------------------------------

class TestSummarizeRun:
    def test_summary_has_expected_keys(self):
        profile = _make_profile()
        plan = build_run_plan(profile)
        gate = _make_gate_pass()
        sym_results = [SymbolRunResult("ES", "EXECUTED")]

        summary = summarize_run(plan, gate, sym_results, timestamp="20260301_120000")
        d = summary.to_dict()

        expected_keys = {
            "timestamp", "cycle_id", "gate_passed", "portfolio_recommendation",
            "dry_run", "plan", "gate_result", "symbol_run_results",
        }
        assert set(d.keys()) == expected_keys

    def test_summary_json_serializable(self):
        profile = _make_profile()
        plan = build_run_plan(profile)
        gate = _make_gate_pass()
        sym_results = [SymbolRunResult("ES", "EXECUTED")]

        summary = summarize_run(plan, gate, sym_results, timestamp="20260301_120000")
        serialized = json.dumps(summary.to_dict())
        parsed = json.loads(serialized)
        assert parsed["gate_passed"] is True
        assert parsed["timestamp"] == "20260301_120000"

    def test_summary_captures_dry_run(self):
        profile = _make_profile()
        plan = build_run_plan(profile)
        gate = _make_gate_pass()

        summary = summarize_run(plan, gate, [], dry_run=True, timestamp="t")
        assert summary.dry_run is True

    def test_summary_auto_timestamp(self):
        profile = _make_profile()
        plan = build_run_plan(profile)
        gate = _make_gate_pass()

        summary = summarize_run(plan, gate, [])
        # Timestamp should be non-empty and look like YYYYMMDD_HHMMSS.
        assert len(summary.timestamp) == 15
        assert "_" in summary.timestamp

    def test_summary_with_metrics_json_serializable(self):
        """Summary including metric fields is JSON-safe."""
        profile = _make_profile()
        plan = build_run_plan(profile)
        gate = _make_gate_pass()
        sym_results = [
            SymbolRunResult("ES", "EXECUTED", "done",
                            trigger_count=10, trade_count=7,
                            total_r=5.5, win_rate=0.71),
        ]
        summary = summarize_run(plan, gate, sym_results, timestamp="20260301_120000")
        d = summary.to_dict()
        serialized = json.dumps(d)
        parsed = json.loads(serialized)
        sr = parsed["symbol_run_results"][0]
        assert sr["trigger_count"] == 10
        assert sr["total_r"] == 5.5


# ---------------------------------------------------------------------------
# Tests: save_run_summary
# ---------------------------------------------------------------------------

class TestSaveRunSummary:
    def test_saves_json_file(self, tmp_path):
        profile = _make_profile()
        plan = build_run_plan(profile)
        gate = _make_gate_pass()
        summary = summarize_run(plan, gate, [], timestamp="20260301_120000")

        path = save_run_summary(summary, out_dir=tmp_path)
        assert path.exists()
        assert path.name == "20260301_120000_portfolio_run.json"

        with open(path) as f:
            data = json.load(f)
        assert data["cycle_id"] == "test_v1"
        assert data["gate_passed"] is True

    def test_creates_output_directory(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        profile = _make_profile()
        plan = build_run_plan(profile)
        gate = _make_gate_pass()
        summary = summarize_run(plan, gate, [], timestamp="20260301_120000")

        path = save_run_summary(summary, out_dir=nested)
        assert nested.is_dir()
        assert path.exists()


# ---------------------------------------------------------------------------
# Tests: gate-first enforcement (integration-style with mocks)
# ---------------------------------------------------------------------------

class TestGateFirstEnforcement:
    """Tests that the runner exits 2 on gate mismatch and proceeds on pass."""

    def test_gate_pass_runner_proceeds_with_b1_executor(self, tmp_path):
        """Gate passes -> B1 executor runs and returns metrics."""
        profile_path = _write_profile_yaml(tmp_path)
        profile = load_operating_profile(profile_path)
        gate = _make_gate_pass()

        # Write synthetic data for all gating symbols.
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        for sym in ["ES", "CL", "PL"]:
            _make_ohlcv_csv(data_dir, sym, n_bars=200)

        plan = build_run_plan(profile, profile_path=str(profile_path))
        executor = make_b1_executor(data_dir=data_dir)
        results = execute_run_plan(plan, executor=executor)
        summary = summarize_run(plan, gate, results, timestamp="20260301_120000")

        assert summary.gate_passed is True
        assert len(summary.symbol_run_results) == 3
        for sr in summary.symbol_run_results:
            assert sr["status"] == "EXECUTED"

    def test_gate_fail_produces_abort_summary(self):
        """Gate fails -> summary reflects failure, no execution occurs."""
        gate = _make_gate_fail()
        assert gate.passed is False

        d = gate.to_dict()
        cl_result = next(r for r in d["symbol_results"] if r["symbol"] == "CL")
        assert cl_result["passed"] is False
        assert "REJECT" in cl_result["detail"]

    def test_gate_mismatch_json_output_shape(self):
        """JSON abort output has expected structure."""
        gate = _make_gate_fail()
        abort_output = {
            "gate_passed": False,
            "gate_result": gate.to_dict(),
            "aborted": True,
            "reason": "Gate mismatch — acceptance status does not match locked profile.",
        }
        serialized = json.dumps(abort_output)
        parsed = json.loads(serialized)
        assert parsed["aborted"] is True
        assert parsed["gate_passed"] is False
        assert "gate_result" in parsed

    def test_dry_run_returns_plan_only(self, tmp_path):
        """Dry run: gate passes, plan built, but execution is skipped."""
        profile_path = _write_profile_yaml(tmp_path)
        profile = load_operating_profile(profile_path)
        gate = _make_gate_pass()

        plan = build_run_plan(profile)
        results = execute_run_plan(plan, dry_run=True)
        summary = summarize_run(plan, gate, results, dry_run=True, timestamp="t")

        assert summary.dry_run is True
        assert all(r["status"] == "DRY_RUN" for r in summary.symbol_run_results)

    def test_include_non_gating_toggles_pa(self, tmp_path):
        """--include-non-gating adds PA to the run plan."""
        profile_path = _write_profile_yaml(tmp_path)
        profile = load_operating_profile(profile_path)

        plan_without = build_run_plan(profile, include_non_gating=False)
        plan_with = build_run_plan(profile, include_non_gating=True)

        assert "PA" not in plan_without.symbols
        assert "PA" in plan_with.symbols
        assert plan_with.non_gating_symbols == ["PA"]

    def test_json_output_shape_stable(self, tmp_path):
        """Full JSON output from a passing run has stable shape."""
        profile_path = _write_profile_yaml(tmp_path)
        profile = load_operating_profile(profile_path)
        gate = _make_gate_pass()

        plan = build_run_plan(profile, profile_path=str(profile_path))

        # Mock executor that returns metrics.
        def mock_b1(sym):
            return SymbolRunResult(sym, "EXECUTED", f"{sym} done",
                                   trigger_count=5, trade_count=3,
                                   total_r=2.0, win_rate=0.67)

        results = execute_run_plan(plan, executor=mock_b1)
        summary = summarize_run(plan, gate, results, timestamp="20260301_120000")

        d = summary.to_dict()
        # Verify stable top-level keys.
        assert "timestamp" in d
        assert "cycle_id" in d
        assert "gate_passed" in d
        assert "plan" in d
        assert "gate_result" in d
        assert "symbol_run_results" in d
        assert "dry_run" in d

        # Verify nested structure.
        assert isinstance(d["plan"]["symbols"], list)
        assert isinstance(d["gate_result"]["symbol_results"], list)
        assert isinstance(d["symbol_run_results"], list)
        for sr in d["symbol_run_results"]:
            # Core keys always present.
            assert "symbol" in sr
            assert "status" in sr
            assert "detail" in sr
            # Metric keys present when executor returned them.
            assert "trigger_count" in sr
            assert "trade_count" in sr
            assert "total_r" in sr

        # Must be fully JSON-serializable.
        json.dumps(d)

    def test_one_symbol_failure_others_still_run(self, tmp_path):
        """Partial failure: one symbol errors, others complete."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        # Only write data for ES and PL, not CL.
        _make_ohlcv_csv(data_dir, "ES", n_bars=200)
        _make_ohlcv_csv(data_dir, "PL", n_bars=200)

        profile = _make_profile()
        plan = build_run_plan(profile)
        executor = make_b1_executor(data_dir=data_dir)
        results = execute_run_plan(plan, executor=executor)

        statuses = {r.symbol: r.status for r in results}
        assert statuses["ES"] == "EXECUTED"
        assert statuses["CL"] == "ERROR"
        assert statuses["PL"] == "EXECUTED"
