"""Unit tests for Gate 1 Pass/Fail Decision Package (Task 14)."""

import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.entry_degradation import AggregateMetrics, DegradationDeltas, DegradationReport
from ctl.gate_decision import (
    GateCriterion,
    GateDecision,
    GateInput,
    KillCheckResult,
    evaluate_gate1,
)
from ctl.negative_controls import BaselineMetrics, ControlResult, NegativeControlReport
from ctl.oos_evaluation import BucketMetrics, CalibrationRow, CriterionResult, OOSResult
from ctl.slippage_stress import DegradationResult, ScenarioMetrics, StressReport


# ---------------------------------------------------------------------------
# Builders — construct minimal upstream results for testing
# ---------------------------------------------------------------------------

def _make_oos_result(
    n_trades: int = 50,
    spread: float = 1.5,
    monotonic: bool = True,
    calibration_pass: bool = True,
    top_avg_r: float = 1.2,
    score_corr: float = 0.3,
) -> OOSResult:
    """Build a minimal OOSResult with controllable gate criteria."""
    # Tercile table.
    mid_r = 0.5 if monotonic else top_avg_r + 0.1
    bot_r = top_avg_r - spread
    tercile_table = [
        BucketMetrics("top", n_trades // 3, top_avg_r, 0.7, top_avg_r * (n_trades // 3)),
        BucketMetrics("mid", n_trades // 3, mid_r, 0.5, mid_r * (n_trades // 3)),
        BucketMetrics("bottom", n_trades - 2 * (n_trades // 3), bot_r, 0.3, bot_r * (n_trades - 2 * (n_trades // 3))),
    ]

    # Quintile table (monotonic or not).
    if calibration_pass:
        q_avg = [0.1, 0.3, 0.5, 0.7, 1.0]
    else:
        q_avg = [0.1, 0.5, 0.3, 0.7, 1.0]  # Q2 > Q3 breaks monotonicity
    quintile_table = [
        BucketMetrics(f"Q{i+1}", n_trades // 5, q_avg[i], 0.5, q_avg[i] * (n_trades // 5))
        for i in range(5)
    ]

    # Criteria.
    criteria = [
        CriterionResult("G1.1", "oos_trade_count", n_trades >= 30, float(n_trades), 30.0,
                         f"OOS trades = {n_trades}"),
        CriterionResult("G1.2", "oos_tercile_spread", spread >= 1.0, spread, 1.0,
                         f"Spread = {spread:.4f}R"),
        CriterionResult("G1.3", "score_monotonicity", monotonic, 0.0, 0.0,
                         "Monotonic" if monotonic else "Monotonicity broken"),
        CriterionResult("G1.9", "quintile_calibration", calibration_pass, 0.0, 0.0,
                         "Calibrated" if calibration_pass else "Calibration broken"),
        CriterionResult("K.1", "top_tercile_min_r", top_avg_r >= 0.5, top_avg_r, 0.5,
                         f"Top avg R = {top_avg_r:.4f}"),
        CriterionResult("K.2", "score_outcome_correlation", score_corr >= 0.05, score_corr, 0.05,
                         f"Corr = {score_corr:.4f}"),
    ]

    return OOSResult(
        n_trades=n_trades,
        scores=np.zeros(n_trades),
        tercile_buckets=np.array(["top"] * (n_trades // 3) + ["mid"] * (n_trades // 3)
                                  + ["bottom"] * (n_trades - 2 * (n_trades // 3))),
        quintile_labels=np.array([f"Q{(i % 5) + 1}" for i in range(n_trades)]),
        tercile_table=tercile_table,
        quintile_table=quintile_table,
        calibration_table=[],
        tercile_spread=spread,
        score_r_correlation=score_corr,
        criteria=criteria,
    )


def _make_stress_report(gate_pass: bool = True) -> StressReport:
    return StressReport(
        scenarios=[
            ScenarioMetrics(0, 50, 0, 50.0, 1.0, 0.7, 45.0, 5.0, 10.0),
            ScenarioMetrics(2, 50, 0, 40.0, 0.8, 0.65, 36.0, 6.0, 6.67),
        ],
        degradations=[
            DegradationResult(2, -20.0, -5.0, -33.3),
        ],
        profitable_at_2_ticks=gate_pass,
        profitable_at_3_ticks=None,
        gate_pass=gate_pass,
    )


def _make_nc_report(all_passed: bool = True) -> NegativeControlReport:
    return NegativeControlReport(
        baseline=BaselineMetrics(r_squared=0.15, tercile_spread=1.2,
                                  tercile_monotonic=True, n_trades=120),
        controls=[
            ControlResult("randomized_labels", all_passed, "ok"),
            ControlResult("lag_shift", all_passed, "ok"),
            ControlResult("placebo_feature", all_passed, "ok"),
        ],
    )


def _make_degradation_report(all_passed: bool = True) -> DegradationReport:
    return DegradationReport(
        mode="adverse_fill",
        degradation_pct=0.30,
        seed=42,
        n_total=50,
        n_degraded=15,
        n_excluded=0,
        baseline=AggregateMetrics(50, 50.0, 1.0, 0.7, 5.0, 10.0),
        degraded=AggregateMetrics(50, 45.0, 0.9, 0.68, 5.5, 8.18),
        deltas=DegradationDeltas(-10.0, -2.0, -18.2),
        total_r_pass=all_passed,
        win_rate_pass=all_passed,
        mar_pass=all_passed,
    )


def _make_all_pass_input() -> GateInput:
    """Build a GateInput where all 9 criteria should pass."""
    return GateInput(
        oos_result=_make_oos_result(),
        stress_report=_make_stress_report(True),
        nc_report=_make_nc_report(True),
        degradation_report=_make_degradation_report(True),
        feature_cap_respected=True,
        model_card_complete=True,
        dataset_hash="sha256_abc123",
        config_hash="sha256_def456",
        code_commit_hash="1a2b3c4d",
    )


# ---------------------------------------------------------------------------
# Tests: all-pass case
# ---------------------------------------------------------------------------

class TestAllPass:
    def test_verdict_is_pass(self):
        decision = evaluate_gate1(_make_all_pass_input())
        assert decision.verdict == "PASS"

    def test_nine_criteria(self):
        decision = evaluate_gate1(_make_all_pass_input())
        assert len(decision.criteria) == 9

    def test_all_criteria_pass(self):
        decision = evaluate_gate1(_make_all_pass_input())
        assert decision.n_passed == 9
        assert decision.n_failed == 0
        assert decision.n_incomplete == 0

    def test_no_kill_flags(self):
        decision = evaluate_gate1(_make_all_pass_input())
        assert not decision.kill_check.any_reject
        assert not decision.kill_check.any_pause

    def test_no_failed_items(self):
        decision = evaluate_gate1(_make_all_pass_input())
        assert decision.failed_items == []

    def test_timestamp_populated(self):
        decision = evaluate_gate1(_make_all_pass_input())
        assert len(decision.timestamp) > 0

    def test_hashes_propagated(self):
        decision = evaluate_gate1(_make_all_pass_input())
        assert decision.dataset_hash == "sha256_abc123"
        assert decision.config_hash == "sha256_def456"
        assert decision.code_commit_hash == "1a2b3c4d"


# ---------------------------------------------------------------------------
# Tests: single-fail cases
# ---------------------------------------------------------------------------

class TestSingleFail:
    def test_oos_trade_count_fail(self):
        inp = _make_all_pass_input()
        inp.oos_result = _make_oos_result(n_trades=20)
        decision = evaluate_gate1(inp)
        assert decision.verdict == "ITERATE"
        assert decision.n_failed >= 1
        failed_names = [c.name for c in decision.failed_items]
        assert "oos_trade_count" in failed_names

    def test_tercile_spread_fail(self):
        inp = _make_all_pass_input()
        inp.oos_result = _make_oos_result(spread=0.5)
        decision = evaluate_gate1(inp)
        assert decision.verdict == "ITERATE"
        failed_names = [c.name for c in decision.failed_items]
        assert "oos_tercile_spread" in failed_names

    def test_negative_controls_fail(self):
        inp = _make_all_pass_input()
        inp.nc_report = _make_nc_report(all_passed=False)
        decision = evaluate_gate1(inp)
        assert decision.verdict == "ITERATE"
        failed_names = [c.name for c in decision.failed_items]
        assert "negative_controls_passed" in failed_names

    def test_entry_degradation_fail(self):
        inp = _make_all_pass_input()
        inp.degradation_report = _make_degradation_report(all_passed=False)
        decision = evaluate_gate1(inp)
        assert decision.verdict == "ITERATE"
        failed_names = [c.name for c in decision.failed_items]
        assert "entry_degradation_pass" in failed_names

    def test_slippage_stress_fail(self):
        inp = _make_all_pass_input()
        inp.stress_report = _make_stress_report(gate_pass=False)
        decision = evaluate_gate1(inp)
        assert decision.verdict == "ITERATE"
        failed_names = [c.name for c in decision.failed_items]
        assert "slippage_stress_pass" in failed_names

    def test_feature_cap_fail(self):
        inp = _make_all_pass_input()
        inp.feature_cap_respected = False
        decision = evaluate_gate1(inp)
        assert decision.verdict == "ITERATE"
        failed_names = [c.name for c in decision.failed_items]
        assert "feature_cap_respected" in failed_names

    def test_model_card_fail(self):
        inp = _make_all_pass_input()
        inp.model_card_complete = False
        decision = evaluate_gate1(inp)
        assert decision.verdict == "ITERATE"
        failed_names = [c.name for c in decision.failed_items]
        assert "model_card_complete" in failed_names

    def test_calibration_fail(self):
        inp = _make_all_pass_input()
        inp.oos_result = _make_oos_result(calibration_pass=False)
        decision = evaluate_gate1(inp)
        assert decision.verdict == "ITERATE"
        failed_names = [c.name for c in decision.failed_items]
        assert "quintile_calibration" in failed_names


# ---------------------------------------------------------------------------
# Tests: multi-fail cases
# ---------------------------------------------------------------------------

class TestMultiFail:
    def test_two_criteria_fail(self):
        inp = _make_all_pass_input()
        inp.nc_report = _make_nc_report(False)
        inp.stress_report = _make_stress_report(False)
        decision = evaluate_gate1(inp)
        assert decision.verdict == "ITERATE"
        assert decision.n_failed >= 2

    def test_all_manual_attestations_fail(self):
        inp = _make_all_pass_input()
        inp.feature_cap_respected = False
        inp.model_card_complete = False
        decision = evaluate_gate1(inp)
        assert decision.n_failed >= 2


# ---------------------------------------------------------------------------
# Tests: kill / pause cases
# ---------------------------------------------------------------------------

class TestKillPause:
    def test_top_tercile_kill_triggers_reject(self):
        inp = _make_all_pass_input()
        inp.oos_result = _make_oos_result(top_avg_r=0.3)  # below 0.5
        decision = evaluate_gate1(inp)
        assert decision.verdict == "REJECT"
        assert decision.kill_check.top_tercile_kill

    def test_correlation_kill_triggers_reject(self):
        inp = _make_all_pass_input()
        inp.oos_result = _make_oos_result(score_corr=0.02)  # below 0.05
        decision = evaluate_gate1(inp)
        assert decision.verdict == "REJECT"
        assert decision.kill_check.correlation_kill

    def test_monotonicity_failure_triggers_pause(self):
        inp = _make_all_pass_input()
        inp.oos_result = _make_oos_result(monotonic=False)
        decision = evaluate_gate1(inp)
        assert decision.verdict == "PAUSE"
        assert decision.kill_check.monotonicity_pause

    def test_reject_overrides_pause(self):
        inp = _make_all_pass_input()
        # Both kill AND pause triggered.
        inp.oos_result = _make_oos_result(top_avg_r=0.3, monotonic=False)
        decision = evaluate_gate1(inp)
        assert decision.verdict == "REJECT"  # REJECT > PAUSE

    def test_kill_details_populated(self):
        inp = _make_all_pass_input()
        inp.oos_result = _make_oos_result(top_avg_r=0.3)
        decision = evaluate_gate1(inp)
        assert len(decision.kill_check.details) >= 1
        assert "REJECT" in decision.kill_check.details[0]


# ---------------------------------------------------------------------------
# Tests: incomplete (missing inputs)
# ---------------------------------------------------------------------------

class TestIncomplete:
    def test_no_oos_result(self):
        inp = _make_all_pass_input()
        inp.oos_result = None
        decision = evaluate_gate1(inp)
        assert decision.verdict == "INCOMPLETE"
        assert decision.n_incomplete >= 4  # items 1,2,3,9

    def test_no_stress_report(self):
        inp = _make_all_pass_input()
        inp.stress_report = None
        decision = evaluate_gate1(inp)
        assert decision.verdict == "INCOMPLETE"
        incomplete_names = [c.name for c in decision.incomplete_items]
        assert "slippage_stress_pass" in incomplete_names

    def test_no_nc_report(self):
        inp = _make_all_pass_input()
        inp.nc_report = None
        decision = evaluate_gate1(inp)
        assert decision.verdict == "INCOMPLETE"

    def test_no_degradation_report(self):
        inp = _make_all_pass_input()
        inp.degradation_report = None
        decision = evaluate_gate1(inp)
        assert decision.verdict == "INCOMPLETE"

    def test_manual_none(self):
        inp = _make_all_pass_input()
        inp.feature_cap_respected = None
        decision = evaluate_gate1(inp)
        assert decision.verdict == "INCOMPLETE"

    def test_all_none(self):
        inp = GateInput()
        decision = evaluate_gate1(inp)
        assert decision.verdict == "INCOMPLETE"
        assert decision.n_incomplete == 9

    def test_fail_overrides_incomplete(self):
        """ITERATE (from a FAIL) outranks INCOMPLETE."""
        inp = GateInput()
        inp.nc_report = _make_nc_report(False)  # explicit FAIL
        decision = evaluate_gate1(inp)
        assert decision.verdict == "ITERATE"


# ---------------------------------------------------------------------------
# Tests: deterministic rendering
# ---------------------------------------------------------------------------

class TestDeterministic:
    def test_same_input_same_verdict(self):
        inp = _make_all_pass_input()
        d1 = evaluate_gate1(inp)
        d2 = evaluate_gate1(inp)
        assert d1.verdict == d2.verdict
        assert d1.n_passed == d2.n_passed

    def test_criteria_order_stable(self):
        decision = evaluate_gate1(_make_all_pass_input())
        items = [c.item for c in decision.criteria]
        assert items == [1, 2, 3, 4, 5, 6, 7, 8, 9]


# ---------------------------------------------------------------------------
# Tests: remediation
# ---------------------------------------------------------------------------

class TestRemediation:
    def test_failed_criteria_have_remediation(self):
        inp = _make_all_pass_input()
        inp.nc_report = _make_nc_report(False)
        decision = evaluate_gate1(inp)
        for c in decision.failed_items:
            assert len(c.remediation) > 0

    def test_passed_criteria_no_remediation(self):
        decision = evaluate_gate1(_make_all_pass_input())
        for c in decision.criteria:
            if c.passed:
                assert c.remediation == ""


# ---------------------------------------------------------------------------
# Tests: summary and artifact schema
# ---------------------------------------------------------------------------

class TestArtifactSchema:
    def test_summary_contains_verdict(self):
        decision = evaluate_gate1(_make_all_pass_input())
        s = decision.summary()
        assert "GATE 1 DECISION" in s
        assert "PASS" in s

    def test_summary_contains_all_items(self):
        decision = evaluate_gate1(_make_all_pass_input())
        s = decision.summary()
        for i in range(1, 10):
            assert f"Item {i}" in s

    def test_to_dict_json_serializable(self):
        decision = evaluate_gate1(_make_all_pass_input())
        d = decision.to_dict()
        serialized = json.dumps(d)
        assert len(serialized) > 0

    def test_to_dict_has_required_keys(self):
        decision = evaluate_gate1(_make_all_pass_input())
        d = decision.to_dict()
        for key in ("verdict", "timestamp", "dataset_hash", "config_hash",
                     "code_commit_hash", "criteria", "kill_check",
                     "n_passed", "n_failed", "n_incomplete"):
            assert key in d

    def test_to_dict_criteria_structure(self):
        decision = evaluate_gate1(_make_all_pass_input())
        d = decision.to_dict()
        assert len(d["criteria"]) == 9
        entry = d["criteria"][0]
        for key in ("item", "name", "status", "value", "reason", "remediation"):
            assert key in entry

    def test_save_json_roundtrip(self):
        decision = evaluate_gate1(_make_all_pass_input())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "decision.json"
            decision.save_json(path)
            with open(path) as f:
                loaded = json.load(f)
        assert loaded["verdict"] == "PASS"
        assert len(loaded["criteria"]) == 9

    def test_save_markdown_creates_file(self):
        decision = evaluate_gate1(_make_all_pass_input())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "decision.md"
            decision.save_markdown(path)
            content = path.read_text()
        assert "Gate 1 Decision" in content
        assert "PASS" in content
        assert "| # |" in content  # table header

    def test_markdown_contains_remediation_on_fail(self):
        inp = _make_all_pass_input()
        inp.nc_report = _make_nc_report(False)
        decision = evaluate_gate1(inp)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "decision.md"
            decision.save_markdown(path)
            content = path.read_text()
        assert "Remediation" in content
