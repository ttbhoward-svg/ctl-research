"""Unit tests for canonical futures acceptance framework (H.5)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.canonical_acceptance import (
    AcceptanceThresholds,
    FuturesAcceptanceInput,
    FuturesAcceptanceResult,
    evaluate_futures_acceptance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_input(**overrides) -> FuturesAcceptanceInput:
    """Build a FuturesAcceptanceInput with sensible defaults (all-pass)."""
    defaults = dict(
        symbol="/ES",
        n_canonical=32,
        n_ts=32,
        n_paired=31,
        n_matched=28,
        n_watch=3,
        n_fail=0,
        unmatched_canonical=1,
        unmatched_ts=1,
        mean_gap_diff=0.5,
        max_gap_diff=1.2,
        mean_drift=2.0,
        max_drift=8.0,
        strict_status="FAIL",
        policy_status="WATCH",
    )
    defaults.update(overrides)
    return FuturesAcceptanceInput(**defaults)


# ---------------------------------------------------------------------------
# Tests: AcceptanceThresholds
# ---------------------------------------------------------------------------

class TestAcceptanceThresholds:
    def test_defaults_correct(self):
        t = AcceptanceThresholds()
        assert t.max_unmatched_frac == 0.10
        assert t.max_fail_frac == 0.15
        assert t.max_mean_gap_diff == 1.0
        assert t.max_mean_drift == 5.0
        assert t.min_paired_rolls == 20

    def test_custom_thresholds_used(self):
        t = AcceptanceThresholds(min_paired_rolls=5, max_mean_drift=50.0)
        assert t.min_paired_rolls == 5
        assert t.max_mean_drift == 50.0


# ---------------------------------------------------------------------------
# Tests: evaluate_futures_acceptance
# ---------------------------------------------------------------------------

class TestEvaluateFuturesAcceptance:
    def test_all_pass_accept(self):
        inp = _make_input()
        result = evaluate_futures_acceptance(inp)
        assert result.decision == "ACCEPT"
        assert result.accepted is True
        assert result.reasons == []

    def test_hard_fail_reject_too_few_paired(self):
        inp = _make_input(n_paired=5)
        result = evaluate_futures_acceptance(inp)
        assert result.decision == "REJECT"
        assert result.accepted is False
        assert any("Too few paired" in r for r in result.reasons)

    def test_hard_fail_reject_unmatched(self):
        # unmatched_frac = (20+20) / (32+32) = 0.625 > 0.10
        inp = _make_input(unmatched_canonical=20, unmatched_ts=20)
        result = evaluate_futures_acceptance(inp)
        assert result.decision == "REJECT"
        assert any("unmatched" in r for r in result.reasons)

    def test_hard_fail_reject_fail_frac(self):
        # fail_frac = 20 / 31 ≈ 0.645 > 0.15
        inp = _make_input(n_fail=20)
        result = evaluate_futures_acceptance(inp)
        assert result.decision == "REJECT"
        assert any("FAIL matches" in r for r in result.reasons)

    def test_soft_fail_watch_drift(self):
        inp = _make_input(mean_drift=10.0)
        result = evaluate_futures_acceptance(inp)
        assert result.decision == "WATCH"
        assert result.accepted is False
        assert any("drift" in r.lower() for r in result.reasons)

    def test_soft_fail_watch_gap_diff(self):
        inp = _make_input(mean_gap_diff=5.0)
        result = evaluate_futures_acceptance(inp)
        assert result.decision == "WATCH"
        assert result.accepted is False
        assert any("gap diff" in r.lower() for r in result.reasons)

    def test_border_case_exact_threshold(self):
        # Exactly at threshold should pass.
        t = AcceptanceThresholds()
        inp = _make_input(
            mean_gap_diff=t.max_mean_gap_diff,
            mean_drift=t.max_mean_drift,
        )
        result = evaluate_futures_acceptance(inp, t)
        assert result.decision == "ACCEPT"

    def test_reasons_populated_on_multiple_failures(self):
        inp = _make_input(n_paired=5, mean_drift=20.0, mean_gap_diff=10.0)
        result = evaluate_futures_acceptance(inp)
        # Hard fail (paired) + 2 soft fails.
        assert len(result.reasons) == 3
        assert result.decision == "REJECT"

    def test_zero_paired_rejects(self):
        inp = _make_input(n_paired=0, n_fail=0)
        result = evaluate_futures_acceptance(inp)
        assert result.decision == "REJECT"
        assert any("Too few paired" in r for r in result.reasons)


# ---------------------------------------------------------------------------
# Tests: FuturesAcceptanceResult.to_dict
# ---------------------------------------------------------------------------

class TestAcceptanceResultDict:
    def test_to_dict_has_expected_keys(self):
        inp = _make_input()
        result = evaluate_futures_acceptance(inp)
        d = result.to_dict()
        expected_keys = {"symbol", "accepted", "decision", "reasons",
                         "thresholds_used", "input_summary"}
        assert set(d.keys()) == expected_keys

    def test_to_dict_deterministic(self):
        inp = _make_input()
        r1 = evaluate_futures_acceptance(inp)
        r2 = evaluate_futures_acceptance(inp)
        assert r1.to_dict() == r2.to_dict()


# ---------------------------------------------------------------------------
# Tests: acceptance_from_diagnostics (integration-style with mock)
# ---------------------------------------------------------------------------

class TestAcceptanceFromDiagnostics:
    """Tests for the convenience function using a lightweight mock."""

    def _make_mock_diag(self, **overrides):
        """Build a minimal mock DiagnosticResult."""
        defaults = _make_input(**overrides)

        class _MockComparison:
            n_canonical = defaults.n_canonical
            n_ts = defaults.n_ts
            n_paired = defaults.n_paired
            n_matched = defaults.n_matched
            n_watch = defaults.n_watch
            n_fail = defaults.n_fail
            unmatched_canonical = defaults.unmatched_canonical
            unmatched_ts = defaults.unmatched_ts

        class _MockL2:
            comparison = _MockComparison()

        class _MockL3:
            mean_gap_diff = defaults.mean_gap_diff
            max_gap_diff = defaults.max_gap_diff

        class _MockL4:
            mean_drift = defaults.mean_drift
            max_drift = defaults.max_drift

        class _MockDiag:
            symbol = defaults.symbol
            l2 = _MockL2()
            l3 = _MockL3()
            l4 = _MockL4()
            strict_status = defaults.strict_status
            policy_status = defaults.policy_status

        return _MockDiag()

    def test_convenience_matches_manual(self):
        from ctl.canonical_acceptance import acceptance_from_diagnostics

        mock_diag = self._make_mock_diag()
        result = acceptance_from_diagnostics(mock_diag)
        # Should match manual evaluate with same inputs.
        inp = _make_input()
        manual = evaluate_futures_acceptance(inp)
        assert result.decision == manual.decision
        assert result.accepted == manual.accepted

    def test_convenience_handles_edge_case(self):
        from ctl.canonical_acceptance import acceptance_from_diagnostics

        mock_diag = self._make_mock_diag(n_paired=0)
        result = acceptance_from_diagnostics(mock_diag)
        assert result.decision == "REJECT"
