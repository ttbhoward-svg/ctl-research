"""Tests for OOS evaluation and Gate 1 checking."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.ranking_gate import (
    Gate1Result,
    KillCheck,
    QuintileResult,
    TercileResult,
    check_gate1,
    check_kill_criteria,
    evaluate_quintiles,
    evaluate_terciles,
)


# =========================================================================
# Tercile evaluation
# =========================================================================

class TestTercileEvaluation:
    def test_monotonic_spread(self):
        """Scores correlate with outcomes -> monotonic + positive spread."""
        scores = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
        outcomes = np.array([0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 1.0, 1.5, 2.0])
        result = evaluate_terciles(scores, outcomes)

        assert result.is_monotonic is True
        assert result.spread > 0
        assert result.n_trades == 9

    def test_non_monotonic(self):
        """Mid > top -> not monotonic."""
        scores = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
        # Top tercile (scores 7-9) has low outcomes.
        outcomes = np.array([0.5, 0.6, 0.7, 2.0, 2.1, 2.2, 0.1, 0.2, 0.3])
        result = evaluate_terciles(scores, outcomes)

        assert result.is_monotonic is False

    def test_spread_above_threshold(self):
        """Large spread when scores predict outcomes."""
        scores = np.arange(30, dtype=float)
        outcomes = np.linspace(-0.5, 2.0, 30)
        result = evaluate_terciles(scores, outcomes)

        assert result.spread >= 1.0

    def test_spread_below_threshold(self):
        """Tiny spread when scores don't predict outcomes."""
        rng = np.random.RandomState(42)
        scores = rng.randn(30)
        outcomes = rng.randn(30) * 0.01
        result = evaluate_terciles(scores, outcomes)

        assert abs(result.spread) < 1.0

    def test_few_trades(self):
        """n < 3 -> graceful NaN result."""
        scores = np.array([1.0, 2.0])
        outcomes = np.array([0.5, 1.5])
        result = evaluate_terciles(scores, outcomes)

        assert result.n_trades == 2
        assert result.is_monotonic is False
        assert np.isnan(result.spread)

    def test_tercile_counts_sum(self):
        """All trades assigned to exactly one tercile."""
        n = 31  # not divisible by 3
        scores = np.arange(n, dtype=float)
        outcomes = np.arange(n, dtype=float)
        result = evaluate_terciles(scores, outcomes)

        assert result.top_count + result.mid_count + result.bottom_count == n

    def test_top_group_absorbs_remainder(self):
        """Top group gets extra trades when n % 3 != 0."""
        scores = np.arange(32, dtype=float)
        outcomes = np.arange(32, dtype=float)
        result = evaluate_terciles(scores, outcomes)

        # 32 // 3 = 10, so bottom=10, mid=10, top=12.
        assert result.bottom_count == 10
        assert result.mid_count == 10
        assert result.top_count == 12


# =========================================================================
# Quintile calibration
# =========================================================================

class TestQuintileCalibration:
    def test_monotonic_quintiles(self):
        """Scores perfectly predict outcomes -> monotonic."""
        scores = np.arange(50, dtype=float)
        outcomes = np.arange(50, dtype=float)
        result = evaluate_quintiles(scores, outcomes)

        assert result.is_monotonic is True
        assert len(result.bin_avg_r) == 5

    def test_inverted_quintile(self):
        """One bin inversion -> not monotonic."""
        scores = np.arange(50, dtype=float)
        outcomes = np.arange(50, dtype=float)
        # Invert bin 2 (indices 20-29).
        outcomes[20:30] = -5.0
        result = evaluate_quintiles(scores, outcomes)

        assert result.is_monotonic is False

    def test_few_trades_quintile(self):
        """n < 5 -> graceful NaN result."""
        scores = np.array([1.0, 2.0, 3.0])
        outcomes = np.array([0.5, 1.0, 1.5])
        result = evaluate_quintiles(scores, outcomes)

        assert result.is_monotonic is False
        assert all(np.isnan(r) for r in result.bin_avg_r)

    def test_quintile_counts_sum(self):
        """All trades assigned to exactly one quintile."""
        n = 53  # not divisible by 5
        scores = np.arange(n, dtype=float)
        outcomes = np.arange(n, dtype=float)
        result = evaluate_quintiles(scores, outcomes)

        assert sum(result.bin_counts) == n

    def test_flat_outcomes_is_monotonic(self):
        """Constant outcomes -> non-decreasing (ties OK)."""
        scores = np.arange(50, dtype=float)
        outcomes = np.ones(50)
        result = evaluate_quintiles(scores, outcomes)

        assert result.is_monotonic is True


# =========================================================================
# Kill criteria
# =========================================================================

class TestKillCriteria:
    def test_top_r_below_threshold(self):
        """Top avg R < 0.5 -> REJECT."""
        tercile = TercileResult(
            n_trades=30, top_avg_r=0.3, mid_avg_r=0.1, bottom_avg_r=-0.5,
            top_count=10, mid_count=10, bottom_count=10,
            spread=0.8, is_monotonic=True,
        )
        scores = np.arange(30, dtype=float)
        outcomes = scores * 0.01
        kill = check_kill_criteria(tercile, scores, outcomes)

        assert kill.top_r_below_threshold is True
        assert kill.any_reject is True

    def test_low_correlation(self):
        """Score-outcome correlation near zero -> REJECT."""
        tercile = TercileResult(
            n_trades=30, top_avg_r=1.0, mid_avg_r=0.5, bottom_avg_r=0.0,
            top_count=10, mid_count=10, bottom_count=10,
            spread=1.0, is_monotonic=True,
        )
        # Outcomes uncorrelated with ascending scores.
        scores = np.arange(30, dtype=float)
        outcomes = np.array([(-1.0) ** i for i in range(30)])
        kill = check_kill_criteria(tercile, scores, outcomes)

        assert kill.no_predictive_power is True
        assert kill.any_reject is True

    def test_monotonicity_failure_pause(self):
        """Mid > top -> PAUSE (not REJECT)."""
        tercile = TercileResult(
            n_trades=30, top_avg_r=0.8, mid_avg_r=1.2, bottom_avg_r=0.0,
            top_count=10, mid_count=10, bottom_count=10,
            spread=0.8, is_monotonic=False,
        )
        scores = np.arange(30, dtype=float)
        outcomes = scores * 0.1
        kill = check_kill_criteria(tercile, scores, outcomes)

        assert kill.monotonicity_failure is True
        assert kill.any_pause is True
        assert kill.any_reject is False

    def test_all_clear(self):
        """No kill criteria triggered."""
        tercile = TercileResult(
            n_trades=30, top_avg_r=1.5, mid_avg_r=0.8, bottom_avg_r=-0.2,
            top_count=10, mid_count=10, bottom_count=10,
            spread=1.7, is_monotonic=True,
        )
        scores = np.arange(30, dtype=float)
        outcomes = scores * 0.1
        kill = check_kill_criteria(tercile, scores, outcomes)

        assert kill.any_reject is False
        assert kill.any_pause is False
        assert len(kill.details) == 0

    def test_zero_variance_scores(self):
        """Constant scores -> correlation undefined -> kill triggers."""
        tercile = TercileResult(
            n_trades=10, top_avg_r=1.0, mid_avg_r=0.5, bottom_avg_r=0.0,
            top_count=4, mid_count=3, bottom_count=3,
            spread=1.0, is_monotonic=True,
        )
        scores = np.ones(10)
        outcomes = np.arange(10, dtype=float)
        kill = check_kill_criteria(tercile, scores, outcomes)

        # Correlation defaults to 0.0 when variance is zero.
        assert kill.no_predictive_power is True


# =========================================================================
# Gate 1
# =========================================================================

class TestGate1:
    def _passing_tercile(self) -> TercileResult:
        return TercileResult(
            n_trades=50, top_avg_r=1.5, mid_avg_r=0.8, bottom_avg_r=-0.2,
            top_count=17, mid_count=16, bottom_count=17,
            spread=1.7, is_monotonic=True,
        )

    def _passing_quintile(self) -> QuintileResult:
        return QuintileResult(
            bin_avg_r=[-0.5, 0.0, 0.5, 1.0, 1.5],
            bin_counts=[10, 10, 10, 10, 10],
            is_monotonic=True,
        )

    def _clear_kill(self) -> KillCheck:
        return KillCheck()

    def test_pass_all_items(self):
        """All 9 items pass + no kills -> PASS."""
        result = check_gate1(
            self._passing_tercile(),
            self._passing_quintile(),
            self._clear_kill(),
            feature_cap_respected=True,
            model_card_complete=True,
            negative_controls_passed=True,
            entry_degradation_pass=True,
            slippage_stress_pass=True,
        )
        assert result.overall == "PASS"

    def test_incomplete_missing_items(self):
        """Some items None -> INCOMPLETE."""
        result = check_gate1(
            self._passing_tercile(),
            self._passing_quintile(),
            self._clear_kill(),
            # External items not set -> None.
        )
        assert result.overall == "INCOMPLETE"

    def test_iterate_on_spread_failure(self):
        """Tercile spread < 1.0 -> ITERATE."""
        low_spread = TercileResult(
            n_trades=50, top_avg_r=0.5, mid_avg_r=0.3, bottom_avg_r=0.0,
            top_count=17, mid_count=16, bottom_count=17,
            spread=0.5, is_monotonic=True,
        )
        result = check_gate1(
            low_spread,
            self._passing_quintile(),
            self._clear_kill(),
            feature_cap_respected=True,
            model_card_complete=True,
            negative_controls_passed=True,
            entry_degradation_pass=True,
            slippage_stress_pass=True,
        )
        assert result.overall == "ITERATE"
        assert result.tercile_spread_pass is False

    def test_reject_on_kill(self):
        """Kill criterion REJECT -> overall REJECT."""
        kill = KillCheck(
            top_r_below_threshold=True,
            details=["REJECT: top R low"],
        )
        result = check_gate1(
            self._passing_tercile(),
            self._passing_quintile(),
            kill,
            feature_cap_respected=True,
            model_card_complete=True,
            negative_controls_passed=True,
            entry_degradation_pass=True,
            slippage_stress_pass=True,
        )
        assert result.overall == "REJECT"

    def test_pause_on_kill(self):
        """Kill criterion PAUSE -> overall PAUSE."""
        kill = KillCheck(
            monotonicity_failure=True,
            details=["PAUSE: mid > top"],
        )
        result = check_gate1(
            self._passing_tercile(),
            self._passing_quintile(),
            kill,
            feature_cap_respected=True,
            model_card_complete=True,
            negative_controls_passed=True,
            entry_degradation_pass=True,
            slippage_stress_pass=True,
        )
        assert result.overall == "PAUSE"

    def test_low_trade_count_iterates(self):
        """< 30 OOS trades -> item 1 fails -> ITERATE."""
        low_count = TercileResult(
            n_trades=15, top_avg_r=1.5, mid_avg_r=0.8, bottom_avg_r=-0.2,
            top_count=5, mid_count=5, bottom_count=5,
            spread=1.7, is_monotonic=True,
        )
        result = check_gate1(
            low_count,
            self._passing_quintile(),
            self._clear_kill(),
            feature_cap_respected=True,
            model_card_complete=True,
            negative_controls_passed=True,
            entry_degradation_pass=True,
            slippage_stress_pass=True,
        )
        assert result.overall == "ITERATE"
        assert result.oos_trade_count_pass is False

    def test_items_summary_has_nine_items(self):
        """items_summary should have exactly 9 entries."""
        result = check_gate1(
            self._passing_tercile(),
            self._passing_quintile(),
            self._clear_kill(),
        )
        assert len(result.items_summary) == 9

    def test_iterate_overrides_incomplete(self):
        """A failing item -> ITERATE even with pending items."""
        non_mono = TercileResult(
            n_trades=50, top_avg_r=0.8, mid_avg_r=1.2, bottom_avg_r=-0.2,
            top_count=17, mid_count=16, bottom_count=17,
            spread=1.0, is_monotonic=False,
        )
        result = check_gate1(
            non_mono,
            None,  # quintile not provided
            self._clear_kill(),
            # External items not set -> None.
        )
        # monotonicity fails -> ITERATE, not INCOMPLETE.
        assert result.overall == "ITERATE"

    def test_reject_overrides_iterate(self):
        """Kill REJECT overrides an ITERATE from failing items."""
        low_spread = TercileResult(
            n_trades=50, top_avg_r=0.3, mid_avg_r=0.2, bottom_avg_r=0.0,
            top_count=17, mid_count=16, bottom_count=17,
            spread=0.3, is_monotonic=True,
        )
        kill = KillCheck(top_r_below_threshold=True)
        result = check_gate1(low_spread, None, kill)

        assert result.overall == "REJECT"

    def test_no_kill_check_provided(self):
        """Gate works without a kill check (defaults to INCOMPLETE)."""
        result = check_gate1(
            self._passing_tercile(),
            self._passing_quintile(),
            kill=None,
            # External items not set.
        )
        # No kill, but external items are None -> INCOMPLETE.
        assert result.overall == "INCOMPLETE"
