"""Unit tests for roll reconciliation (Data Cutover Task H)."""

import datetime
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.roll_reconciliation import (
    DEFAULT_DEDUP_WINDOW,
    DriftExplanationResult,
    RollComparisonResult,
    RollManifestEntry,
    RollMatch,
    StepExplanation,
    TSRollEvent,
    _worst_status,
    compare_roll_schedules,
    derive_ts_roll_events_from_spread,
    derive_ts_roll_events_from_unadjusted,
    explain_step_changes,
    load_roll_manifest,
    save_roll_manifest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manifest_entry(
    roll_date: str,
    from_contract: str = "ESH5",
    to_contract: str = "ESM5",
    from_close: float = 5000.0,
    to_close: float = 5010.0,
    gap: float = 10.0,
    cumulative_adj: float = 10.0,
    convention: str = "subtract",
) -> RollManifestEntry:
    return RollManifestEntry(
        roll_date=roll_date,
        from_contract=from_contract,
        to_contract=to_contract,
        from_close=from_close,
        to_close=to_close,
        gap=gap,
        cumulative_adj=cumulative_adj,
        convention=convention,
    )


def _make_ts_roll(
    date: str,
    gap: float = 10.0,
    close_before: float = 5000.0,
    close_after: float = 5010.0,
) -> TSRollEvent:
    return TSRollEvent(
        date=pd.Timestamp(date).date(),
        gap=gap,
        close_before=close_before,
        close_after=close_after,
    )


def _make_unadjusted_ts_df(
    dates: list,
    closes: list,
) -> pd.DataFrame:
    return pd.DataFrame({"Date": dates, "Close": closes})


def _make_adjusted_df(
    start: str,
    n_bars: int,
    base_price: float = 100.0,
    trend: float = 0.1,
) -> pd.DataFrame:
    """Create a simple adjusted series for testing."""
    dates = pd.bdate_range(start, periods=n_bars)
    prices = [base_price + i * trend for i in range(n_bars)]
    return pd.DataFrame({"Date": dates, "Close": prices})


# ===========================================================================
# Status helpers
# ===========================================================================

class TestWorstStatus:
    def test_all_pass(self):
        assert _worst_status(["PASS", "PASS"]) == "PASS"

    def test_watch_trumps_pass(self):
        assert _worst_status(["PASS", "WATCH"]) == "WATCH"

    def test_fail_trumps_all(self):
        assert _worst_status(["PASS", "WATCH", "FAIL"]) == "FAIL"

    def test_empty_is_pass(self):
        assert _worst_status([]) == "PASS"


# ===========================================================================
# RollManifestEntry
# ===========================================================================

class TestRollManifestEntry:
    def test_to_dict_round_trips_fields(self):
        entry = _make_manifest_entry("2024-03-15")
        d = entry.to_dict()
        assert d["roll_date"] == "2024-03-15"
        assert d["from_contract"] == "ESH5"
        assert d["to_contract"] == "ESM5"
        assert d["gap"] == 10.0
        assert d["convention"] == "subtract"
        assert d["session_template"] == "electronic"
        assert d["close_type"] == "settlement"

    def test_defaults(self):
        entry = _make_manifest_entry("2024-01-01")
        assert entry.trigger_reason == "volume_crossover"
        assert entry.confirmation_days == 2


# ===========================================================================
# Manifest I/O
# ===========================================================================

class TestManifestIO:
    def test_save_and_load_round_trip(self, tmp_path):
        entries = [
            _make_manifest_entry("2024-03-15"),
            _make_manifest_entry("2024-06-20", from_contract="ESM5", to_contract="ESU5"),
        ]
        path = tmp_path / "manifest.json"
        save_roll_manifest(entries, path)

        loaded = load_roll_manifest(path)
        assert len(loaded) == 2
        assert loaded[0].roll_date == "2024-03-15"
        assert loaded[1].from_contract == "ESM5"

    def test_load_with_rolls_key(self, tmp_path):
        """Manifest can also be wrapped in {"rolls": [...]}."""
        data = {"rolls": [_make_manifest_entry("2024-01-01").to_dict()]}
        path = tmp_path / "manifest.json"
        with open(path, "w") as f:
            json.dump(data, f)

        loaded = load_roll_manifest(path)
        assert len(loaded) == 1
        assert loaded[0].roll_date == "2024-01-01"

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "deep" / "manifest.json"
        save_roll_manifest([_make_manifest_entry("2024-01-01")], path)
        assert path.exists()


# ===========================================================================
# TS roll inference
# ===========================================================================

class TestDeriveRollEvents:
    def test_detects_step_change(self):
        """A large close-to-close gap should be detected as a roll."""
        df = _make_unadjusted_ts_df(
            dates=["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            closes=[100.0, 100.05, 130.0, 130.05],
        )
        # tick_size=0.25 * min_gap_ticks=100 = threshold of 25.0
        events = derive_ts_roll_events_from_unadjusted(df, tick_size=0.25, min_gap_ticks=100)
        assert len(events) == 1
        assert events[0].gap == pytest.approx(29.95, abs=0.01)

    def test_no_roll_small_changes(self):
        """Small changes below threshold should not trigger."""
        df = _make_unadjusted_ts_df(
            dates=["2024-01-02", "2024-01-03", "2024-01-04"],
            closes=[100.0, 100.1, 100.2],
        )
        events = derive_ts_roll_events_from_unadjusted(df, tick_size=0.25, min_gap_ticks=2)
        assert len(events) == 0

    def test_multiple_rolls(self):
        """Multiple step changes produce multiple events."""
        df = _make_unadjusted_ts_df(
            dates=["2024-01-02", "2024-01-03", "2024-01-04",
                   "2024-01-05", "2024-01-08"],
            closes=[100.0, 150.0, 151.0, 200.0, 201.0],
        )
        events = derive_ts_roll_events_from_unadjusted(df, tick_size=1.0, min_gap_ticks=2)
        assert len(events) == 2

    def test_negative_gap(self):
        """Downward step changes (negative gap) should also be detected."""
        df = _make_unadjusted_ts_df(
            dates=["2024-01-02", "2024-01-03", "2024-01-04"],
            closes=[100.0, 50.0, 51.0],
        )
        events = derive_ts_roll_events_from_unadjusted(df, tick_size=1.0, min_gap_ticks=2)
        assert len(events) == 1
        assert events[0].gap < 0

    def test_sorted_by_date(self):
        """Events should be returned sorted by date."""
        df = _make_unadjusted_ts_df(
            dates=["2024-01-05", "2024-01-02", "2024-01-03", "2024-01-04"],
            closes=[200.0, 100.0, 150.0, 151.0],
        )
        events = derive_ts_roll_events_from_unadjusted(df, tick_size=1.0, min_gap_ticks=2)
        if len(events) >= 2:
            assert events[0].date <= events[1].date

    def test_empty_df(self):
        """Empty DataFrame should produce no events."""
        df = pd.DataFrame(columns=["Date", "Close"])
        events = derive_ts_roll_events_from_unadjusted(df, tick_size=0.25, min_gap_ticks=2)
        assert len(events) == 0


# ===========================================================================
# Schedule comparison — exact match
# ===========================================================================

class TestCompareScheduleExactMatch:
    def test_identical_schedules_all_pass(self):
        """When canonical and TS rolls are on the same dates, all PASS."""
        can = [
            _make_manifest_entry("2024-03-15", gap=10.0),
            _make_manifest_entry("2024-06-20", gap=8.0),
        ]
        ts = [
            _make_ts_roll("2024-03-15", gap=10.0),
            _make_ts_roll("2024-06-20", gap=8.0),
        ]
        result = compare_roll_schedules(can, ts)
        assert result.status == "PASS"
        assert result.n_matched == 2
        assert result.n_fail == 0
        assert result.n_watch == 0

    def test_all_match_details(self):
        """Matched rolls should have day_delta=0."""
        can = [_make_manifest_entry("2024-03-15")]
        ts = [_make_ts_roll("2024-03-15")]
        result = compare_roll_schedules(can, ts)
        assert len(result.matches) == 1
        assert result.matches[0].day_delta == 0
        assert result.matches[0].status == "PASS"


# ===========================================================================
# Schedule comparison — 1-day shift
# ===========================================================================

class TestCompareSchedule1DayShift:
    def test_one_day_shift_is_watch(self):
        """A 1-day difference should produce WATCH status."""
        can = [_make_manifest_entry("2024-03-15")]
        ts = [_make_ts_roll("2024-03-14")]
        result = compare_roll_schedules(can, ts, max_day_delta=2)
        assert result.status == "WATCH"
        assert result.n_watch == 1
        assert result.matches[0].day_delta == 1

    def test_two_day_shift_is_watch(self):
        """A 2-day difference should also be WATCH (at edge of tolerance)."""
        can = [_make_manifest_entry("2024-03-15")]
        ts = [_make_ts_roll("2024-03-17")]
        result = compare_roll_schedules(can, ts, max_day_delta=2)
        assert result.status == "WATCH"
        assert result.matches[0].day_delta == 2

    def test_three_day_shift_is_fail(self):
        """A 3-day difference exceeds max_day_delta=2, so FAIL."""
        can = [_make_manifest_entry("2024-03-15")]
        ts = [_make_ts_roll("2024-03-18")]
        result = compare_roll_schedules(can, ts, max_day_delta=2)
        assert result.status == "FAIL"
        # Both should be unmatched.
        assert result.n_fail == 2


# ===========================================================================
# Schedule comparison — missing roll event
# ===========================================================================

class TestCompareScheduleMissing:
    def test_extra_canonical_roll(self):
        """A canonical roll with no TS match should be FAIL."""
        can = [
            _make_manifest_entry("2024-03-15"),
            _make_manifest_entry("2024-06-20"),
        ]
        ts = [_make_ts_roll("2024-03-15")]
        result = compare_roll_schedules(can, ts)
        assert result.n_fail == 1
        assert result.n_matched == 1

    def test_extra_ts_roll(self):
        """A TS roll with no canonical match should be FAIL."""
        can = [_make_manifest_entry("2024-03-15")]
        ts = [
            _make_ts_roll("2024-03-15"),
            _make_ts_roll("2024-09-20"),
        ]
        result = compare_roll_schedules(can, ts)
        assert result.n_fail == 1
        assert result.n_matched == 1

    def test_empty_ts_all_fail(self):
        """No TS rolls means all canonical rolls are FAIL."""
        can = [
            _make_manifest_entry("2024-03-15"),
            _make_manifest_entry("2024-06-20"),
        ]
        result = compare_roll_schedules(can, [])
        assert result.status == "FAIL"
        assert result.n_fail == 2

    def test_empty_canonical_all_fail(self):
        """No canonical rolls means all TS rolls are unmatched FAIL."""
        ts = [_make_ts_roll("2024-03-15")]
        result = compare_roll_schedules([], ts)
        assert result.status == "FAIL"
        assert result.n_fail == 1

    def test_both_empty_is_pass(self):
        """No rolls on either side is PASS (nothing to compare)."""
        result = compare_roll_schedules([], [])
        assert result.status == "PASS"
        assert len(result.matches) == 0


# ===========================================================================
# Schedule comparison — gap comparison
# ===========================================================================

class TestCompareScheduleGaps:
    def test_gap_diff_computed(self):
        """Gap difference should be computed for matched rolls."""
        can = [_make_manifest_entry("2024-03-15", gap=10.0)]
        ts = [_make_ts_roll("2024-03-15", gap=12.5)]
        result = compare_roll_schedules(can, ts)
        match = result.matches[0]
        assert match.gap_diff == pytest.approx(2.5, abs=0.01)

    def test_gap_diff_none_for_unmatched(self):
        """Unmatched rolls should have gap_diff=None."""
        can = [_make_manifest_entry("2024-03-15", gap=10.0)]
        result = compare_roll_schedules(can, [])
        assert result.matches[0].gap_diff is None


# ===========================================================================
# RollComparisonResult
# ===========================================================================

class TestRollComparisonResult:
    def test_to_dict(self):
        can = [_make_manifest_entry("2024-03-15")]
        ts = [_make_ts_roll("2024-03-15")]
        result = compare_roll_schedules(can, ts)
        d = result.to_dict()
        assert "status" in d
        assert "matches" in d
        assert d["n_matched"] == 1

    def test_to_dataframe(self):
        can = [_make_manifest_entry("2024-03-15")]
        ts = [_make_ts_roll("2024-03-15")]
        result = compare_roll_schedules(can, ts)
        df = result.to_dataframe()
        assert len(df) == 1
        assert "status" in df.columns
        assert "canonical_date" in df.columns

    def test_to_dataframe_empty(self):
        result = compare_roll_schedules([], [])
        df = result.to_dataframe()
        assert len(df) == 0
        assert "status" in df.columns


# ===========================================================================
# Drift explanation
# ===========================================================================

class TestExplainStepChanges:
    def test_basic_drift(self):
        """Should compute overall drift metrics."""
        can_df = _make_adjusted_df("2024-01-02", 50, base_price=100.0, trend=0.5)
        ts_df = _make_adjusted_df("2024-01-02", 50, base_price=102.0, trend=0.5)
        roll_df = pd.DataFrame(columns=[
            "canonical_date", "ts_date", "day_delta", "status",
            "canonical_gap", "ts_gap", "gap_diff", "from_contract", "to_contract",
        ])
        result = explain_step_changes(can_df, ts_df, roll_df)
        assert result.overall_mean_drift == pytest.approx(2.0, abs=0.01)
        assert result.overall_max_drift == pytest.approx(2.0, abs=0.01)

    def test_with_roll_boundaries(self):
        """Should split into intervals at roll boundaries."""
        can_df = _make_adjusted_df("2024-01-02", 50, base_price=100.0)
        ts_df = _make_adjusted_df("2024-01-02", 50, base_price=100.0)
        # Add a matched roll in the middle.
        mid_date = str(can_df.iloc[25]["Date"].date())
        roll_df = pd.DataFrame([{
            "canonical_date": mid_date,
            "ts_date": mid_date,
            "day_delta": 0,
            "status": "PASS",
            "canonical_gap": 5.0,
            "ts_gap": 5.0,
            "gap_diff": 0.0,
            "from_contract": "ESH5",
            "to_contract": "ESM5",
        }])
        result = explain_step_changes(can_df, ts_df, roll_df)
        assert result.n_intervals >= 2

    def test_empty_overlap(self):
        """Non-overlapping series should produce empty result."""
        can_df = _make_adjusted_df("2020-01-02", 10)
        ts_df = _make_adjusted_df("2025-01-02", 10)
        roll_df = pd.DataFrame()
        result = explain_step_changes(can_df, ts_df, roll_df)
        assert result.overall_mean_drift == 0.0
        assert result.n_intervals == 0

    def test_drift_contribution_sums_to_100(self):
        """Interval contributions should sum to approximately 100%."""
        can_df = _make_adjusted_df("2024-01-02", 60, base_price=100.0, trend=0.3)
        ts_df = _make_adjusted_df("2024-01-02", 60, base_price=105.0, trend=0.3)
        mid_date = str(can_df.iloc[30]["Date"].date())
        roll_df = pd.DataFrame([{
            "canonical_date": mid_date,
            "ts_date": mid_date,
            "day_delta": 0,
            "status": "PASS",
            "canonical_gap": 5.0,
            "ts_gap": 5.0,
            "gap_diff": 0.0,
            "from_contract": "ESH5",
            "to_contract": "ESM5",
        }])
        result = explain_step_changes(can_df, ts_df, roll_df)
        total_pct = sum(i.drift_contribution_pct for i in result.intervals)
        assert total_pct == pytest.approx(100.0, abs=1.0)

    def test_unexplained_drift_no_rolls(self):
        """With no rolls, all drift is in one interval."""
        can_df = _make_adjusted_df("2024-01-02", 30, base_price=100.0)
        ts_df = _make_adjusted_df("2024-01-02", 30, base_price=110.0)
        roll_df = pd.DataFrame()
        result = explain_step_changes(can_df, ts_df, roll_df)
        assert result.n_intervals == 1
        assert result.overall_mean_drift == pytest.approx(10.0, abs=0.01)


# ===========================================================================
# DriftExplanationResult serialization
# ===========================================================================

class TestDriftExplanationSerialization:
    def test_to_dict(self):
        result = DriftExplanationResult(
            symbol="ES",
            overall_mean_drift=5.0,
            overall_max_drift=10.0,
            n_intervals=1,
            intervals=[
                StepExplanation(
                    interval_start="2024-01-02",
                    interval_end="2024-03-15",
                    roll_status="PASS",
                    mean_drift=5.0,
                    max_drift=10.0,
                    drift_contribution_pct=100.0,
                )
            ],
        )
        d = result.to_dict()
        assert d["symbol"] == "ES"
        assert len(d["intervals"]) == 1
        assert d["intervals"][0]["roll_status"] == "PASS"


# ===========================================================================
# Spread-based TS roll inference (Task H.1 hotfix)
# ===========================================================================

def _make_spread_test_data(
    n_bars: int = 500,
    base_unadj: float = 5000.0,
    base_adj: float = 4900.0,
    daily_volatility: float = 20.0,
    roll_dates: list = None,
    roll_gaps: list = None,
    seed: int = 42,
):
    """Build synthetic unadjusted + adjusted TS series.

    Between rolls the spread (unadj - adj) stays constant.  On each roll
    date the spread shifts by the corresponding gap while both unadjusted
    and adjusted prices exhibit realistic daily volatility.
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2019-01-02", periods=n_bars)

    # Random walk for common price movement (affects both series equally).
    daily_moves = rng.normal(0, daily_volatility, size=n_bars)
    cumulative = np.cumsum(daily_moves)

    unadj = base_unadj + cumulative
    adj = base_adj + cumulative  # identical daily moves

    # Apply roll gap shifts: on roll date and after, unadjusted jumps but
    # adjusted stays continuous → spread changes.
    if roll_dates and roll_gaps:
        for rd_str, gap in zip(roll_dates, roll_gaps):
            rd = pd.Timestamp(rd_str)
            mask = dates >= rd
            unadj[mask] += gap

    unadj_df = pd.DataFrame({"Date": dates, "Close": unadj})
    adj_df = pd.DataFrame({"Date": dates, "Close": adj})
    return unadj_df, adj_df


class TestDeriveFromSpreadBasic:
    """Core behaviour: spread-step method detects true rolls only."""

    def test_no_rolls_in_volatile_series(self):
        """Normal volatility with no rolls → zero detections."""
        unadj, adj = _make_spread_test_data(
            n_bars=500, daily_volatility=30.0,
            roll_dates=None, roll_gaps=None,
        )
        events = derive_ts_roll_events_from_spread(
            unadj, adj, tick_size=0.25, min_gap_ticks=2,
        )
        assert len(events) == 0

    def test_detects_single_roll(self):
        """One spread step-change → one detection."""
        unadj, adj = _make_spread_test_data(
            n_bars=200, daily_volatility=20.0,
            roll_dates=["2019-05-15"], roll_gaps=[25.0],
        )
        events = derive_ts_roll_events_from_spread(
            unadj, adj, tick_size=0.25, min_gap_ticks=2,
        )
        assert len(events) == 1
        assert events[0].gap == pytest.approx(25.0, abs=0.01)

    def test_detects_multiple_rolls(self):
        """Multiple spread steps → correct count."""
        unadj, adj = _make_spread_test_data(
            n_bars=500, daily_volatility=20.0,
            roll_dates=["2019-03-15", "2019-06-20", "2019-09-18"],
            roll_gaps=[10.0, -8.0, 15.0],
        )
        events = derive_ts_roll_events_from_spread(
            unadj, adj, tick_size=0.25, min_gap_ticks=2,
        )
        assert len(events) == 3

    def test_negative_gap_detected(self):
        """Negative spread step-change should be detected."""
        unadj, adj = _make_spread_test_data(
            n_bars=200, daily_volatility=10.0,
            roll_dates=["2019-04-01"], roll_gaps=[-20.0],
        )
        events = derive_ts_roll_events_from_spread(
            unadj, adj, tick_size=0.25, min_gap_ticks=2,
        )
        assert len(events) == 1
        assert events[0].gap < 0


class TestDeriveFromSpreadPlausibility:
    """n_ts count should be plausible (near expected, not thousands)."""

    def test_es_like_quarterly_rolls(self):
        """ES-like symbol: ~8 rolls over 2 years, not thousands."""
        roll_dates = [
            "2019-03-14", "2019-06-13", "2019-09-12", "2019-12-12",
            "2020-03-12", "2020-06-11", "2020-09-10", "2020-12-10",
        ]
        roll_gaps = [12.5, -7.25, 15.0, -10.5, 8.75, -5.5, 11.0, -9.25]
        unadj, adj = _make_spread_test_data(
            n_bars=520, daily_volatility=50.0,
            roll_dates=roll_dates, roll_gaps=roll_gaps,
        )
        events = derive_ts_roll_events_from_spread(
            unadj, adj, tick_size=0.25, min_gap_ticks=2,
        )
        assert len(events) == 8
        # Old method would fire on nearly every bar.
        old_events = derive_ts_roll_events_from_unadjusted(
            unadj, tick_size=0.25, min_gap_ticks=2,
        )
        assert len(old_events) > 50  # old method overfires

    def test_cl_like_monthly_rolls(self):
        """CL-like symbol: ~20 monthly rolls, not thousands."""
        roll_dates = [f"2019-{m:02d}-15" for m in range(2, 13)] + \
                     [f"2020-{m:02d}-15" for m in range(1, 11)]
        roll_gaps = [0.5 + i * 0.1 for i in range(len(roll_dates))]
        unadj, adj = _make_spread_test_data(
            n_bars=500, daily_volatility=3.0,
            roll_dates=roll_dates, roll_gaps=roll_gaps,
        )
        events = derive_ts_roll_events_from_spread(
            unadj, adj, tick_size=0.01, min_gap_ticks=2,
        )
        assert len(events) == len(roll_dates)


class TestDeriveFromSpreadDedup:
    """Clustered step-changes within dedup_window are collapsed."""

    def test_adjacent_day_cluster_deduped(self):
        """Two spread-steps 1 day apart → only first kept."""
        dates = pd.bdate_range("2024-01-02", periods=10)
        # Spread is 0 everywhere except steps on days 4 and 5.
        spread_vals = [0, 0, 0, 0, 10, 15, 15, 15, 15, 15]
        unadj_closes = [100 + s for s in spread_vals]
        adj_closes = [100] * 10
        unadj = pd.DataFrame({"Date": dates, "Close": unadj_closes})
        adj = pd.DataFrame({"Date": dates, "Close": adj_closes})
        events = derive_ts_roll_events_from_spread(
            unadj, adj, tick_size=1.0, min_gap_ticks=2, dedup_window=3,
        )
        assert len(events) == 1  # clustered pair collapsed

    def test_distant_events_not_deduped(self):
        """Two spread-steps far apart → both kept."""
        dates = pd.bdate_range("2024-01-02", periods=30)
        spread_vals = [0] * 5 + [10] * 10 + [30] * 15
        unadj_closes = [100 + s for s in spread_vals]
        adj_closes = [100] * 30
        unadj = pd.DataFrame({"Date": dates, "Close": unadj_closes})
        adj = pd.DataFrame({"Date": dates, "Close": adj_closes})
        events = derive_ts_roll_events_from_spread(
            unadj, adj, tick_size=1.0, min_gap_ticks=2, dedup_window=3,
        )
        assert len(events) == 2

    def test_dedup_window_configurable(self):
        """Wider dedup window collapses events that shorter window wouldn't."""
        dates = pd.bdate_range("2024-01-02", periods=20)
        spread_vals = [0] * 3 + [10] * 5 + [25] * 12
        unadj_closes = [100 + s for s in spread_vals]
        adj_closes = [100] * 20
        unadj = pd.DataFrame({"Date": dates, "Close": unadj_closes})
        adj = pd.DataFrame({"Date": dates, "Close": adj_closes})

        # Window=1: each step is its own event.
        events_narrow = derive_ts_roll_events_from_spread(
            unadj, adj, tick_size=1.0, min_gap_ticks=2, dedup_window=1,
        )
        # Window=10: all within 10 days → collapsed.
        events_wide = derive_ts_roll_events_from_spread(
            unadj, adj, tick_size=1.0, min_gap_ticks=2, dedup_window=10,
        )
        assert len(events_narrow) >= len(events_wide)

    def test_empty_inputs(self):
        """Empty DataFrames → zero events."""
        unadj = pd.DataFrame(columns=["Date", "Close"])
        adj = pd.DataFrame(columns=["Date", "Close"])
        events = derive_ts_roll_events_from_spread(
            unadj, adj, tick_size=0.25, min_gap_ticks=2,
        )
        assert len(events) == 0

    def test_single_bar(self):
        """Single bar → zero events."""
        unadj = pd.DataFrame({"Date": ["2024-01-02"], "Close": [100.0]})
        adj = pd.DataFrame({"Date": ["2024-01-02"], "Close": [95.0]})
        events = derive_ts_roll_events_from_spread(
            unadj, adj, tick_size=0.25, min_gap_ticks=2,
        )
        assert len(events) == 0
