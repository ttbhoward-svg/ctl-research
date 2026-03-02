"""Tests for late-interval PL deep-dive helpers."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.pl_late_interval import compute_align_stats, date_overlap_breakdown  # noqa: E402


def _df(dates, closes):
    return pd.DataFrame({"Date": pd.to_datetime(dates), "Close": closes})


class TestComputeAlignStats:
    def test_basic_same_day(self):
        can = _df(["2026-01-01", "2026-01-02", "2026-01-03"], [100, 110, 120])
        ref = _df(["2026-01-01", "2026-01-02", "2026-01-03"], [95, 111, 119])
        s = compute_align_stats(can, ref)
        assert s.rows == 3
        assert round(s.median_signed_diff, 6) == 1.0
        assert round(s.mean_abs_diff, 6) == 2.333333

    def test_shift_can_improve_or_worsen(self):
        can = _df(["2026-01-02", "2026-01-03"], [101, 102])
        ref = _df(["2026-01-01", "2026-01-02"], [100, 101])
        s0 = compute_align_stats(can, ref, 0)
        s1 = compute_align_stats(can, ref, 1)
        assert s0.rows == 1
        assert s1.rows == 2

    def test_empty_overlap(self):
        can = _df(["2026-01-01"], [100])
        ref = _df(["2026-02-01"], [100])
        s = compute_align_stats(can, ref)
        assert s.rows == 0
        assert s.mean_abs_diff == 0.0


class TestDateOverlapBreakdown:
    def test_overlap_counts(self):
        can = _df(["2026-01-01", "2026-01-02", "2026-01-03"], [1, 2, 3])
        ref = _df(["2026-01-02", "2026-01-03", "2026-01-04"], [2, 3, 4])
        out = date_overlap_breakdown(can, ref)
        assert len(out["overlap"]) == 2
        assert len(out["can_only"]) == 1
        assert len(out["ref_only"]) == 1
