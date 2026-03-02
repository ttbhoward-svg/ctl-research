"""Tests for PL basis regime split analysis."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.pl_basis_regime import split_regime_stats  # noqa: E402


def _df(dates, closes):
    return pd.DataFrame({"Date": pd.to_datetime(dates), "Close": closes})


class TestSplitRegimeStats:
    def test_two_splits_compute(self):
        can = _df(
            ["2019-01-01", "2019-01-02", "2025-01-01", "2025-01-02"],
            [100, 102, 120, 119],
        )
        ts = _df(
            ["2019-01-01", "2019-01-02", "2025-01-01", "2025-01-02"],
            [99, 101, 118, 122],
        )
        splits = [
            ("pre", "2019-01-01", "2019-12-31"),
            ("post", "2025-01-01", "2025-12-31"),
        ]
        out = split_regime_stats(can, ts, splits)
        assert len(out) == 2
        assert out[0].label == "pre"
        assert out[1].label == "post"
        assert out[0].n_rows == 2
        assert out[1].n_rows == 2

    def test_empty_split_returns_zero_row(self):
        can = _df(["2019-01-01"], [100])
        ts = _df(["2019-01-01"], [99])
        splits = [("future", "2030-01-01", "2030-12-31")]
        out = split_regime_stats(can, ts, splits)
        assert len(out) == 1
        assert out[0].n_rows == 0
        assert out[0].mean_abs_diff == 0.0

    def test_pct_can_above_ts(self):
        can = _df(["2025-01-01", "2025-01-02", "2025-01-03"], [10, 9, 8])
        ts = _df(["2025-01-01", "2025-01-02", "2025-01-03"], [9, 10, 7])
        out = split_regime_stats(can, ts, [("x", "2025-01-01", "2025-12-31")])
        # signed diffs: +1, -1, +1 => 2/3 above
        assert round(out[0].pct_can_above_ts, 6) == round(2 / 3, 6)
