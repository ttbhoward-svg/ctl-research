"""Tests for ES walk-forward drift helpers."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.es_drift_walkforward import (  # noqa: E402
    apply_walkforward_offset,
    derive_walkforward_offset,
    window_abs_drift_mean,
)


class TestDeriveWalkforwardOffset:
    def test_uses_train_window_median(self):
        can = pd.DataFrame(
            {"Date": ["2019-01-01", "2019-01-02", "2024-01-01"], "Close": [100.0, 102.0, 200.0]}
        )
        ts = pd.DataFrame(
            {"Date": ["2019-01-01", "2019-01-02", "2024-01-01"], "Close": [90.0, 91.0, 195.0]}
        )
        off = derive_walkforward_offset(
            can, ts,
            train_start="2019-01-01", train_end="2019-12-31",
            apply_start="2024-01-01", apply_end="2024-12-31",
        )
        # signed diffs in train: 10, 11 => median 10.5
        assert round(off.median_signed_diff, 6) == 10.5
        assert off.n_train_rows == 2


class TestApplyWalkforwardOffset:
    def test_applies_only_in_apply_window(self):
        can = pd.DataFrame(
            {"Date": ["2023-12-31", "2024-01-01", "2024-01-02"], "Close": [100.0, 110.0, 120.0]}
        )
        off = type(
            "O",
            (),
            {
                "apply_start": "2024-01-01",
                "apply_end": "2024-01-31",
                "median_signed_diff": 5.0,
            },
        )
        out = apply_walkforward_offset(can, off)
        assert out["Close"].tolist() == [100.0, 105.0, 115.0]


class TestWindowAbsDriftMean:
    def test_computes_overlap_window_mean(self):
        can = pd.DataFrame(
            {"Date": ["2024-01-01", "2024-01-02"], "Close": [100.0, 102.0]}
        )
        ts = pd.DataFrame(
            {"Date": ["2024-01-01", "2024-01-02"], "Close": [99.0, 105.0]}
        )
        mean_abs, n = window_abs_drift_mean(can, ts, "2024-01-01", "2024-01-02")
        # abs diffs: 1,3 => mean 2
        assert round(mean_abs, 6) == 2.0
        assert n == 2
