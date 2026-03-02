"""Tests for PL regime-aware correction helpers."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.pl_basis_correction import apply_regime_offsets, derive_regime_offsets  # noqa: E402


def _df(dates, closes):
    return pd.DataFrame({"Date": pd.to_datetime(dates), "Close": closes})


class TestDeriveRegimeOffsets:
    def test_median_offsets_by_regime(self):
        can = _df(["2019-01-01", "2019-01-02", "2025-01-01", "2025-01-02"], [110, 111, 95, 94])
        ts = _df(["2019-01-01", "2019-01-02", "2025-01-01", "2025-01-02"], [100, 101, 100, 101])
        regimes = [("pre", "2019-01-01", "2019-12-31"), ("post", "2025-01-01", "2025-12-31")]
        offs = derive_regime_offsets(can, ts, regimes)
        assert len(offs) == 2
        assert round(offs[0].median_signed_diff, 6) == 10.0
        assert round(offs[1].median_signed_diff, 6) == -6.0


class TestApplyRegimeOffsets:
    def test_applies_only_within_windows(self):
        can = _df(["2019-01-01", "2021-01-01", "2025-01-01"], [110, 200, 95])
        offs = [
            type("O", (), {"start": "2019-01-01", "end": "2019-12-31", "median_signed_diff": 10.0}),
            type("O", (), {"start": "2025-01-01", "end": "2025-12-31", "median_signed_diff": -5.0}),
        ]
        out = apply_regime_offsets(can, offs)
        # 2019 row: 110 - 10 = 100
        # 2021 row untouched: 200
        # 2025 row: 95 - (-5) = 100
        assert list(out["Close"].round(6)) == [100.0, 200.0, 100.0]

    def test_raises_without_close(self):
        bad = pd.DataFrame({"Date": pd.to_datetime(["2020-01-01"])})
        try:
            apply_regime_offsets(bad, [])
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "Close" in str(e)
