"""Unit tests for external feature merge logic.

Includes leakage-guard tests that verify no future data is used.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.b1_detector import B1Trigger
from ctl.cot_loader import compute_cot_features
from ctl.external_merge import (
    _build_cot_lookup,
    _build_vix_lookup,
    lookup_cot,
    lookup_vix,
    merge_external_features,
)
from ctl.universe import Universe
from ctl.vix_loader import compute_vix_regime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trigger(
    symbol: str = "/ES",
    trigger_date: str = "2024-06-15",
) -> B1Trigger:
    """Minimal B1Trigger for merge testing."""
    return B1Trigger(
        trigger_bar_idx=100,
        trigger_date=pd.Timestamp(trigger_date),
        symbol=symbol,
        timeframe="daily",
        slope_20=10.0,
        bars_of_air=8,
        ema10_at_trigger=4500.0,
        atr14_at_trigger=50.0,
        stop_price=4450.0,
        swing_high=4550.0,
        tp1=4511.8,
        tp2=4528.6,
        tp3=4550.0,
        tp4=4611.8,
        tp5=4711.8,
    )


def _make_cot_features(
    symbol: str = "/ES",
    dates=None,
    deltas=None,
    zscores=None,
) -> pd.DataFrame:
    """Create pre-computed COT feature DataFrame."""
    if dates is None:
        dates = pd.date_range("2024-01-05", periods=60, freq="W-FRI")
    n = len(dates)
    if deltas is None:
        deltas = np.arange(n, dtype=float) * 100
    if zscores is None:
        zscores = np.linspace(-1, 1, n)
    return pd.DataFrame({
        "publication_date": dates,
        "symbol": symbol,
        "commercial_net": 50000.0,  # not used by merge, only by compute
        "cot_20d_delta": deltas,
        "cot_zscore_1y": zscores,
    })


def _make_vix_data(
    dates=None,
    closes=None,
) -> pd.DataFrame:
    """Create pre-computed VIX data with regime."""
    if dates is None:
        dates = pd.bdate_range("2024-06-01", periods=30)
    n = len(dates)
    if closes is None:
        closes = np.where(np.arange(n) % 2 == 0, 15.0, 25.0)
    df = pd.DataFrame({"date": dates, "vix_close": closes})
    return compute_vix_regime(df)


# ---------------------------------------------------------------------------
# Tests: COT lookup
# ---------------------------------------------------------------------------

class TestCOTLookup:
    def test_uses_most_recent_before_trigger(self):
        """Trigger on Wed 2024-06-12 should use Fri 2024-06-07 COT, not 2024-06-14."""
        cot = _make_cot_features(
            dates=pd.to_datetime(["2024-06-07", "2024-06-14"]),
            deltas=[100.0, 200.0],
            zscores=[0.5, 1.0],
        )
        lookup = _build_cot_lookup(cot)
        delta, zscore = lookup_cot("/ES", pd.Timestamp("2024-06-12"), lookup)
        assert delta == 100.0
        assert zscore == 0.5

    def test_same_day_publication_excluded(self):
        """Trigger on Fri 2024-06-14 should NOT use same-day COT (strict <)."""
        cot = _make_cot_features(
            dates=pd.to_datetime(["2024-06-07", "2024-06-14"]),
            deltas=[100.0, 200.0],
            zscores=[0.5, 1.0],
        )
        lookup = _build_cot_lookup(cot)
        delta, zscore = lookup_cot("/ES", pd.Timestamp("2024-06-14"), lookup)
        # Should use 2024-06-07, not 2024-06-14.
        assert delta == 100.0
        assert zscore == 0.5

    def test_no_data_before_trigger(self):
        cot = _make_cot_features(
            dates=pd.to_datetime(["2024-06-14"]),
            deltas=[100.0],
            zscores=[0.5],
        )
        lookup = _build_cot_lookup(cot)
        delta, zscore = lookup_cot("/ES", pd.Timestamp("2024-06-10"), lookup)
        assert delta is None
        assert zscore is None

    def test_unknown_symbol_returns_none(self):
        cot = _make_cot_features(symbol="/ES")
        lookup = _build_cot_lookup(cot)
        delta, zscore = lookup_cot("/CL", pd.Timestamp("2024-12-01"), lookup)
        assert delta is None
        assert zscore is None

    def test_nan_values_mapped_to_none(self):
        cot = _make_cot_features(
            dates=pd.to_datetime(["2024-06-07"]),
            deltas=[np.nan],
            zscores=[np.nan],
        )
        lookup = _build_cot_lookup(cot)
        delta, zscore = lookup_cot("/ES", pd.Timestamp("2024-06-14"), lookup)
        assert delta is None
        assert zscore is None


# ---------------------------------------------------------------------------
# Tests: VIX lookup
# ---------------------------------------------------------------------------

class TestVIXLookup:
    def test_uses_prior_day(self):
        """Trigger on Tue should use Mon VIX, not Tue."""
        vix = _make_vix_data(
            dates=pd.to_datetime(["2024-06-10", "2024-06-11"]),
            closes=[15.0, 25.0],
        )
        lookup = _build_vix_lookup(vix)
        # Trigger on 2024-06-11 (Tue) should use 2024-06-10 (Mon) = 15.0 < 20 => True
        regime = lookup_vix(pd.Timestamp("2024-06-11"), lookup)
        assert regime is True

    def test_same_day_excluded(self):
        """Trigger on Mon should NOT use Mon's own VIX close (strict <)."""
        vix = _make_vix_data(
            dates=pd.to_datetime(["2024-06-07", "2024-06-10"]),
            closes=[25.0, 15.0],
        )
        lookup = _build_vix_lookup(vix)
        # Trigger on 2024-06-10 (Mon) should use 2024-06-07 (Fri) = 25 >= 20 => False
        regime = lookup_vix(pd.Timestamp("2024-06-10"), lookup)
        assert regime is False

    def test_no_data_before_trigger(self):
        vix = _make_vix_data(
            dates=pd.to_datetime(["2024-06-10"]),
            closes=[15.0],
        )
        lookup = _build_vix_lookup(vix)
        regime = lookup_vix(pd.Timestamp("2024-06-09"), lookup)
        assert regime is None

    def test_weekend_trigger_uses_friday(self):
        """Hypothetical weekend date should look back to Friday."""
        vix = _make_vix_data(
            dates=pd.to_datetime(["2024-06-07"]),
            closes=[15.0],
        )
        lookup = _build_vix_lookup(vix)
        regime = lookup_vix(pd.Timestamp("2024-06-08"), lookup)  # Saturday
        assert regime is True


# ---------------------------------------------------------------------------
# Tests: Leakage guard
# ---------------------------------------------------------------------------

class TestLeakageGuard:
    """Tests that WOULD FAIL if future data were used."""

    def test_cot_leakage_guard(self):
        """If lookup used <= instead of <, trigger on COT pub date would see
        same-day data.  This test catches that bug."""
        # COT published on 2024-06-14 has delta=999, but trigger is also 2024-06-14.
        # Correct behavior: use prior week's data (delta=100).
        cot = _make_cot_features(
            dates=pd.to_datetime(["2024-06-07", "2024-06-14"]),
            deltas=[100.0, 999.0],
            zscores=[0.5, 9.9],
        )
        lookup = _build_cot_lookup(cot)
        delta, zscore = lookup_cot("/ES", pd.Timestamp("2024-06-14"), lookup)
        assert delta == 100.0, "Leakage! Used same-day COT publication"
        assert zscore == 0.5, "Leakage! Used same-day COT publication"

    def test_vix_leakage_guard(self):
        """If lookup used <= instead of <, trigger would see same-day VIX.
        This test catches that bug."""
        # VIX on 2024-06-11 is 15 (low), but trigger is also 2024-06-11.
        # Correct behavior: use 2024-06-10's VIX (25 = elevated).
        vix = _make_vix_data(
            dates=pd.to_datetime(["2024-06-10", "2024-06-11"]),
            closes=[25.0, 15.0],
        )
        lookup = _build_vix_lookup(vix)
        regime = lookup_vix(pd.Timestamp("2024-06-11"), lookup)
        assert regime is False, "Leakage! Used same-day VIX close"

    def test_cot_future_publication_never_used(self):
        """COT published AFTER the trigger date must never be returned."""
        cot = _make_cot_features(
            dates=pd.to_datetime(["2024-06-21"]),
            deltas=[999.0],
            zscores=[9.9],
        )
        lookup = _build_cot_lookup(cot)
        delta, zscore = lookup_cot("/ES", pd.Timestamp("2024-06-14"), lookup)
        assert delta is None, "Leakage! Used future COT publication"
        assert zscore is None, "Leakage! Used future COT publication"


# ---------------------------------------------------------------------------
# Tests: merge_external_features integration
# ---------------------------------------------------------------------------

class TestMergeIntegration:
    @pytest.fixture
    def universe(self):
        return Universe.from_yaml()

    def test_futures_get_cot_and_vix(self, universe):
        trig = _make_trigger(symbol="/ES", trigger_date="2024-07-01")
        cot = _make_cot_features(symbol="/ES")
        vix = _make_vix_data()

        result = merge_external_features([trig], cot, vix, universe)
        assert result[0].cot_20d_delta is not None
        assert result[0].cot_zscore_1y is not None
        assert result[0].vix_regime is not None

    def test_etf_gets_no_cot(self, universe):
        trig = _make_trigger(symbol="XLE", trigger_date="2024-07-01")
        cot = _make_cot_features(symbol="/ES")
        vix = _make_vix_data()

        result = merge_external_features([trig], cot, vix, universe)
        assert result[0].cot_20d_delta is None
        assert result[0].cot_zscore_1y is None
        # VIX still applies.
        assert result[0].vix_regime is not None

    def test_equity_gets_no_cot(self, universe):
        trig = _make_trigger(symbol="XOM", trigger_date="2024-07-01")
        cot = _make_cot_features(symbol="/ES")
        vix = _make_vix_data()

        result = merge_external_features([trig], cot, vix, universe)
        assert result[0].cot_20d_delta is None
        assert result[0].cot_zscore_1y is None

    def test_no_external_data_leaves_none(self, universe):
        trig = _make_trigger(symbol="/ES")
        result = merge_external_features([trig], None, None, universe)
        assert result[0].cot_20d_delta is None
        assert result[0].cot_zscore_1y is None
        assert result[0].vix_regime is None

    def test_trigger_fields_unchanged(self, universe):
        """Merge should not alter existing trigger fields."""
        trig = _make_trigger(symbol="/ES", trigger_date="2024-07-01")
        original_slope = trig.slope_20
        original_stop = trig.stop_price
        cot = _make_cot_features(symbol="/ES")
        vix = _make_vix_data()

        merge_external_features([trig], cot, vix, universe)
        assert trig.slope_20 == original_slope
        assert trig.stop_price == original_stop

    def test_b1_trigger_has_external_fields(self):
        """B1Trigger dataclass must have the three external feature fields."""
        trig = _make_trigger()
        assert hasattr(trig, "cot_20d_delta")
        assert hasattr(trig, "cot_zscore_1y")
        assert hasattr(trig, "vix_regime")
        # Defaults are None.
        assert trig.cot_20d_delta is None
        assert trig.cot_zscore_1y is None
        assert trig.vix_regime is None
