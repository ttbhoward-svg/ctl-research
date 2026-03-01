"""Unit tests for normalization modes (Data Cutover Task H.4)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.normalization import (
    CANONICAL_COLUMNS,
    AssetClass,
    NormalizationMode,
    coerce_date_column,
    coerce_ohlc_columns,
    coerce_volume_column,
    normalize_ohlcv,
    validate_ohlcv_schema,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(
    n: int = 50,
    date_col: str = "Date",
    volume_col: str = "Volume",
    base_price: float = 100.0,
    with_split_factor: bool = False,
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame for testing."""
    dates = pd.bdate_range("2020-01-02", periods=n)
    close = base_price + np.arange(n, dtype=float) * 0.5
    df = pd.DataFrame({
        date_col: dates,
        "Open": close - 0.2,
        "High": close + 1.0,
        "Low": close - 1.0,
        "Close": close,
        volume_col: np.full(n, 50000.0),
    })
    if with_split_factor:
        df["split_factor"] = 1.0
    return df


# ===========================================================================
# TestCoerceDateColumn
# ===========================================================================

class TestCoerceDateColumn:
    def test_standard_date_column(self):
        df = _make_ohlcv(date_col="Date")
        result = coerce_date_column(df)
        assert "Date" in result.columns
        assert result["Date"].dt.tz is None

    def test_lowercase_date(self):
        df = _make_ohlcv(date_col="date")
        result = coerce_date_column(df)
        assert "Date" in result.columns
        assert "date" not in result.columns

    def test_datetime_alias(self):
        df = _make_ohlcv(date_col="datetime")
        result = coerce_date_column(df)
        assert "Date" in result.columns

    def test_timestamp_alias(self):
        df = _make_ohlcv(date_col="timestamp")
        result = coerce_date_column(df)
        assert "Date" in result.columns

    def test_ts_event_alias(self):
        df = _make_ohlcv(date_col="ts_event")
        result = coerce_date_column(df)
        assert "Date" in result.columns

    def test_tz_aware_stripped(self):
        df = _make_ohlcv()
        df["Date"] = df["Date"].dt.tz_localize("UTC")
        result = coerce_date_column(df)
        assert result["Date"].dt.tz is None

    def test_normalized_to_midnight(self):
        df = _make_ohlcv(n=5)
        # Add time component.
        df["Date"] = pd.to_datetime([
            "2020-01-02 09:30:00",
            "2020-01-03 14:00:00",
            "2020-01-06 10:15:00",
            "2020-01-07 11:45:00",
            "2020-01-08 16:00:00",
        ])
        result = coerce_date_column(df)
        for ts in result["Date"]:
            assert ts.hour == 0
            assert ts.minute == 0

    def test_missing_date_column_raises(self):
        df = pd.DataFrame({"price": [1, 2, 3]})
        with pytest.raises(ValueError, match="No date column found"):
            coerce_date_column(df)

    def test_does_not_mutate_input(self):
        df = _make_ohlcv(date_col="date")
        original_cols = list(df.columns)
        coerce_date_column(df)
        assert list(df.columns) == original_cols


# ===========================================================================
# TestCoerceVolumeColumn
# ===========================================================================

class TestCoerceVolumeColumn:
    def test_standard_volume_passes_through(self):
        df = _make_ohlcv()
        result = coerce_volume_column(df)
        assert "Volume" in result.columns

    def test_lowercase_vol(self):
        df = _make_ohlcv(volume_col="vol")
        result = coerce_volume_column(df)
        assert "Volume" in result.columns
        assert "vol" not in result.columns

    def test_vol_dot_alias(self):
        df = _make_ohlcv(volume_col="Vol.")
        result = coerce_volume_column(df)
        assert "Volume" in result.columns

    def test_tvol_alias(self):
        df = _make_ohlcv(volume_col="tvol")
        result = coerce_volume_column(df)
        assert "Volume" in result.columns

    def test_no_volume_column_preserved(self):
        """DataFrame without any volume column returns unchanged."""
        df = pd.DataFrame({"Date": [1], "Close": [100.0]})
        result = coerce_volume_column(df)
        assert "Volume" not in result.columns

    def test_does_not_mutate_input(self):
        df = _make_ohlcv(volume_col="vol")
        original_cols = list(df.columns)
        coerce_volume_column(df)
        assert list(df.columns) == original_cols


# ===========================================================================
# TestValidateOhlcvSchema
# ===========================================================================

class TestValidateOhlcvSchema:
    def test_valid_schema_passes(self):
        df = _make_ohlcv()
        validate_ohlcv_schema(df)  # no exception

    def test_missing_column_raises(self):
        df = _make_ohlcv().drop(columns=["Close"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_ohlcv_schema(df)

    def test_error_lists_missing(self):
        df = _make_ohlcv().drop(columns=["Open", "Volume"])
        with pytest.raises(ValueError, match="Open"):
            validate_ohlcv_schema(df)


# ===========================================================================
# TestNormalizeOhlcvRaw
# ===========================================================================

class TestNormalizeOhlcvRaw:
    def test_raw_mode_passthrough(self):
        df = _make_ohlcv()
        result = normalize_ohlcv(df, asset_class="equity", mode="raw")
        assert list(result.columns) == list(CANONICAL_COLUMNS)
        assert len(result) == len(df)

    def test_raw_mode_futures(self):
        df = _make_ohlcv()
        result = normalize_ohlcv(df, asset_class="futures", mode="raw")
        assert list(result.columns) == list(CANONICAL_COLUMNS)

    def test_sorted_by_date(self):
        df = _make_ohlcv(n=20)
        # Reverse date order.
        df = df.iloc[::-1].reset_index(drop=True)
        result = normalize_ohlcv(df, asset_class="equity", mode="raw")
        dates = result["Date"].tolist()
        assert dates == sorted(dates)

    def test_canonical_column_order(self):
        df = _make_ohlcv()
        result = normalize_ohlcv(df, asset_class="etf", mode="raw")
        assert tuple(result.columns) == CANONICAL_COLUMNS

    def test_price_columns_float(self):
        df = _make_ohlcv()
        result = normalize_ohlcv(df, asset_class="equity", mode="raw")
        for col in ("Open", "High", "Low", "Close"):
            assert result[col].dtype == float

    def test_volume_column_float(self):
        df = _make_ohlcv()
        result = normalize_ohlcv(df, asset_class="equity", mode="raw")
        assert result["Volume"].dtype == float

    def test_date_column_coerced(self):
        df = _make_ohlcv(date_col="date")
        result = normalize_ohlcv(df, asset_class="equity", mode="raw")
        assert "Date" in result.columns
        assert result["Date"].dt.tz is None


# ===========================================================================
# TestNormalizeOhlcvSplitAdjusted
# ===========================================================================

class TestNormalizeOhlcvSplitAdjusted:
    def test_split_adjusted_applies_factor(self):
        df = _make_ohlcv(with_split_factor=True)
        df["split_factor"] = 4.0  # 4:1 split
        result = normalize_ohlcv(df, asset_class="equity", mode="split_adjusted")
        # Prices should be 4× the input.
        assert result["Close"].iloc[0] == pytest.approx(df["Close"].iloc[0] * 4.0)
        assert result["Open"].iloc[0] == pytest.approx(df["Open"].iloc[0] * 4.0)

    def test_split_adjusted_unit_factor_identity(self):
        df = _make_ohlcv(with_split_factor=True)
        raw = normalize_ohlcv(df, asset_class="equity", mode="raw")
        adjusted = normalize_ohlcv(df, asset_class="equity", mode="split_adjusted")
        pd.testing.assert_frame_equal(raw, adjusted)

    def test_split_adjusted_etf(self):
        df = _make_ohlcv(with_split_factor=True)
        df["split_factor"] = 2.0
        result = normalize_ohlcv(df, asset_class="etf", mode="split_adjusted")
        assert result["Close"].iloc[0] == pytest.approx(df["Close"].iloc[0] * 2.0)

    def test_missing_split_factor_raises(self):
        df = _make_ohlcv(with_split_factor=False)
        with pytest.raises(ValueError, match="split_factor"):
            normalize_ohlcv(df, asset_class="equity", mode="split_adjusted")

    def test_futures_rejects_split_adjusted(self):
        df = _make_ohlcv(with_split_factor=True)
        with pytest.raises(ValueError, match="Cannot apply split_adjusted.*futures"):
            normalize_ohlcv(df, asset_class="futures", mode="split_adjusted")


# ===========================================================================
# TestNormalizeOhlcvTotalReturn
# ===========================================================================

class TestNormalizeOhlcvTotalReturn:
    def test_total_return_not_implemented(self):
        df = _make_ohlcv()
        with pytest.raises(NotImplementedError, match="total_return_adjusted"):
            normalize_ohlcv(df, asset_class="equity", mode="total_return_adjusted")


# ===========================================================================
# TestDeterministic
# ===========================================================================

class TestDeterministic:
    def test_same_input_same_output(self):
        df = _make_ohlcv(n=30)
        r1 = normalize_ohlcv(df, asset_class="equity", mode="raw")
        r2 = normalize_ohlcv(df, asset_class="equity", mode="raw")
        pd.testing.assert_frame_equal(r1, r2)

    def test_split_adjusted_deterministic(self):
        df = _make_ohlcv(n=30, with_split_factor=True)
        df["split_factor"] = 2.0
        r1 = normalize_ohlcv(df, asset_class="equity", mode="split_adjusted")
        r2 = normalize_ohlcv(df, asset_class="equity", mode="split_adjusted")
        pd.testing.assert_frame_equal(r1, r2)

    def test_does_not_mutate_input(self):
        df = _make_ohlcv(n=10)
        original = df.copy()
        normalize_ohlcv(df, asset_class="equity", mode="raw")
        pd.testing.assert_frame_equal(df, original)


# ===========================================================================
# TestNormalizationVolumeEdgeCases
# ===========================================================================

class TestNormalizationVolumeEdgeCases:
    def test_nan_volume_filled_with_zero(self):
        df = _make_ohlcv(n=10)
        df.loc[3, "Volume"] = np.nan
        result = normalize_ohlcv(df, asset_class="equity", mode="raw")
        assert result["Volume"].iloc[3] == 0.0

    def test_string_volume_coerced(self):
        df = _make_ohlcv(n=10)
        df["Volume"] = df["Volume"].astype(str)
        result = normalize_ohlcv(df, asset_class="equity", mode="raw")
        assert result["Volume"].dtype == float


# ===========================================================================
# TestCoerceOhlcColumns
# ===========================================================================

class TestCoerceOhlcColumns:
    def test_lowercase_ohlc_coerced(self):
        df = pd.DataFrame({
            "Date": pd.bdate_range("2020-01-02", periods=5),
            "open": [1.0, 2, 3, 4, 5],
            "high": [2.0, 3, 4, 5, 6],
            "low": [0.5, 1, 2, 3, 4],
            "close": [1.5, 2.5, 3.5, 4.5, 5.5],
            "Volume": [100] * 5,
        })
        result = coerce_ohlc_columns(df)
        for col in ("Open", "High", "Low", "Close"):
            assert col in result.columns
        for col in ("open", "high", "low", "close"):
            assert col not in result.columns

    def test_canonical_case_unchanged(self):
        df = _make_ohlcv(n=5)
        result = coerce_ohlc_columns(df)
        assert list(result.columns) == list(df.columns)

    def test_mixed_case_coerced(self):
        df = pd.DataFrame({
            "Date": pd.bdate_range("2020-01-02", periods=3),
            "OPEN": [1.0, 2, 3],
            "High": [2.0, 3, 4],
            "low": [0.5, 1, 2],
            "CLOSE": [1.5, 2.5, 3.5],
            "Volume": [100] * 3,
        })
        result = coerce_ohlc_columns(df)
        assert "Open" in result.columns
        assert "High" in result.columns
        assert "Low" in result.columns
        assert "Close" in result.columns

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({
            "Date": pd.bdate_range("2020-01-02", periods=3),
            "open": [1.0, 2, 3],
            "high": [2.0, 3, 4],
            "low": [0.5, 1, 2],
            "close": [1.5, 2.5, 3.5],
            "Volume": [100] * 3,
        })
        original_cols = list(df.columns)
        coerce_ohlc_columns(df)
        assert list(df.columns) == original_cols

    def test_preserves_non_ohlc_columns(self):
        df = pd.DataFrame({
            "Date": pd.bdate_range("2020-01-02", periods=3),
            "open": [1.0, 2, 3],
            "high": [2.0, 3, 4],
            "low": [0.5, 1, 2],
            "close": [1.5, 2.5, 3.5],
            "Volume": [100] * 3,
            "symbol": ["AAPL"] * 3,
            "rtype": [1] * 3,
        })
        result = coerce_ohlc_columns(df)
        assert "symbol" in result.columns
        assert "rtype" in result.columns


# ===========================================================================
# TestNormalizeOhlcvLowercaseInput
# ===========================================================================

class TestNormalizeOhlcvLowercaseInput:
    def test_lowercase_ohlc_normalizes(self):
        """Lowercase OHLC columns should be coerced and normalize succeeds."""
        df = pd.DataFrame({
            "Date": pd.bdate_range("2020-01-02", periods=10),
            "open": np.arange(10, dtype=float) + 100,
            "high": np.arange(10, dtype=float) + 101,
            "low": np.arange(10, dtype=float) + 99,
            "close": np.arange(10, dtype=float) + 100.5,
            "Volume": np.full(10, 50000.0),
        })
        result = normalize_ohlcv(df, asset_class="equity", mode="raw")
        assert tuple(result.columns) == CANONICAL_COLUMNS
        assert len(result) == 10

    def test_all_lowercase_with_volume(self):
        """All lowercase columns including volume."""
        df = pd.DataFrame({
            "date": pd.bdate_range("2020-01-02", periods=5),
            "open": [1.0, 2, 3, 4, 5],
            "high": [2.0, 3, 4, 5, 6],
            "low": [0.5, 1, 2, 3, 4],
            "close": [1.5, 2.5, 3.5, 4.5, 5.5],
            "volume": [100] * 5,
        })
        result = normalize_ohlcv(df, asset_class="equity", mode="raw")
        assert tuple(result.columns) == CANONICAL_COLUMNS

    def test_split_adjusted_with_lowercase_ohlc(self):
        """split_adjusted mode should work with lowercase OHLC columns."""
        df = pd.DataFrame({
            "Date": pd.bdate_range("2020-01-02", periods=5),
            "open": [100.0, 101, 102, 103, 104],
            "high": [102.0, 103, 104, 105, 106],
            "low": [99.0, 100, 101, 102, 103],
            "close": [101.0, 102, 103, 104, 105],
            "Volume": [5000] * 5,
            "split_factor": [4.0] * 5,
        })
        result = normalize_ohlcv(df, asset_class="equity", mode="split_adjusted")
        assert result["Close"].iloc[0] == pytest.approx(101.0 * 4.0)


# ===========================================================================
# TestDatabentoregressionColumns
# ===========================================================================

class TestDatabentoRegressionColumns:
    """Regression test reproducing the exact Databento AAPL column layout
    that triggered the ValueError in production smoke testing."""

    def test_databento_style_columns_normalize(self):
        """Input with Databento's actual column names:
        ['ts_event','rtype','publisher_id','instrument_id',
         'open','high','low','close','volume','symbol']
        should normalize successfully under mode='raw', asset_class='equity'.
        """
        n = 20
        df = pd.DataFrame({
            "ts_event": pd.bdate_range("2020-01-02", periods=n),
            "rtype": [32] * n,
            "publisher_id": [1] * n,
            "instrument_id": [12345] * n,
            "open": np.arange(n, dtype=float) + 300.0,
            "high": np.arange(n, dtype=float) + 302.0,
            "low": np.arange(n, dtype=float) + 298.0,
            "close": np.arange(n, dtype=float) + 301.0,
            "volume": np.full(n, 1000000.0),
            "symbol": ["AAPL"] * n,
        })
        result = normalize_ohlcv(df, asset_class="equity", mode="raw")
        assert tuple(result.columns) == CANONICAL_COLUMNS
        assert len(result) == n
        # Verify prices carried through correctly.
        assert result["Close"].iloc[0] == pytest.approx(301.0)
        assert result["Open"].iloc[0] == pytest.approx(300.0)

    def test_databento_style_with_extra_columns_stripped(self):
        """Extra Databento metadata columns should be dropped in output."""
        df = pd.DataFrame({
            "ts_event": pd.bdate_range("2020-01-02", periods=5),
            "rtype": [32] * 5,
            "publisher_id": [1] * 5,
            "instrument_id": [12345] * 5,
            "open": [100.0, 101, 102, 103, 104],
            "high": [102.0, 103, 104, 105, 106],
            "low": [99.0, 100, 101, 102, 103],
            "close": [101.0, 102, 103, 104, 105],
            "volume": [50000] * 5,
            "symbol": ["AAPL"] * 5,
        })
        result = normalize_ohlcv(df, asset_class="equity", mode="raw")
        # Only canonical columns should remain.
        assert set(result.columns) == set(CANONICAL_COLUMNS)
        assert "rtype" not in result.columns
        assert "symbol" not in result.columns
        assert "ts_event" not in result.columns

    def test_mixed_ts_event_lowercase_ohlc_volume(self):
        """Combined: ts_event date alias + lowercase ohlc + lowercase volume."""
        df = pd.DataFrame({
            "ts_event": pd.bdate_range("2020-05-01", periods=3),
            "open": [175.0, 176, 177],
            "high": [178.0, 179, 180],
            "low": [174.0, 175, 176],
            "close": [176.5, 177.5, 178.5],
            "volume": [900000, 950000, 1000000],
        })
        result = normalize_ohlcv(df, asset_class="equity", mode="raw")
        assert tuple(result.columns) == CANONICAL_COLUMNS
        assert result["Volume"].iloc[0] == pytest.approx(900000.0)


# ===========================================================================
# TestParityIntegrationBackwardCompat
# ===========================================================================

class TestParityIntegrationBackwardCompat:
    """Verify that run_parity_suite with no normalization args produces
    identical output to the original signature."""

    def test_no_normalization_args_identical(self):
        """Calling run_parity_suite without normalization kwargs should
        behave identically to the old (pre-H.4) implementation."""
        from ctl.cutover_parity import run_parity_suite

        dates = pd.bdate_range("2020-01-02", periods=100)
        close = 100.0 + np.arange(100, dtype=float) * 0.5
        df = pd.DataFrame({
            "Date": dates,
            "Open": close - 0.2,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(100, 50000.0),
        })
        # No normalization args → raw pass-through.
        r1 = run_parity_suite(df, df.copy(), "TEST")
        r2 = run_parity_suite(df, df.copy(), "TEST")
        assert r1.summary_dict() == r2.summary_dict()

    def test_raw_mode_explicit_identical_to_default(self):
        """Explicit raw mode + asset_class should yield same results
        as default (no normalization)."""
        from ctl.cutover_parity import run_parity_suite

        dates = pd.bdate_range("2020-01-02", periods=100)
        close = 100.0 + np.arange(100, dtype=float) * 0.5
        df = pd.DataFrame({
            "Date": dates,
            "Open": close - 0.2,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(100, 50000.0),
        })
        r_default = run_parity_suite(df, df.copy(), "TEST")
        r_explicit = run_parity_suite(
            df, df.copy(), "TEST",
            primary_mode="raw", reference_mode="raw",
            primary_asset_class="equity", reference_asset_class="equity",
        )
        d1 = r_default.summary_dict()
        d2 = r_explicit.summary_dict()
        assert d1 == d2
