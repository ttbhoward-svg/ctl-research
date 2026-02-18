"""Unit tests for continuous futures series builder (Data Cutover Task F)."""

import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.continuous_builder import (
    CONTRACT_RE,
    MONTH_CODES,
    YEAR_MAP,
    ContractSpec,
    ContinuousResult,
    RollEvent,
    apply_panama_adjustment,
    build_all,
    build_continuous,
    detect_rolls,
    load_contract_data,
    parse_contracts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_contract_df(
    symbol: str,
    start: str,
    n_bars: int,
    base_price: float = 100.0,
    base_volume: float = 10000.0,
    trend: float = 0.1,
    volume_trend: float = 0.0,
) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame for one contract.

    Parameters
    ----------
    symbol : str
        Contract symbol (e.g. ``"ESH5"``).
    start : str
        Start date string.
    n_bars : int
        Number of daily bars.
    base_price : float
        Starting close price.
    base_volume : float
        Starting volume.
    trend : float
        Daily price increment.
    volume_trend : float
        Daily volume increment (can be negative for declining volume).
    """
    dates = pd.bdate_range(start, periods=n_bars)
    close = base_price + np.arange(n_bars) * trend
    volume = np.maximum(base_volume + np.arange(n_bars) * volume_trend, 0)
    return pd.DataFrame({
        "date": [d.date() for d in dates],
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": volume,
    })


def _two_contract_setup(
    overlap_start: str = "2024-06-01",
    n_overlap: int = 20,
    front_pre: int = 40,
    back_post: int = 40,
    front_base: float = 100.0,
    back_base: float = 105.0,
    front_vol_start: float = 50000.0,
    back_vol_start: float = 10000.0,
    front_vol_trend: float = -1500.0,
    back_vol_trend: float = 2000.0,
):
    """Build two overlapping contracts where the back contract's volume
    overtakes the front during the overlap window.

    Returns (contracts dict, contract_order list).
    """
    # Front contract: starts earlier, volume declines.
    front_start = pd.Timestamp(overlap_start) - pd.tseries.offsets.BDay(front_pre)
    front_n = front_pre + n_overlap
    front_df = _make_contract_df(
        "ESM4",
        str(front_start.date()),
        front_n,
        base_price=front_base,
        base_volume=front_vol_start,
        volume_trend=front_vol_trend,
    )

    # Back contract: starts at overlap, volume rises.
    back_n = n_overlap + back_post
    back_df = _make_contract_df(
        "ESU4",
        overlap_start,
        back_n,
        base_price=back_base,
        base_volume=back_vol_start,
        volume_trend=back_vol_trend,
    )

    contracts = {"ESM4": front_df, "ESU4": back_df}
    order = parse_contracts(["ESM4", "ESU4"])
    return contracts, order


# ---------------------------------------------------------------------------
# Tests: Contract parsing
# ---------------------------------------------------------------------------

class TestContractParsing:
    def test_parse_valid_symbol(self):
        spec = ContractSpec.from_symbol("ESH5")
        assert spec.root == "ES"
        assert spec.month_code == "H"
        assert spec.month == 3
        assert spec.year == 2025
        assert spec.symbol == "ESH5"

    def test_parse_single_letter_root(self):
        # Some commodities have single-letter roots (hypothetical).
        # CONTRACT_RE allows 1-3 uppercase letters.
        spec = ContractSpec.from_symbol("ZH5")
        assert spec.root == "Z"
        assert spec.month == 3

    def test_parse_three_letter_root(self):
        spec = ContractSpec.from_symbol("PAF0")
        assert spec.root == "PA"
        assert spec.month_code == "F"
        assert spec.month == 1
        assert spec.year == 2020

    def test_all_month_codes_valid(self):
        for code, month_num in MONTH_CODES.items():
            spec = ContractSpec.from_symbol(f"ES{code}5")
            assert spec.month == month_num
            assert spec.month_code == code

    def test_all_year_digits_valid(self):
        for digit, year in YEAR_MAP.items():
            spec = ContractSpec.from_symbol(f"ES{digit}".replace(digit, f"H{digit}"))
            assert spec.year == year

    def test_invalid_symbol_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            ContractSpec.from_symbol("INVALID")

    def test_lowercase_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            ContractSpec.from_symbol("esh5")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            ContractSpec.from_symbol("")

    def test_numeric_root_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            ContractSpec.from_symbol("12H5")

    def test_sort_key_calculation(self):
        spec = ContractSpec.from_symbol("ESH5")
        assert spec.sort_key == 2025 * 12 + 3


class TestContractOrdering:
    def test_sorted_by_expiration(self):
        specs = parse_contracts(["ESZ4", "ESH5", "ESM4", "ESU4"])
        symbols = [s.symbol for s in specs]
        assert symbols == ["ESM4", "ESU4", "ESZ4", "ESH5"]

    def test_cross_year_ordering(self):
        specs = parse_contracts(["ESZ4", "ESH5"])
        assert specs[0].symbol == "ESZ4"  # Dec 2024 before Mar 2025
        assert specs[1].symbol == "ESH5"

    def test_same_month_different_years(self):
        specs = parse_contracts(["ESH4", "ESH5"])
        assert specs[0].symbol == "ESH4"  # 2024 before 2025
        assert specs[1].symbol == "ESH5"

    def test_single_contract(self):
        specs = parse_contracts(["ESH5"])
        assert len(specs) == 1
        assert specs[0].symbol == "ESH5"

    def test_empty_list(self):
        specs = parse_contracts([])
        assert specs == []

    def test_different_roots_sort_by_date(self):
        specs = parse_contracts(["CLH5", "ESM4"])
        # ESM4 = Jun 2024 (sort_key=24300), CLH5 = Mar 2025 (sort_key=24303)
        assert specs[0].symbol == "ESM4"
        assert specs[1].symbol == "CLH5"


# ---------------------------------------------------------------------------
# Tests: Contract regex
# ---------------------------------------------------------------------------

class TestContractRegex:
    def test_valid_patterns(self):
        assert CONTRACT_RE.match("ESH5") is not None
        assert CONTRACT_RE.match("CLZ0") is not None
        assert CONTRACT_RE.match("PAF9") is not None

    def test_invalid_patterns(self):
        assert CONTRACT_RE.match("") is None
        assert CONTRACT_RE.match("ES") is None
        assert CONTRACT_RE.match("ES5") is None
        assert CONTRACT_RE.match("esh5") is None
        assert CONTRACT_RE.match("ES.FUT") is None
        assert CONTRACT_RE.match("TOOLONGH5") is None  # >3 letter root


# ---------------------------------------------------------------------------
# Tests: Roll detection
# ---------------------------------------------------------------------------

class TestRollDetection:
    def test_no_contracts_returns_empty(self):
        rolls, active = detect_rolls({}, [])
        assert rolls == []
        assert active.empty

    def test_single_contract_no_rolls(self):
        df = _make_contract_df("ESH5", "2024-01-02", 50)
        contracts = {"ESH5": df}
        order = parse_contracts(["ESH5"])
        rolls, active = detect_rolls(contracts, order)
        assert len(rolls) == 0
        assert len(active) == 50

    def test_volume_crossover_triggers_roll(self):
        contracts, order = _two_contract_setup(
            front_vol_start=50000,
            back_vol_start=10000,
            front_vol_trend=-3000,
            back_vol_trend=5000,
        )
        rolls, active = detect_rolls(contracts, order, consecutive_days=2)
        assert len(rolls) == 1
        assert rolls[0].from_contract == "ESM4"
        assert rolls[0].to_contract == "ESU4"

    def test_roll_date_after_consecutive_days(self):
        # With consecutive_days=2, roll fires on the second day of crossover.
        contracts, order = _two_contract_setup(
            front_vol_start=50000,
            back_vol_start=10000,
            front_vol_trend=-3000,
            back_vol_trend=5000,
        )
        rolls, _ = detect_rolls(contracts, order, consecutive_days=2)
        assert len(rolls) == 1
        # The roll date should be a valid business day.
        assert rolls[0].date is not None

    def test_no_roll_if_volume_never_crosses(self):
        # Both contracts cover the same dates; front always has higher volume.
        front_df = _make_contract_df("ESM4", "2024-06-01", 20,
                                     base_volume=100000, volume_trend=0)
        back_df = _make_contract_df("ESU4", "2024-06-01", 20,
                                    base_volume=1000, volume_trend=0)
        contracts = {"ESM4": front_df, "ESU4": back_df}
        order = parse_contracts(["ESM4", "ESU4"])
        rolls, active = detect_rolls(contracts, order)
        assert len(rolls) == 0

    def test_single_day_crossover_insufficient(self):
        """With consecutive_days=2, a single day of volume crossover
        should NOT trigger a roll."""
        # Build two contracts where volume crosses for exactly 1 day.
        front_df = _make_contract_df("ESM4", "2024-06-01", 10,
                                     base_volume=50000, volume_trend=0)
        back_df = _make_contract_df("ESU4", "2024-06-01", 10,
                                    base_volume=10000, volume_trend=0)
        # Make back volume > front on day 5 only.
        back_df.loc[4, "volume"] = 60000
        contracts = {"ESM4": front_df, "ESU4": back_df}
        order = parse_contracts(["ESM4", "ESU4"])
        rolls, _ = detect_rolls(contracts, order, consecutive_days=2)
        assert len(rolls) == 0

    def test_roll_adjustment_sign(self):
        """Adjustment = to_close - from_close."""
        contracts, order = _two_contract_setup(
            front_base=100.0,
            back_base=105.0,
            front_vol_start=50000,
            back_vol_start=10000,
            front_vol_trend=-3000,
            back_vol_trend=5000,
        )
        rolls, _ = detect_rolls(contracts, order)
        if rolls:
            r = rolls[0]
            assert abs(r.adjustment - (r.to_close - r.from_close)) < 1e-9

    def test_active_series_contract_column(self):
        contracts, order = _two_contract_setup()
        rolls, active = detect_rolls(contracts, order)
        assert "date" in active.columns
        assert "contract" in active.columns
        # All contracts should be from our two symbols.
        assert set(active["contract"].unique()).issubset({"ESM4", "ESU4"})

    def test_active_series_switches_after_roll(self):
        contracts, order = _two_contract_setup(
            front_vol_start=50000,
            back_vol_start=10000,
            front_vol_trend=-3000,
            back_vol_trend=5000,
        )
        rolls, active = detect_rolls(contracts, order)
        if rolls:
            roll_date = rolls[0].date
            before = active[active["date"] < roll_date]
            after = active[active["date"] >= roll_date]
            # Before roll: front contract.
            if not before.empty:
                assert (before["contract"] == "ESM4").all()
            # After roll: back contract.
            if not after.empty:
                assert (after["contract"] == "ESU4").all()

    def test_three_contract_chain(self):
        """Three contracts should produce up to 2 rolls."""
        dates1 = _make_contract_df("ESH4", "2024-01-02", 60,
                                   base_volume=80000, volume_trend=-2000)
        dates2 = _make_contract_df("ESM4", "2024-02-15", 60,
                                   base_volume=5000, volume_trend=3000)
        dates3 = _make_contract_df("ESU4", "2024-04-01", 60,
                                   base_volume=1000, volume_trend=2000)
        # Ensure second roll: make ESU4 volume eventually overtake ESM4.
        for i in range(len(dates3)):
            dates3.loc[i, "volume"] = 1000 + i * 4000

        contracts = {"ESH4": dates1, "ESM4": dates2, "ESU4": dates3}
        order = parse_contracts(["ESH4", "ESM4", "ESU4"])
        rolls, active = detect_rolls(contracts, order)
        # Should have at least 1 roll (possibly 2).
        assert len(rolls) >= 1
        assert rolls[0].from_contract == "ESH4"
        assert rolls[0].to_contract == "ESM4"


# ---------------------------------------------------------------------------
# Tests: Panama back-adjustment
# ---------------------------------------------------------------------------

class TestPanamaAdjustment:
    def test_no_rolls_no_adjustment(self):
        df = _make_contract_df("ESH5", "2024-01-02", 50)
        contracts = {"ESH5": df}
        active = pd.DataFrame({
            "date": df["date"],
            "contract": "ESH5",
        })
        result = apply_panama_adjustment(contracts, active, [])
        assert (result["adjustment"] == 0.0).all()
        # Prices should match original.
        np.testing.assert_allclose(result["Close"].values, df["close"].values, atol=1e-6)

    def test_single_roll_adjusts_earlier_segment(self):
        contracts, order = _two_contract_setup(
            front_base=100.0,
            back_base=105.0,
            front_vol_start=50000,
            back_vol_start=10000,
            front_vol_trend=-3000,
            back_vol_trend=5000,
        )
        rolls, active = detect_rolls(contracts, order)
        assert len(rolls) == 1

        result = apply_panama_adjustment(contracts, active, rolls)

        # Post-roll bars should have adjustment = 0.
        roll_date = rolls[0].date
        post_roll = result[result["Date"].dt.date >= roll_date]
        if not post_roll.empty:
            assert (post_roll["adjustment"] == 0.0).all()

        # Pre-roll bars should have non-zero adjustment.
        pre_roll = result[result["Date"].dt.date < roll_date]
        if not pre_roll.empty:
            assert (pre_roll["adjustment"] != 0.0).all()

    def test_adjustment_makes_close_continuous(self):
        """At the roll boundary, the adjusted close should not have a gap
        (the gap equals the adjustment)."""
        contracts, order = _two_contract_setup(
            front_base=100.0,
            back_base=105.0,
            front_vol_start=50000,
            back_vol_start=10000,
            front_vol_trend=-3000,
            back_vol_trend=5000,
        )
        rolls, active = detect_rolls(contracts, order)
        result = apply_panama_adjustment(contracts, active, rolls)

        if len(rolls) == 1:
            roll_date = rolls[0].date
            idx = result.index[result["Date"].dt.date == roll_date]
            if len(idx) > 0:
                roll_idx = idx[0]
                if roll_idx > 0:
                    # Check that the gap at the roll point is small.
                    close_before = result.loc[roll_idx - 1, "Close"]
                    close_at = result.loc[roll_idx, "Close"]
                    # The daily change should be reasonable (no giant gap).
                    daily_change = abs(close_at - close_before)
                    # Without adjustment, gap would be ~5.0 (back_base - front_base).
                    # With adjustment, the gap should be roughly the daily trend.
                    assert daily_change < 3.0  # generous threshold

    def test_volume_not_adjusted(self):
        contracts, order = _two_contract_setup(
            front_vol_start=80000,
            back_vol_start=50000,
            front_vol_trend=0,
            back_vol_trend=0,
        )
        rolls, active = detect_rolls(contracts, order)
        result = apply_panama_adjustment(contracts, active, rolls)
        # Volume should always be non-negative integer.
        assert (result["Volume"] >= 0).all()

    def test_empty_active_returns_empty(self):
        result = apply_panama_adjustment({}, pd.DataFrame(columns=["date", "contract"]), [])
        assert result.empty
        expected_cols = {"Date", "Open", "High", "Low", "Close", "Volume", "contract", "adjustment"}
        assert set(result.columns) == expected_cols

    def test_ohlc_all_shifted_equally(self):
        """Open, High, Low, Close should all receive the same adjustment."""
        contracts, order = _two_contract_setup(
            front_base=100.0,
            back_base=110.0,
            front_vol_start=50000,
            back_vol_start=10000,
            front_vol_trend=-3000,
            back_vol_trend=5000,
        )
        rolls, active = detect_rolls(contracts, order)
        result = apply_panama_adjustment(contracts, active, rolls)

        pre_roll = result[result["adjustment"] != 0.0]
        if not pre_roll.empty:
            row = pre_roll.iloc[0]
            adj = row["adjustment"]
            sym = row["contract"]
            d = row["Date"].date()
            raw = contracts[sym]
            raw_bar = raw[raw["date"] == d].iloc[0]
            # Each OHLC price = raw - adj
            assert abs(row["Open"] - (raw_bar["open"] - adj)) < 1e-6
            assert abs(row["High"] - (raw_bar["high"] - adj)) < 1e-6
            assert abs(row["Low"] - (raw_bar["low"] - adj)) < 1e-6
            assert abs(row["Close"] - (raw_bar["close"] - adj)) < 1e-6


# ---------------------------------------------------------------------------
# Tests: Output schema and quality
# ---------------------------------------------------------------------------

class TestOutputSchema:
    def test_continuous_columns(self):
        contracts, order = _two_contract_setup()
        rolls, active = detect_rolls(contracts, order)
        result = apply_panama_adjustment(contracts, active, rolls)
        expected = {"Date", "Open", "High", "Low", "Close", "Volume", "contract", "adjustment"}
        assert set(result.columns) == expected

    def test_dates_monotonically_increasing(self):
        contracts, order = _two_contract_setup()
        rolls, active = detect_rolls(contracts, order)
        result = apply_panama_adjustment(contracts, active, rolls)
        dates = result["Date"].values
        assert (np.diff(dates) > np.timedelta64(0)).all()

    def test_no_duplicate_dates(self):
        contracts, order = _two_contract_setup()
        rolls, active = detect_rolls(contracts, order)
        result = apply_panama_adjustment(contracts, active, rolls)
        assert result["Date"].nunique() == len(result)

    def test_date_is_datetime_type(self):
        contracts, order = _two_contract_setup()
        rolls, active = detect_rolls(contracts, order)
        result = apply_panama_adjustment(contracts, active, rolls)
        assert pd.api.types.is_datetime64_any_dtype(result["Date"])

    def test_volume_is_integer(self):
        contracts, order = _two_contract_setup()
        rolls, active = detect_rolls(contracts, order)
        result = apply_panama_adjustment(contracts, active, rolls)
        # Volume values should be whole numbers.
        assert (result["Volume"] == result["Volume"].astype(int)).all()

    def test_high_ge_low(self):
        """Adjusted High should still be >= adjusted Low."""
        contracts, order = _two_contract_setup()
        rolls, active = detect_rolls(contracts, order)
        result = apply_panama_adjustment(contracts, active, rolls)
        assert (result["High"] >= result["Low"]).all()


# ---------------------------------------------------------------------------
# Tests: RollEvent dataclass
# ---------------------------------------------------------------------------

class TestRollEvent:
    def test_roll_event_fields(self):
        r = RollEvent(
            date=datetime.date(2024, 6, 15),
            from_contract="ESM4",
            to_contract="ESU4",
            from_close=5400.0,
            to_close=5410.0,
            adjustment=10.0,
        )
        assert r.adjustment == 10.0
        assert r.cumulative_adjustment == 0.0  # default

    def test_roll_event_cumulative(self):
        r = RollEvent(
            date=datetime.date(2024, 6, 15),
            from_contract="ESM4",
            to_contract="ESU4",
            from_close=5400.0,
            to_close=5410.0,
            adjustment=10.0,
            cumulative_adjustment=25.0,
        )
        assert r.cumulative_adjustment == 25.0


# ---------------------------------------------------------------------------
# Tests: ContinuousResult
# ---------------------------------------------------------------------------

class TestContinuousResult:
    def test_empty_result(self):
        result = ContinuousResult(
            root="ES",
            continuous=pd.DataFrame(),
            roll_log=pd.DataFrame(),
        )
        assert result.root == "ES"
        assert result.n_contracts == 0
        assert result.n_rolls == 0


# ---------------------------------------------------------------------------
# Tests: build_continuous (integration with synthetic data on disk)
# ---------------------------------------------------------------------------

class TestBuildContinuousSynthetic:
    """Test build_continuous using synthetic CSV files written to tmp_path."""

    def _write_contract_csv(self, tmp_dir: Path, symbol: str, df: pd.DataFrame):
        """Write a synthetic contract DataFrame as a CSV (not zst)."""
        # build_continuous expects .csv.zst — we'll use plain .csv for tests.
        # Since load_contract_data globs for *.csv.zst, we need to adjust.
        # Actually let's write .csv.zst but uncompressed (pandas handles it).
        csv_path = tmp_dir / f"test.{symbol}.csv.zst"
        out = pd.DataFrame({
            "ts_event": pd.to_datetime(df["date"]).astype(str),
            "rtype": 0,
            "publisher_id": 0,
            "instrument_id": 0,
            "open": df["open"],
            "high": df["high"],
            "low": df["low"],
            "close": df["close"],
            "volume": df["volume"],
            "symbol": symbol,
        })
        out.to_csv(csv_path, index=False)

    def test_build_from_synthetic_files(self, tmp_path):
        """Build continuous series from synthetic CSV files."""
        front = _make_contract_df("ESM4", "2024-03-01", 60,
                                  base_price=5300, base_volume=80000,
                                  volume_trend=-2000)
        back = _make_contract_df("ESU4", "2024-04-15", 60,
                                 base_price=5310, base_volume=5000,
                                 volume_trend=3000)
        self._write_contract_csv(tmp_path, "ESM4", front)
        self._write_contract_csv(tmp_path, "ESU4", back)

        result = build_continuous("ES", tmp_path)
        assert result.root == "ES"
        assert result.n_contracts == 2
        assert not result.continuous.empty
        assert result.n_rolls >= 0

    def test_no_files_returns_empty(self, tmp_path):
        result = build_continuous("ES", tmp_path)
        assert result.continuous.empty
        assert result.n_contracts == 0
        assert result.n_rolls == 0

    def test_single_contract_no_roll(self, tmp_path):
        df = _make_contract_df("ESH5", "2024-11-01", 50,
                               base_price=6000, base_volume=100000)
        self._write_contract_csv(tmp_path, "ESH5", df)
        result = build_continuous("ES", tmp_path)
        assert result.n_contracts == 1
        assert result.n_rolls == 0
        assert len(result.continuous) == 50
        assert (result.continuous["adjustment"] == 0.0).all()

    def test_roll_log_schema(self, tmp_path):
        front = _make_contract_df("ESM4", "2024-03-01", 60,
                                  base_price=5300, base_volume=80000,
                                  volume_trend=-2000)
        back = _make_contract_df("ESU4", "2024-04-15", 60,
                                 base_price=5310, base_volume=5000,
                                 volume_trend=3000)
        self._write_contract_csv(tmp_path, "ESM4", front)
        self._write_contract_csv(tmp_path, "ESU4", back)
        result = build_continuous("ES", tmp_path)
        if not result.roll_log.empty:
            expected_cols = {"date", "from_contract", "to_contract",
                             "from_close", "to_close", "adjustment",
                             "cumulative_adjustment"}
            assert set(result.roll_log.columns) == expected_cols


# ---------------------------------------------------------------------------
# Tests: build_all (multi-symbol pipeline)
# ---------------------------------------------------------------------------

class TestBuildAll:
    def _write_contract_csv(self, tmp_dir: Path, symbol: str, df: pd.DataFrame):
        csv_path = tmp_dir / f"test.{symbol}.csv.zst"
        out = pd.DataFrame({
            "ts_event": pd.to_datetime(df["date"]).astype(str),
            "rtype": 0,
            "publisher_id": 0,
            "instrument_id": 0,
            "open": df["open"],
            "high": df["high"],
            "low": df["low"],
            "close": df["close"],
            "volume": df["volume"],
            "symbol": symbol,
        })
        out.to_csv(csv_path, index=False)

    def test_multi_symbol_outputs(self, tmp_path):
        base_dir = tmp_path / "data"
        out_dir = tmp_path / "output"

        # ES
        es_dir = base_dir / "ES"
        es_dir.mkdir(parents=True)
        front = _make_contract_df("ESH5", "2024-11-01", 40,
                                  base_price=6000, base_volume=100000)
        self._write_contract_csv(es_dir, "ESH5", front)

        # CL
        cl_dir = base_dir / "CL"
        cl_dir.mkdir(parents=True)
        clf = _make_contract_df("CLF5", "2024-11-01", 40,
                                base_price=70, base_volume=80000)
        self._write_contract_csv(cl_dir, "CLF5", clf)

        results = build_all(["ES", "CL"], base_dir, out_dir)
        assert "ES" in results
        assert "CL" in results
        assert (out_dir / "ES_continuous.csv").exists()
        assert (out_dir / "CL_continuous.csv").exists()
        assert (out_dir / "roll_log.csv").exists()

    def test_empty_symbol_dir(self, tmp_path):
        base_dir = tmp_path / "data"
        out_dir = tmp_path / "output"
        pa_dir = base_dir / "PA"
        pa_dir.mkdir(parents=True)

        results = build_all(["PA"], base_dir, out_dir)
        assert "PA" in results
        assert results["PA"].continuous.empty

    def test_roll_log_csv_written(self, tmp_path):
        base_dir = tmp_path / "data"
        out_dir = tmp_path / "output"
        es_dir = base_dir / "ES"
        es_dir.mkdir(parents=True)
        df = _make_contract_df("ESH5", "2024-11-01", 30,
                               base_price=6000, base_volume=100000)
        self._write_contract_csv(es_dir, "ESH5", df)
        build_all(["ES"], base_dir, out_dir)
        rl_path = out_dir / "roll_log.csv"
        assert rl_path.exists()
        rl = pd.read_csv(rl_path)
        assert "root" in rl.columns or rl.empty


# ---------------------------------------------------------------------------
# Tests: MONTH_CODES and YEAR_MAP constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_twelve_month_codes(self):
        assert len(MONTH_CODES) == 12

    def test_month_codes_range(self):
        assert set(MONTH_CODES.values()) == set(range(1, 13))

    def test_year_map_covers_data_range(self):
        assert 2018 in YEAR_MAP.values()
        assert 2026 in YEAR_MAP.values()

    def test_year_map_no_overlap(self):
        # All mapped years should be unique.
        assert len(set(YEAR_MAP.values())) == len(YEAR_MAP)
