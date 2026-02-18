"""Unit tests for cutover parity test harness (Data Cutover Task D)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.b1_detector import B1Params, compute_indicators, detect_triggers
from ctl.cutover_parity import (
    EMA_MAX_DIVERGENCE_PCT,
    R_DIFF_THRESHOLD,
    check_ema_parity,
    check_trade_parity,
    check_trigger_parity,
    run_cutover_suite,
    run_parity_suite,
    save_parity_artifacts,
)
from ctl.simulator import SimConfig, simulate_all


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_perfect_b1(n_tail: int = 30) -> pd.DataFrame:
    """Build OHLCV data guaranteed to produce exactly one B1 trigger.

    Extended with *n_tail* bars after entry so the simulator can resolve
    a trade outcome (stop or TP hit).

    Layout:
      bars 0-199:   warmup (steady uptrend, close always above EMA10)
      bars 200-209: "air" bars (low > EMA10, steep uptrend)
      bar 210:      trigger bar (low touches EMA, close near EMA)
      bar 211:      confirmation bar (close > EMA10)
      bar 212:      entry bar
      bars 213+:    continuation bars (gentle uptrend for TP hits)
    """
    total = 213 + n_tail
    dates = pd.bdate_range("2019-01-02", periods=total)

    # Warmup: steady uptrend.
    warmup_n = 200
    base = np.linspace(100, 160, warmup_n)

    # Air phase: steep rise.
    air_n = 10
    air_base = np.linspace(160, 185, air_n)

    # Trigger + confirm + entry.
    tail_base = np.array([175.0, 180.0, 181.0])

    # Continuation: gentle uptrend so TP levels get hit.
    cont_base = np.linspace(182.0, 200.0, n_tail)

    full_base = np.concatenate([base, air_base, tail_base, cont_base])

    close = full_base.copy()
    high = full_base + 3.0
    low = full_base - 1.0
    opn = full_base.copy()
    vol = np.full(total, 50000.0)

    # Air bars: lows well above EMA.
    for i in range(warmup_n, warmup_n + air_n):
        low[i] = full_base[i] + 0.5
        close[i] = full_base[i] + 1.5
        high[i] = full_base[i] + 4.0
        opn[i] = full_base[i] + 1.0

    # Trigger bar (210).
    trigger_idx = warmup_n + air_n
    low[trigger_idx] = 168.0
    close[trigger_idx] = 175.0
    high[trigger_idx] = 178.0
    opn[trigger_idx] = 176.0

    # Confirmation bar (211).
    confirm_idx = trigger_idx + 1
    close[confirm_idx] = 180.0
    high[confirm_idx] = 182.0
    low[confirm_idx] = 177.0
    opn[confirm_idx] = 178.0

    # Entry bar (212).
    entry_idx = confirm_idx + 1
    opn[entry_idx] = 181.0
    close[entry_idx] = 182.0
    high[entry_idx] = 183.0
    low[entry_idx] = 180.0

    # Continuation bars: make highs progressively higher so TPs get hit.
    for i in range(entry_idx + 1, total):
        offset = i - entry_idx
        high[i] = 183.0 + offset * 1.5
        low[i] = 180.0 + offset * 0.3
        close[i] = 182.0 + offset * 0.8
        opn[i] = close[i] - 0.5

    return pd.DataFrame({
        "Date": dates[:total],
        "Open": opn,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": vol,
    })


def _make_simple_ohlcv(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Simple uptrending OHLCV (may not produce triggers — that's OK)."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    base = 100.0 + np.arange(n, dtype=float) * 0.5
    noise = rng.randn(n) * 0.3
    closes = base + noise
    return pd.DataFrame({
        "Date": dates,
        "Open": closes - 0.2,
        "High": closes + 1.0,
        "Low": closes - 1.0,
        "Close": closes,
        "Volume": np.full(n, 50000.0),
    })


# ---------------------------------------------------------------------------
# Tests: EMA parity
# ---------------------------------------------------------------------------

class TestEmaParity:
    def test_identical_data_passes(self):
        df = _make_simple_ohlcv()
        result = check_ema_parity(df, df.copy())
        assert result.passed is True
        assert result.max_divergence_pct == 0.0

    def test_shifted_close_fails(self):
        primary = _make_simple_ohlcv()
        reference = primary.copy()
        reference["Close"] = reference["Close"] + 5.0  # big shift
        result = check_ema_parity(primary, reference)
        assert result.passed is False
        assert result.max_divergence_pct > EMA_MAX_DIVERGENCE_PCT

    def test_tiny_shift_within_threshold(self):
        primary = _make_simple_ohlcv()
        reference = primary.copy()
        # Shift close by a tiny amount (<0.01% of ~150)
        reference["Close"] = reference["Close"] + 0.001
        result = check_ema_parity(primary, reference)
        # With a 0.001 shift on ~150 base, divergence ≈ 0.0007% < 0.01%
        assert result.passed is True

    def test_n_compared_excludes_warmup(self):
        df = _make_simple_ohlcv(n=50)
        result = check_ema_parity(df, df.copy(), ema_period=10)
        # 50 bars - 9 warmup = 41 compared
        assert result.n_compared == 41

    def test_non_overlapping_dates(self):
        df1 = _make_simple_ohlcv(n=50)
        df2 = _make_simple_ohlcv(n=50)
        df2["Date"] = pd.bdate_range("2021-01-01", periods=50)
        result = check_ema_parity(df1, df2)
        # No overlap → 0 bars compared → passes vacuously
        assert result.passed is True
        assert result.n_compared == 0
        assert result.n_primary_only == 50
        assert result.n_reference_only == 50

    def test_partial_overlap(self):
        df = _make_simple_ohlcv(n=100)
        primary = df.iloc[:70].reset_index(drop=True)
        reference = df.iloc[30:].reset_index(drop=True)
        result = check_ema_parity(primary, reference)
        # 40 bars overlap, minus warmup
        assert result.n_compared > 0
        assert result.n_primary_only > 0
        assert result.n_reference_only > 0

    def test_detail_df_has_columns(self):
        df = _make_simple_ohlcv()
        result = check_ema_parity(df, df.copy())
        assert "Date" in result.detail_df.columns
        assert "ema_primary" in result.detail_df.columns
        assert "ema_reference" in result.detail_df.columns
        assert "divergence_pct" in result.detail_df.columns


# ---------------------------------------------------------------------------
# Tests: Trigger parity
# ---------------------------------------------------------------------------

class TestTriggerParity:
    def test_identical_data_passes(self):
        df = _make_perfect_b1()
        result = check_trigger_parity(df, df.copy(), "/ES")
        assert result.passed is True
        assert result.n_primary == result.n_reference
        assert result.n_matched == result.n_primary
        assert result.extra_in_primary == []
        assert result.extra_in_reference == []

    def test_produces_at_least_one_trigger(self):
        df = _make_perfect_b1()
        result = check_trigger_parity(df, df.copy(), "/ES")
        assert result.n_primary >= 1

    def test_different_data_fails(self):
        primary = _make_perfect_b1()
        reference = _make_simple_ohlcv(n=243)
        result = check_trigger_parity(primary, reference, "/ES")
        # Primary has trigger(s), reference (short flat data) has none
        assert result.passed is False
        assert len(result.extra_in_primary) > 0

    def test_identical_no_triggers_passes(self):
        # Short data: no triggers possible (< 200 warmup bars)
        df = _make_simple_ohlcv(n=50)
        result = check_trigger_parity(df, df.copy(), "/ES")
        assert result.passed is True
        assert result.n_primary == 0
        assert result.n_reference == 0

    def test_detail_df_populated(self):
        df = _make_perfect_b1()
        result = check_trigger_parity(df, df.copy(), "/ES")
        assert len(result.detail_df) > 0
        assert "trigger_date" in result.detail_df.columns
        assert "matched" in result.detail_df.columns
        assert result.detail_df["matched"].all()


# ---------------------------------------------------------------------------
# Tests: Trade outcome parity
# ---------------------------------------------------------------------------

class TestTradeParity:
    def test_identical_data_passes(self):
        df = _make_perfect_b1()
        result = check_trade_parity(df, df.copy(), "/ES")
        assert result.passed is True
        assert result.max_r_diff == 0.0

    def test_identical_data_produces_trades(self):
        df = _make_perfect_b1()
        result = check_trade_parity(df, df.copy(), "/ES")
        assert result.n_compared >= 1

    def test_modified_entry_price_fails(self):
        primary = _make_perfect_b1(n_tail=30)
        reference = primary.copy()
        # Shift entry bar Open in reference → different entry price → different R.
        # Trigger date stays the same (trigger fires at bar 210, before entry).
        entry_bar = 212
        reference.loc[entry_bar, "Open"] = reference.loc[entry_bar, "Open"] + 5.0
        result = check_trade_parity(primary, reference, "/ES")
        # Both trigger on the same date, but different entry prices → different R.
        assert result.n_compared >= 1
        assert result.max_r_diff > R_DIFF_THRESHOLD
        assert result.passed is False

    def test_no_triggers_passes_vacuously(self):
        df = _make_simple_ohlcv(n=50)
        result = check_trade_parity(df, df.copy(), "/ES")
        assert result.passed is True
        assert result.n_compared == 0

    def test_detail_df_has_columns(self):
        df = _make_perfect_b1()
        result = check_trade_parity(df, df.copy(), "/ES")
        if not result.detail_df.empty:
            assert "trigger_date" in result.detail_df.columns
            assert "r_primary" in result.detail_df.columns
            assert "r_reference" in result.detail_df.columns
            assert "r_diff" in result.detail_df.columns


# ---------------------------------------------------------------------------
# Tests: Suite runner
# ---------------------------------------------------------------------------

class TestParitySuite:
    def test_all_pass_identical_data(self):
        df = _make_perfect_b1()
        result = run_parity_suite(df, df.copy(), "/ES")
        assert result.all_passed is True
        assert result.ema.passed is True
        assert result.triggers.passed is True
        assert result.trades.passed is True

    def test_summary_dict_structure(self):
        df = _make_perfect_b1()
        result = run_parity_suite(df, df.copy(), "/ES")
        d = result.summary_dict()
        assert d["symbol"] == "/ES"
        assert d["all_passed"] is True
        assert "ema_parity" in d
        assert "trigger_parity" in d
        assert "trade_parity" in d

    def test_fail_propagates_to_all_passed(self):
        primary = _make_perfect_b1()
        reference = primary.copy()
        reference["Close"] = reference["Close"] + 10.0  # break EMA parity
        result = run_parity_suite(primary, reference, "/ES")
        assert result.ema.passed is False
        assert result.all_passed is False


# ---------------------------------------------------------------------------
# Tests: Multi-symbol suite
# ---------------------------------------------------------------------------

class TestCutoverSuite:
    def test_two_symbols_pass(self):
        df1 = _make_perfect_b1()
        df2 = _make_simple_ohlcv()
        primary = {"/ES": df1, "/CL": df2}
        reference = {"/ES": df1.copy(), "/CL": df2.copy()}
        results = run_cutover_suite(primary, reference)
        assert len(results) == 2
        assert all(r.all_passed for r in results.values())

    def test_missing_symbol_skipped(self):
        df = _make_perfect_b1()
        primary = {"/ES": df}
        reference = {"/PA": df.copy()}  # different symbol
        results = run_cutover_suite(primary, reference)
        # No intersection → no results
        assert len(results) == 0

    def test_explicit_symbol_list(self):
        df = _make_perfect_b1()
        primary = {"/ES": df, "/CL": df}
        reference = {"/ES": df.copy(), "/CL": df.copy()}
        results = run_cutover_suite(primary, reference, symbols=["/ES"])
        assert len(results) == 1
        assert "/ES" in results


# ---------------------------------------------------------------------------
# Tests: Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_creates_all_files(self, tmp_path):
        df = _make_perfect_b1()
        result = run_parity_suite(df, df.copy(), "/ES")
        paths = save_parity_artifacts(result, tmp_path)
        assert paths["ema_csv"].exists()
        assert paths["trigger_csv"].exists()
        assert paths["trade_csv"].exists()
        assert paths["summary_json"].exists()

    def test_summary_json_content(self, tmp_path):
        df = _make_perfect_b1()
        result = run_parity_suite(df, df.copy(), "/ES")
        paths = save_parity_artifacts(result, tmp_path)
        with open(paths["summary_json"]) as f:
            data = json.load(f)
        assert data["all_passed"] is True
        assert data["symbol"] == "/ES"

    def test_ema_csv_readable(self, tmp_path):
        df = _make_perfect_b1()
        result = run_parity_suite(df, df.copy(), "/ES")
        paths = save_parity_artifacts(result, tmp_path)
        csv_df = pd.read_csv(paths["ema_csv"])
        assert "ema_primary" in csv_df.columns

    def test_prefix_applied(self, tmp_path):
        df = _make_simple_ohlcv()
        result = run_parity_suite(df, df.copy(), "/PA")
        paths = save_parity_artifacts(result, tmp_path, prefix="PA")
        assert "PA_" in paths["ema_csv"].name


# ---------------------------------------------------------------------------
# Tests: Deterministic
# ---------------------------------------------------------------------------

class TestDeterministic:
    def test_same_input_same_output(self):
        df = _make_perfect_b1()
        r1 = run_parity_suite(df, df.copy(), "/ES")
        r2 = run_parity_suite(df, df.copy(), "/ES")
        d1 = r1.summary_dict()
        d2 = r2.summary_dict()
        assert d1 == d2

    def test_ema_divergence_deterministic(self):
        primary = _make_simple_ohlcv()
        reference = primary.copy()
        reference["Close"] = reference["Close"] + 2.0
        r1 = check_ema_parity(primary, reference)
        r2 = check_ema_parity(primary, reference)
        assert r1.max_divergence_pct == r2.max_divergence_pct
