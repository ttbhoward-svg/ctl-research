"""Unit tests for reconciliation engine + health gating (Data Cutover Task C)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.reconciliation import (
    BAR_COUNT_ALERT,
    BAR_COUNT_WATCH,
    CLOSE_TICK_ALERT,
    CLOSE_TICK_WATCH,
    MISSING_BAR_ALERT,
    MISSING_BAR_WATCH,
    ROLL_THRESHOLD_TICKS,
    VOLUME_PCT_ALERT,
    VOLUME_PCT_WATCH,
    ReconciliationReport,
    SymbolReconciliation,
    reconcile_multi,
    reconcile_symbol,
    save_report,
)
from ctl.universe import EQUITY_TICK, TICK_VALUES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_canonical(
    symbol: str = "/ES",
    n: int = 50,
    start: str = "2024-01-01",
    provider: str = "databento",
    close_base: float = 5000.0,
    volume: float = 100000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a minimal canonical-schema DataFrame for reconciliation tests."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range(start, periods=n, freq="B", tz="UTC")
    closes = close_base + np.cumsum(rng.randn(n) * 5)
    return pd.DataFrame({
        "timestamp": dates,
        "Open": closes - 2.0,
        "High": closes + 3.0,
        "Low": closes - 4.0,
        "Close": closes,
        "Volume": np.full(n, volume),
        "symbol": symbol,
        "timeframe": "1D",
        "provider": provider,
        "session_type": "electronic",
        "roll_method": "back_adjusted",
        "close_type": "settlement",
    })


def _perturb_close(df: pd.DataFrame, tick: float, n_ticks: float) -> pd.DataFrame:
    """Shift Close by a fixed tick amount (for testing divergence)."""
    out = df.copy()
    out["Close"] = out["Close"] + tick * n_ticks
    return out


def _inject_roll(df: pd.DataFrame, idx: int, tick: float, n_ticks: float) -> pd.DataFrame:
    """Inject a large price jump at a specific index to simulate a roll."""
    out = df.copy()
    jump = tick * n_ticks
    out.loc[idx:, "Close"] = out.loc[idx:, "Close"] + jump
    out.loc[idx:, "Open"] = out.loc[idx:, "Open"] + jump
    out.loc[idx:, "High"] = out.loc[idx:, "High"] + jump
    out.loc[idx:, "Low"] = out.loc[idx:, "Low"] + jump
    return out


# ---------------------------------------------------------------------------
# Tests: Identical data → all OK
# ---------------------------------------------------------------------------

class TestIdenticalData:
    def test_perfect_match_all_ok(self):
        primary = _make_canonical(provider="databento")
        secondary = _make_canonical(provider="norgate")
        result = reconcile_symbol(primary, secondary, "/ES")
        assert result.status == "OK"
        assert result.matched_bars == 50
        assert result.primary_bars == 50
        assert result.secondary_bars == 50

    def test_no_missing_bars(self):
        primary = _make_canonical()
        secondary = _make_canonical(provider="norgate")
        result = reconcile_symbol(primary, secondary, "/ES")
        assert result.missing_in_primary == []
        assert result.missing_in_secondary == []

    def test_no_duplicates(self):
        primary = _make_canonical()
        secondary = _make_canonical(provider="norgate")
        result = reconcile_symbol(primary, secondary, "/ES")
        assert result.duplicate_primary == 0
        assert result.duplicate_secondary == 0

    def test_close_divergence_zero(self):
        primary = _make_canonical()
        secondary = _make_canonical(provider="norgate")
        result = reconcile_symbol(primary, secondary, "/ES")
        close_check = [c for c in result.checks if c.name == "close_divergence"][0]
        assert close_check.value == 0.0
        assert close_check.status == "OK"

    def test_volume_divergence_zero(self):
        primary = _make_canonical()
        secondary = _make_canonical(provider="norgate")
        result = reconcile_symbol(primary, secondary, "/ES")
        vol_check = [c for c in result.checks if c.name == "volume_divergence"][0]
        assert vol_check.value == 0.0
        assert vol_check.status == "OK"


# ---------------------------------------------------------------------------
# Tests: Bar count parity
# ---------------------------------------------------------------------------

class TestBarCount:
    def test_small_diff_ok(self):
        primary = _make_canonical(n=50)
        secondary = _make_canonical(n=49, provider="norgate")
        result = reconcile_symbol(primary, secondary, "/ES")
        bc = [c for c in result.checks if c.name == "bar_count"][0]
        assert bc.status == "OK"
        assert bc.value == 1.0

    def test_watch_threshold(self):
        primary = _make_canonical(n=50)
        secondary = _make_canonical(n=46, provider="norgate")
        result = reconcile_symbol(primary, secondary, "/ES")
        bc = [c for c in result.checks if c.name == "bar_count"][0]
        assert bc.status == "WATCH"
        assert bc.value == 4.0

    def test_alert_threshold(self):
        primary = _make_canonical(n=50)
        secondary = _make_canonical(n=40, provider="norgate")
        result = reconcile_symbol(primary, secondary, "/ES")
        bc = [c for c in result.checks if c.name == "bar_count"][0]
        assert bc.status == "ALERT"
        assert bc.value == 10.0


# ---------------------------------------------------------------------------
# Tests: Close divergence
# ---------------------------------------------------------------------------

class TestCloseDivergence:
    def test_small_divergence_ok(self):
        tick = TICK_VALUES["/ES"]  # 12.50
        primary = _make_canonical()
        secondary = _perturb_close(
            _make_canonical(provider="norgate"), tick, 0.5
        )
        result = reconcile_symbol(primary, secondary, "/ES")
        cd = [c for c in result.checks if c.name == "close_divergence"][0]
        assert cd.status == "OK"
        assert cd.value <= CLOSE_TICK_WATCH

    def test_watch_divergence(self):
        tick = TICK_VALUES["/ES"]
        primary = _make_canonical()
        secondary = _perturb_close(
            _make_canonical(provider="norgate"), tick, 2.0
        )
        result = reconcile_symbol(primary, secondary, "/ES")
        cd = [c for c in result.checks if c.name == "close_divergence"][0]
        assert cd.status == "WATCH"

    def test_alert_divergence(self):
        tick = TICK_VALUES["/ES"]
        primary = _make_canonical()
        secondary = _perturb_close(
            _make_canonical(provider="norgate"), tick, 4.0
        )
        result = reconcile_symbol(primary, secondary, "/ES")
        cd = [c for c in result.checks if c.name == "close_divergence"][0]
        assert cd.status == "ALERT"

    def test_equity_uses_equity_tick(self):
        primary = _make_canonical(symbol="XOM", close_base=100.0, provider="databento")
        secondary = _perturb_close(
            _make_canonical(symbol="XOM", close_base=100.0, provider="norgate"),
            EQUITY_TICK, 0.5,
        )
        result = reconcile_symbol(primary, secondary, "XOM")
        cd = [c for c in result.checks if c.name == "close_divergence"][0]
        assert cd.status == "OK"


# ---------------------------------------------------------------------------
# Tests: Volume divergence
# ---------------------------------------------------------------------------

class TestVolumeDivergence:
    def test_small_volume_diff_ok(self):
        primary = _make_canonical(volume=100000.0)
        secondary = _make_canonical(volume=98000.0, provider="norgate")
        result = reconcile_symbol(primary, secondary, "/ES")
        vd = [c for c in result.checks if c.name == "volume_divergence"][0]
        assert vd.status == "OK"
        assert vd.value <= VOLUME_PCT_WATCH

    def test_watch_volume(self):
        primary = _make_canonical(volume=100000.0)
        secondary = _make_canonical(volume=88000.0, provider="norgate")
        result = reconcile_symbol(primary, secondary, "/ES")
        vd = [c for c in result.checks if c.name == "volume_divergence"][0]
        assert vd.status == "WATCH"

    def test_alert_volume(self):
        primary = _make_canonical(volume=100000.0)
        secondary = _make_canonical(volume=80000.0, provider="norgate")
        result = reconcile_symbol(primary, secondary, "/ES")
        vd = [c for c in result.checks if c.name == "volume_divergence"][0]
        assert vd.status == "ALERT"

    def test_zero_primary_volume_excluded(self):
        primary = _make_canonical(volume=0.0)
        secondary = _make_canonical(volume=50000.0, provider="norgate")
        result = reconcile_symbol(primary, secondary, "/ES")
        vd = [c for c in result.checks if c.name == "volume_divergence"][0]
        # All primary volume is 0 → excluded → OK
        assert vd.status == "OK"


# ---------------------------------------------------------------------------
# Tests: Missing bars
# ---------------------------------------------------------------------------

class TestMissingBars:
    def test_missing_in_secondary(self):
        primary = _make_canonical(n=50)
        secondary = _make_canonical(n=50, provider="norgate").iloc[5:]  # drop 5 bars
        result = reconcile_symbol(primary, secondary, "/ES")
        assert len(result.missing_in_secondary) == 5
        mb = [c for c in result.checks if c.name == "missing_secondary"][0]
        assert mb.value == 5.0
        assert mb.status == "WATCH"  # 5 is within (2, 5] so WATCH at boundary

    def test_missing_in_primary(self):
        primary = _make_canonical(n=50).iloc[3:]  # drop 3 bars
        secondary = _make_canonical(n=50, provider="norgate")
        result = reconcile_symbol(primary, secondary, "/ES")
        assert len(result.missing_in_primary) == 3
        mp = [c for c in result.checks if c.name == "missing_primary"][0]
        assert mp.value == 3.0

    def test_alert_many_missing(self):
        primary = _make_canonical(n=50)
        secondary = _make_canonical(n=50, provider="norgate").iloc[10:]
        result = reconcile_symbol(primary, secondary, "/ES")
        mb = [c for c in result.checks if c.name == "missing_secondary"][0]
        assert mb.status == "ALERT"


# ---------------------------------------------------------------------------
# Tests: Duplicate bars
# ---------------------------------------------------------------------------

class TestDuplicateBars:
    def test_duplicate_raises_alert(self):
        primary = _make_canonical(n=50)
        # Duplicate first row.
        secondary = pd.concat([
            _make_canonical(n=50, provider="norgate"),
            _make_canonical(n=50, provider="norgate").iloc[:1],
        ], ignore_index=True)
        result = reconcile_symbol(primary, secondary, "/ES")
        dup = [c for c in result.checks if c.name == "duplicates_secondary"][0]
        assert dup.status == "ALERT"
        assert result.duplicate_secondary >= 1

    def test_no_duplicates_ok(self):
        primary = _make_canonical(n=50)
        secondary = _make_canonical(n=50, provider="norgate")
        result = reconcile_symbol(primary, secondary, "/ES")
        dup_p = [c for c in result.checks if c.name == "duplicates_primary"][0]
        dup_s = [c for c in result.checks if c.name == "duplicates_secondary"][0]
        assert dup_p.status == "OK"
        assert dup_s.status == "OK"


# ---------------------------------------------------------------------------
# Tests: Roll date alignment
# ---------------------------------------------------------------------------

class TestRollAlignment:
    def test_aligned_rolls_ok(self):
        tick = TICK_VALUES["/ES"]
        primary = _make_canonical(n=50)
        secondary = _make_canonical(n=50, provider="norgate")
        # Inject same roll at same index in both.
        primary = _inject_roll(primary, 25, tick, ROLL_THRESHOLD_TICKS + 2)
        secondary = _inject_roll(secondary, 25, tick, ROLL_THRESHOLD_TICKS + 2)
        result = reconcile_symbol(primary, secondary, "/ES")
        ra = [c for c in result.checks if c.name == "roll_alignment"][0]
        assert ra.status == "OK"

    def test_misaligned_roll_flagged(self):
        tick = TICK_VALUES["/ES"]
        primary = _make_canonical(n=50)
        secondary = _make_canonical(n=50, provider="norgate")
        # Roll at different indices.
        primary = _inject_roll(primary, 25, tick, ROLL_THRESHOLD_TICKS + 2)
        secondary = _inject_roll(secondary, 30, tick, ROLL_THRESHOLD_TICKS + 2)
        result = reconcile_symbol(primary, secondary, "/ES")
        ra = [c for c in result.checks if c.name == "roll_alignment"][0]
        assert ra.status in ("WATCH", "ALERT")
        assert ra.value >= 1

    def test_non_futures_skipped(self):
        primary = _make_canonical(symbol="XLE", provider="databento")
        secondary = _make_canonical(symbol="XLE", provider="norgate")
        result = reconcile_symbol(primary, secondary, "XLE")
        ra = [c for c in result.checks if c.name == "roll_alignment"][0]
        assert ra.status == "OK"
        assert "N/A" in ra.detail


# ---------------------------------------------------------------------------
# Tests: Status hierarchy
# ---------------------------------------------------------------------------

class TestStatusHierarchy:
    def test_worst_status_propagated(self):
        primary = _make_canonical(n=50)
        # Extreme bar count mismatch → ALERT.
        secondary = _make_canonical(n=20, provider="norgate")
        result = reconcile_symbol(primary, secondary, "/ES")
        assert result.status == "ALERT"

    def test_ok_when_all_ok(self):
        primary = _make_canonical()
        secondary = _make_canonical(provider="norgate")
        result = reconcile_symbol(primary, secondary, "/ES")
        assert result.status == "OK"


# ---------------------------------------------------------------------------
# Tests: Gate behaviour (ALERT blocks downstream)
# ---------------------------------------------------------------------------

class TestGateBehaviour:
    def test_gate_open_when_all_ok(self):
        primary = _make_canonical()
        secondary = _make_canonical(provider="norgate")
        result = reconcile_symbol(primary, secondary, "/ES")
        report = ReconciliationReport.from_symbols([result])
        assert report.gate_allows_downstream is True

    def test_gate_blocked_on_alert(self):
        primary = _make_canonical(n=50)
        secondary = _make_canonical(n=20, provider="norgate")  # big mismatch
        result = reconcile_symbol(primary, secondary, "/ES")
        report = ReconciliationReport.from_symbols([result])
        assert result.status == "ALERT"
        assert report.aggregate_status == "ALERT"
        assert report.gate_allows_downstream is False

    def test_gate_open_on_watch(self):
        primary = _make_canonical(n=50)
        secondary = _make_canonical(n=46, provider="norgate")  # diff=4 → WATCH
        result = reconcile_symbol(primary, secondary, "/ES")
        report = ReconciliationReport.from_symbols([result])
        assert result.status == "WATCH"
        assert report.gate_allows_downstream is True

    def test_gate_blocked_if_any_symbol_alert(self):
        ok = reconcile_symbol(
            _make_canonical(symbol="/GC"),
            _make_canonical(symbol="/GC", provider="norgate"),
            "/GC",
        )
        alert = reconcile_symbol(
            _make_canonical(symbol="/ES", n=50),
            _make_canonical(symbol="/ES", n=20, provider="norgate"),
            "/ES",
        )
        report = ReconciliationReport.from_symbols([ok, alert])
        assert ok.status == "OK"
        assert alert.status == "ALERT"
        assert report.aggregate_status == "ALERT"
        assert report.gate_allows_downstream is False

    def test_empty_report_gate_open(self):
        report = ReconciliationReport.from_symbols([])
        assert report.gate_allows_downstream is True


# ---------------------------------------------------------------------------
# Tests: Multi-symbol reconciliation
# ---------------------------------------------------------------------------

class TestReconcileMulti:
    def test_reconcile_two_symbols(self):
        primary = {
            "/ES": _make_canonical(symbol="/ES"),
            "/GC": _make_canonical(symbol="/GC"),
        }
        secondary = {
            "/ES": _make_canonical(symbol="/ES", provider="norgate"),
            "/GC": _make_canonical(symbol="/GC", provider="norgate"),
        }
        report = reconcile_multi(primary, secondary)
        assert len(report.symbols) == 2
        assert report.aggregate_status == "OK"

    def test_missing_symbol_in_secondary(self):
        primary = {"/ES": _make_canonical(symbol="/ES")}
        secondary = {}
        report = reconcile_multi(primary, secondary)
        assert len(report.symbols) == 1
        sym = report.symbols[0]
        assert sym.secondary_bars == 0

    def test_union_of_symbols(self):
        primary = {"/ES": _make_canonical(symbol="/ES")}
        secondary = {"/GC": _make_canonical(symbol="/GC", provider="norgate")}
        report = reconcile_multi(primary, secondary)
        assert len(report.symbols) == 2
        syms = {s.symbol for s in report.symbols}
        assert syms == {"/ES", "/GC"}


# ---------------------------------------------------------------------------
# Tests: Serialisation
# ---------------------------------------------------------------------------

class TestSerialisation:
    def test_to_dict_structure(self):
        primary = _make_canonical()
        secondary = _make_canonical(provider="norgate")
        result = reconcile_symbol(primary, secondary, "/ES")
        report = ReconciliationReport.from_symbols([result])
        d = report.to_dict()
        assert "aggregate_status" in d
        assert "gate_allows_downstream" in d
        assert "symbols" in d
        assert len(d["symbols"]) == 1

    def test_to_csv_rows(self):
        primary = _make_canonical()
        secondary = _make_canonical(provider="norgate")
        result = reconcile_symbol(primary, secondary, "/ES")
        report = ReconciliationReport.from_symbols([result])
        csv_df = report.to_csv_rows()
        assert len(csv_df) == 1
        assert "symbol" in csv_df.columns
        assert "status" in csv_df.columns

    def test_summary_string(self):
        primary = _make_canonical()
        secondary = _make_canonical(provider="norgate")
        result = reconcile_symbol(primary, secondary, "/ES")
        report = ReconciliationReport.from_symbols([result])
        text = report.summary()
        assert "/ES" in text
        assert "OK" in text

    def test_save_report(self, tmp_path):
        primary = _make_canonical()
        secondary = _make_canonical(provider="norgate")
        result = reconcile_symbol(primary, secondary, "/ES")
        report = ReconciliationReport.from_symbols([result])
        paths = save_report(report, tmp_path)
        assert paths["json"].exists()
        assert paths["csv"].exists()
        with open(paths["json"]) as f:
            data = json.load(f)
        assert data["aggregate_status"] == "OK"

    def test_symbol_to_dict_roundtrip(self):
        primary = _make_canonical()
        secondary = _make_canonical(provider="norgate")
        result = reconcile_symbol(primary, secondary, "/ES")
        d = result.to_dict()
        assert d["symbol"] == "/ES"
        assert d["status"] == "OK"
        assert isinstance(d["checks"], list)
