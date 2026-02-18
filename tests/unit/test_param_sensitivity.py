"""Unit tests for Phase 1a parameter sensitivity sweep."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.b1_detector import B1Params
from ctl.param_sensitivity import (
    DEFAULT_SWEEP,
    RobustnessResult,
    SweepMetrics,
    _are_neighbors,
    _infer_steps,
    _max_drawdown_r,
    _make_params,
    analyze_robustness,
    build_grid,
    identify_plateau,
    plot_1d_sensitivity,
    plot_2d_heatmap,
    run_single_combo,
    run_sweep,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 200, seed: int = 42, uptrend: bool = True) -> pd.DataFrame:
    """Generate synthetic OHLCV data with an uptrend that triggers B1.

    Creates a clear uptrend with periodic pullbacks to EMA10 so that the
    B1 detector fires triggers with various parameter settings.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    # Base price: uptrend with periodic dips.
    base = 100.0 + np.cumsum(rng.uniform(0.1, 0.5, size=n))
    if not uptrend:
        base = 200.0 - np.cumsum(rng.uniform(0.1, 0.5, size=n))

    # Add periodic pullbacks every ~15 bars.
    for i in range(n):
        if i > 30 and i % 15 == 0:
            base[i] -= rng.uniform(2.0, 5.0)

    noise = rng.uniform(-0.5, 0.5, size=n)
    close = base + noise
    high = close + rng.uniform(0.5, 2.0, size=n)
    low = close - rng.uniform(0.5, 2.0, size=n)
    open_ = close + rng.uniform(-1.0, 1.0, size=n)
    volume = rng.integers(1000, 10000, size=n).astype(float)

    return pd.DataFrame({
        "Date": dates,
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })


SMALL_SWEEP = {
    "slope_threshold": [4, 8],
    "min_bars_of_air": [3, 6],
    "entry_grace_bars": [2, 3],
    "swing_lookback_daily": [15, 20],
}


# ---------------------------------------------------------------------------
# Tests: grid generation
# ---------------------------------------------------------------------------

class TestBuildGrid:
    def test_default_grid_size(self):
        grid = build_grid()
        # 6 * 8 * 5 * 3 = 720
        assert len(grid) == 720

    def test_custom_sweep_size(self):
        grid = build_grid(SMALL_SWEEP)
        assert len(grid) == 2 * 2 * 2 * 2  # 16

    def test_grid_entries_have_all_params(self):
        grid = build_grid(SMALL_SWEEP)
        for combo in grid:
            assert set(combo.keys()) == set(SMALL_SWEEP.keys())

    def test_max_combos_cap(self):
        grid = build_grid(max_combos=10)
        assert len(grid) == 10

    def test_max_combos_larger_than_grid(self):
        grid = build_grid(SMALL_SWEEP, max_combos=100)
        assert len(grid) == 16  # not capped

    def test_grid_values_from_sweep(self):
        sweep = {"slope_threshold": [4, 8, 12], "min_bars_of_air": [3]}
        grid = build_grid(sweep)
        assert len(grid) == 3
        slopes = {c["slope_threshold"] for c in grid}
        assert slopes == {4, 8, 12}

    def test_deterministic(self):
        g1 = build_grid(SMALL_SWEEP, max_combos=5)
        g2 = build_grid(SMALL_SWEEP, max_combos=5)
        assert g1 == g2


# ---------------------------------------------------------------------------
# Tests: make_params
# ---------------------------------------------------------------------------

class TestMakeParams:
    def test_applies_combo_values(self):
        combo = {"slope_threshold": 10, "min_bars_of_air": 4,
                 "entry_grace_bars": 2, "swing_lookback_daily": 25}
        p = _make_params(combo)
        assert p.slope_threshold == 10
        assert p.min_bars_of_air == 4
        assert p.entry_grace_bars == 2
        assert p.swing_lookback_daily == 25

    def test_preserves_non_swept_defaults(self):
        combo = {"slope_threshold": 10}
        p = _make_params(combo)
        assert p.ema_period == 10  # not swept
        assert p.atr_period == 14
        assert p.breakdown_buffer_atr == 0.5

    def test_uses_base_params(self):
        base = B1Params(breakdown_buffer_atr=0.7)
        combo = {"slope_threshold": 6}
        p = _make_params(combo, base)
        assert p.breakdown_buffer_atr == 0.7
        assert p.slope_threshold == 6


# ---------------------------------------------------------------------------
# Tests: max drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_no_drawdown(self):
        assert _max_drawdown_r([1.0, 1.0, 1.0]) == 0.0

    def test_simple_drawdown(self):
        dd = _max_drawdown_r([1.0, -2.0, 0.5])
        assert dd == pytest.approx(2.0)

    def test_empty(self):
        assert _max_drawdown_r([]) == 0.0

    def test_all_losses(self):
        # cumulative = [-1, -2, -3], peak = [-1, -1, -1], dd = [0, 1, 2]
        dd = _max_drawdown_r([-1.0, -1.0, -1.0])
        assert dd == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Tests: single combo execution
# ---------------------------------------------------------------------------

class TestRunSingleCombo:
    def test_returns_sweep_metrics(self):
        df = _make_ohlcv(200, seed=42)
        combo = {"slope_threshold": 4, "min_bars_of_air": 3,
                 "entry_grace_bars": 3, "swing_lookback_daily": 20}
        result = run_single_combo(
            combo, {"/ES": df}, ["/ES"],
        )
        assert isinstance(result, SweepMetrics)
        assert result.params == combo

    def test_no_data_returns_zero_trades(self):
        combo = {"slope_threshold": 8, "min_bars_of_air": 6,
                 "entry_grace_bars": 3, "swing_lookback_daily": 20}
        result = run_single_combo(combo, {}, ["/ES"])
        assert result.n_trades == 0
        assert result.avg_r == 0.0

    def test_tight_params_fewer_trades(self):
        """Tighter parameters should yield fewer or equal trades."""
        df = _make_ohlcv(300, seed=42)
        loose = {"slope_threshold": 4, "min_bars_of_air": 3,
                 "entry_grace_bars": 5, "swing_lookback_daily": 15}
        tight = {"slope_threshold": 14, "min_bars_of_air": 10,
                 "entry_grace_bars": 1, "swing_lookback_daily": 25}
        r_loose = run_single_combo(loose, {"/ES": df}, ["/ES"])
        r_tight = run_single_combo(tight, {"/ES": df}, ["/ES"])
        assert r_loose.n_trades >= r_tight.n_trades

    def test_downtrend_few_triggers(self):
        """Downtrend data should produce few/no B1 triggers."""
        df = _make_ohlcv(200, seed=42, uptrend=False)
        combo = {"slope_threshold": 8, "min_bars_of_air": 6,
                 "entry_grace_bars": 3, "swing_lookback_daily": 20}
        result = run_single_combo(combo, {"/ES": df}, ["/ES"])
        assert result.n_trades <= 2  # might get spurious triggers, but very few

    def test_deterministic(self):
        """Same data + same combo => same results."""
        df = _make_ohlcv(200, seed=42)
        combo = {"slope_threshold": 6, "min_bars_of_air": 4,
                 "entry_grace_bars": 3, "swing_lookback_daily": 20}
        r1 = run_single_combo(combo, {"/ES": df}, ["/ES"])
        r2 = run_single_combo(combo, {"/ES": df}, ["/ES"])
        assert r1.n_trades == r2.n_trades
        assert r1.total_r == r2.total_r
        assert r1.win_rate == r2.win_rate


# ---------------------------------------------------------------------------
# Tests: full sweep
# ---------------------------------------------------------------------------

class TestRunSweep:
    def test_output_shape(self):
        df = _make_ohlcv(200, seed=42)
        result = run_sweep(
            {"/ES": df}, ["/ES"], sweep=SMALL_SWEEP,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 16  # 2^4
        expected_cols = {"slope_threshold", "min_bars_of_air",
                         "entry_grace_bars", "swing_lookback_daily",
                         "n_trades", "avg_r", "total_r", "win_rate",
                         "max_dd_r", "mar_proxy"}
        assert expected_cols.issubset(set(result.columns))

    def test_max_combos_respected(self):
        df = _make_ohlcv(200, seed=42)
        result = run_sweep(
            {"/ES": df}, ["/ES"], sweep=SMALL_SWEEP, max_combos=5,
        )
        assert len(result) == 5

    def test_no_symbols_all_zero_trades(self):
        result = run_sweep({}, [], sweep=SMALL_SWEEP)
        assert (result["n_trades"] == 0).all()


# ---------------------------------------------------------------------------
# Tests: neighbor detection
# ---------------------------------------------------------------------------

class TestNeighborDetection:
    def test_one_step_apart(self):
        steps = {"slope_threshold": 2, "min_bars_of_air": 1}
        a = {"slope_threshold": 8, "min_bars_of_air": 6}
        b = {"slope_threshold": 10, "min_bars_of_air": 6}
        assert _are_neighbors(a, b, steps)

    def test_two_params_differ(self):
        steps = {"slope_threshold": 2, "min_bars_of_air": 1}
        a = {"slope_threshold": 8, "min_bars_of_air": 6}
        b = {"slope_threshold": 10, "min_bars_of_air": 7}
        assert not _are_neighbors(a, b, steps)

    def test_same_point(self):
        steps = {"slope_threshold": 2, "min_bars_of_air": 1}
        a = {"slope_threshold": 8, "min_bars_of_air": 6}
        assert not _are_neighbors(a, a, steps)

    def test_two_steps_apart(self):
        steps = {"slope_threshold": 2, "min_bars_of_air": 1}
        a = {"slope_threshold": 8, "min_bars_of_air": 6}
        b = {"slope_threshold": 12, "min_bars_of_air": 6}
        assert not _are_neighbors(a, b, steps)


class TestInferSteps:
    def test_default_sweep_steps(self):
        steps = _infer_steps(DEFAULT_SWEEP)
        assert steps["slope_threshold"] == 2
        assert steps["min_bars_of_air"] == 1
        assert steps["entry_grace_bars"] == 1
        assert steps["swing_lookback_daily"] == 5

    def test_single_value(self):
        steps = _infer_steps({"x": [5]})
        assert steps["x"] == 1.0  # fallback


# ---------------------------------------------------------------------------
# Tests: robustness / plateau
# ---------------------------------------------------------------------------

class TestRobustnessAnalysis:
    def _make_sweep_df(self):
        """Create a small sweep result with known structure."""
        sweep = {"x": [1, 2, 3], "y": [1, 2]}
        grid = build_grid(sweep)
        # Give combo (2, 1) a high total_r, neighbors lower.
        rows = []
        for combo in grid:
            if combo["x"] == 2 and combo["y"] == 1:
                total_r = 10.0
            elif combo["x"] == 2 and combo["y"] == 2:
                total_r = 8.0
            elif combo["x"] == 1:
                total_r = 2.0
            else:
                total_r = 3.0
            rows.append({**combo, "total_r": total_r, "avg_r": total_r / 10,
                         "n_trades": 10, "win_rate": 0.6, "max_dd_r": 1.0,
                         "mar_proxy": total_r})
        return pd.DataFrame(rows), sweep

    def test_robustness_scores_computed(self):
        df, sweep = self._make_sweep_df()
        results = analyze_robustness(df, sweep)
        assert len(results) == 6
        for r in results:
            assert isinstance(r, RobustnessResult)

    def test_peak_has_lower_robustness(self):
        """The high-value combo (2,1) should have robustness < 1.0 because
        neighbors are lower."""
        df, sweep = self._make_sweep_df()
        results = analyze_robustness(df, sweep)
        peak = [r for r in results
                if r.params["x"] == 2 and r.params["y"] == 1][0]
        # Neighbors: (1,1)=2, (3,1)=3, (2,2)=8 → mean=4.33 / 10 = 0.433
        assert peak.robustness_score < 1.0

    def test_fragility_flag(self):
        df, sweep = self._make_sweep_df()
        results = analyze_robustness(df, sweep, fragility_percentile=0.5,
                                     fragility_ratio=0.5)
        peak = [r for r in results
                if r.params["x"] == 2 and r.params["y"] == 1][0]
        assert peak.is_fragile

    def test_plateau_identification(self):
        df, sweep = self._make_sweep_df()
        results = analyze_robustness(df, sweep)
        plateau = identify_plateau(results, min_robustness=0.7)
        # At least some points should be in the plateau.
        assert isinstance(plateau, list)

    def test_neighbor_count(self):
        df, sweep = self._make_sweep_df()
        results = analyze_robustness(df, sweep)
        # Corner (1,1) has 2 neighbors: (2,1) and (1,2).
        corner = [r for r in results
                  if r.params["x"] == 1 and r.params["y"] == 1][0]
        assert corner.n_neighbors == 2
        # Edge (2,1) has 3 neighbors: (1,1), (3,1), (2,2).
        edge = [r for r in results
                if r.params["x"] == 2 and r.params["y"] == 1][0]
        assert edge.n_neighbors == 3


# ---------------------------------------------------------------------------
# Tests: output schema integrity
# ---------------------------------------------------------------------------

class TestOutputSchema:
    def test_sweep_df_columns(self):
        df = _make_ohlcv(200, seed=42)
        result = run_sweep(
            {"/ES": df}, ["/ES"], sweep=SMALL_SWEEP,
        )
        required = ["slope_threshold", "min_bars_of_air",
                     "entry_grace_bars", "swing_lookback_daily",
                     "n_trades", "avg_r", "total_r", "win_rate",
                     "max_dd_r", "mar_proxy"]
        for col in required:
            assert col in result.columns, f"Missing column: {col}"

    def test_sweep_df_types(self):
        df = _make_ohlcv(200, seed=42)
        result = run_sweep(
            {"/ES": df}, ["/ES"], sweep=SMALL_SWEEP,
        )
        assert result["n_trades"].dtype in [np.int64, np.int32, int]
        assert result["avg_r"].dtype == np.float64
        assert result["total_r"].dtype == np.float64


# ---------------------------------------------------------------------------
# Tests: visualization (smoke tests)
# ---------------------------------------------------------------------------

class TestVisualization:
    def test_1d_plot_saves(self):
        df = _make_ohlcv(200, seed=42)
        result = run_sweep(
            {"/ES": df}, ["/ES"], sweep=SMALL_SWEEP,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = plot_1d_sensitivity(
                result, "slope_threshold", "avg_r", Path(tmp),
            )
            assert path is not None
            assert path.exists()
            assert path.suffix == ".png"

    def test_2d_heatmap_saves(self):
        df = _make_ohlcv(200, seed=42)
        result = run_sweep(
            {"/ES": df}, ["/ES"], sweep=SMALL_SWEEP,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = plot_2d_heatmap(
                result, "slope_threshold", "min_bars_of_air", "avg_r",
                Path(tmp),
            )
            assert path is not None
            assert path.exists()
            assert path.suffix == ".png"

    def test_1d_no_save_returns_none(self):
        df = _make_ohlcv(200, seed=42)
        result = run_sweep(
            {"/ES": df}, ["/ES"], sweep=SMALL_SWEEP,
        )
        path = plot_1d_sensitivity(result, "slope_threshold", "avg_r")
        assert path is None
