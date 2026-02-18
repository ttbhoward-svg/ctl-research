"""Unit tests for Phase 1a negative controls."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.dataset_assembler import SCHEMA_COLUMNS
from ctl.negative_controls import (
    BaselineMetrics,
    ControlResult,
    NegativeControlReport,
    run_lag_shift_control,
    run_negative_controls,
    run_placebo_feature_control,
    run_randomized_label_control,
)
from ctl.regression import load_pre_reg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
PRE_REG_PATH = REPO_ROOT / "configs" / "pre_registration_v1.yaml"


def _make_dataset(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic dataset with learnable signal.

    BarsOfAir and Slope_20 predict TheoreticalR, so the baseline model
    should achieve R² > 0 and positive tercile spread.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=n, freq="7D")

    bars_of_air = rng.integers(3, 15, size=n).astype(float)
    slope_20 = rng.uniform(5, 20, size=n)
    theoretical_r = 0.05 * bars_of_air + 0.02 * slope_20 + rng.normal(0, 0.5, size=n)

    clusters = ["IDX_FUT", "METALS_FUT", "ENERGY_FUT", "ETF_SECTOR"]
    cluster_col = [clusters[i % len(clusters)] for i in range(n)]
    sym_map = {"IDX_FUT": "/ES", "METALS_FUT": "/GC",
               "ENERGY_FUT": "/CL", "ETF_SECTOR": "XLE"}
    tickers = [sym_map[c] for c in cluster_col]

    df = pd.DataFrame({
        "Date": dates,
        "Ticker": tickers,
        "Timeframe": "daily",
        "SetupType": "B1",
        "EntryPrice": rng.uniform(100, 200, size=n),
        "StopPrice": rng.uniform(90, 100, size=n),
        "TP1": rng.uniform(110, 120, size=n),
        "TP2": rng.uniform(120, 130, size=n),
        "TP3": rng.uniform(130, 140, size=n),
        "TP4": rng.uniform(140, 150, size=n),
        "TP5": rng.uniform(150, 160, size=n),
        "BarsOfAir": bars_of_air,
        "Slope_20": slope_20,
        "WR_Divergence": rng.choice([True, False], size=n),
        "CleanPullback": rng.choice([True, False], size=n),
        "VolumeDeclining": rng.choice([True, False], size=n),
        "FibConfluence": rng.choice([True, False, None], size=n),
        "GapFillBelow": rng.choice([True, False], size=n),
        "MultiYearHighs": rng.choice([True, False], size=n),
        "SingleBarPullback": rng.choice([True, False], size=n),
        "WeeklyTrendAligned": rng.choice([True, False], size=n),
        "MonthlyTrendAligned": rng.choice([True, False, None], size=n),
        "COT_20D_Delta": np.where(
            np.array([t.startswith("/") for t in tickers]),
            rng.normal(0, 1000, size=n), np.nan,
        ),
        "COT_ZScore_1Y": np.where(
            np.array([t.startswith("/") for t in tickers]),
            rng.normal(0, 1, size=n), np.nan,
        ),
        "VIX_Regime": rng.choice([True, False], size=n),
        "AssetCluster": cluster_col,
        "TradableStatus": "tradable",
        "RMultiple_Actual": theoretical_r * 0.9,
        "TheoreticalR": theoretical_r,
        "MFE_R": rng.uniform(0, 2, size=n),
        "MAE_R": rng.uniform(0, 1, size=n),
        "Day1Fail": False,
        "TradeOutcome": np.where(theoretical_r > 0, "Win", "Loss"),
        "ExitDate": dates + pd.Timedelta(days=10),
        "ExitPrice": rng.uniform(100, 200, size=n),
        "ExitReason": "TP1",
        "HoldBars": rng.integers(1, 30, size=n),
        "SameBarCollision": False,
        "ExitOnLastBar": False,
        "TP1_Hit": True,
        "TP2_Hit": False,
        "TP3_Hit": False,
        "EntryDate": dates + pd.Timedelta(days=2),
        "EntryBarIdx": rng.integers(100, 300, size=n),
        "TriggerBarIdx": rng.integers(98, 298, size=n),
        "SwingHigh": rng.uniform(110, 150, size=n),
        "RiskPerUnit": rng.uniform(5, 15, size=n),
        "ScoreBucket": None,
    })
    return df[SCHEMA_COLUMNS]


def _make_baseline(r_squared: float = 0.15, spread: float = 0.8) -> BaselineMetrics:
    """Create a BaselineMetrics for testing individual controls."""
    return BaselineMetrics(
        r_squared=r_squared,
        tercile_spread=spread,
        tercile_monotonic=True,
        n_trades=120,
    )


# ---------------------------------------------------------------------------
# Tests: Randomized-label control
# ---------------------------------------------------------------------------

class TestRandomizedLabelControl:
    def test_returns_control_result(self):
        df = _make_dataset(120)
        cfg = load_pre_reg(PRE_REG_PATH)
        baseline = _make_baseline()
        result = run_randomized_label_control(
            df, cfg, baseline, n_permutations=2, base_seed=42,
        )
        assert isinstance(result, ControlResult)
        assert result.name == "randomized_labels"

    def test_shuffled_r2_near_zero(self):
        """Shuffling labels should collapse explanatory power."""
        df = _make_dataset(120)
        cfg = load_pre_reg(PRE_REG_PATH)
        baseline = _make_baseline()
        result = run_randomized_label_control(
            df, cfg, baseline, n_permutations=3, base_seed=42,
        )
        assert result.metrics["mean_r2"] < 0.10  # should be near zero

    def test_shuffled_spread_small(self):
        """Shuffled labels should produce small tercile spread."""
        df = _make_dataset(120)
        cfg = load_pre_reg(PRE_REG_PATH)
        baseline = _make_baseline()
        result = run_randomized_label_control(
            df, cfg, baseline, n_permutations=3, base_seed=42,
        )
        assert result.metrics["mean_abs_spread"] < 1.0

    def test_deterministic(self):
        """Same seed => same results."""
        df = _make_dataset(120)
        cfg = load_pre_reg(PRE_REG_PATH)
        baseline = _make_baseline()
        r1 = run_randomized_label_control(
            df, cfg, baseline, n_permutations=2, base_seed=42,
        )
        r2 = run_randomized_label_control(
            df, cfg, baseline, n_permutations=2, base_seed=42,
        )
        assert r1.metrics["mean_r2"] == r2.metrics["mean_r2"]
        assert r1.metrics["mean_abs_spread"] == r2.metrics["mean_abs_spread"]

    def test_has_required_metrics(self):
        df = _make_dataset(120)
        cfg = load_pre_reg(PRE_REG_PATH)
        baseline = _make_baseline()
        result = run_randomized_label_control(
            df, cfg, baseline, n_permutations=2, base_seed=42,
        )
        assert "mean_r2" in result.metrics
        assert "mean_abs_spread" in result.metrics
        assert "n_permutations" in result.metrics


# ---------------------------------------------------------------------------
# Tests: Lag-shift control
# ---------------------------------------------------------------------------

class TestLagShiftControl:
    def test_returns_control_result(self):
        df = _make_dataset(120)
        cfg = load_pre_reg(PRE_REG_PATH)
        baseline = _make_baseline()
        result = run_lag_shift_control(df, cfg, baseline)
        assert isinstance(result, ControlResult)
        assert result.name == "lag_shift"

    def test_lag_r2_lower_than_baseline(self):
        """Lagging features should degrade R²."""
        df = _make_dataset(120)
        cfg = load_pre_reg(PRE_REG_PATH)
        # Use actual baseline from model training.
        from ctl.regression import train_model
        model_result = train_model(df, PRE_REG_PATH)
        y = df["TheoreticalR"].values
        from ctl.ranking_gate import evaluate_terciles
        tercile = evaluate_terciles(model_result.scores, y)
        baseline = BaselineMetrics(
            r_squared=model_result.diagnostics.r_squared_is,
            tercile_spread=tercile.spread,
            tercile_monotonic=tercile.is_monotonic,
            n_trades=len(y),
        )
        result = run_lag_shift_control(df, cfg, baseline)
        assert result.metrics["lag_r2"] <= baseline.r_squared

    def test_has_required_metrics(self):
        df = _make_dataset(120)
        cfg = load_pre_reg(PRE_REG_PATH)
        baseline = _make_baseline()
        result = run_lag_shift_control(df, cfg, baseline)
        assert "lag_r2" in result.metrics
        assert "lag_spread" in result.metrics
        assert "r2_ratio" in result.metrics
        assert "spread_ratio" in result.metrics

    def test_deterministic(self):
        df = _make_dataset(120)
        cfg = load_pre_reg(PRE_REG_PATH)
        baseline = _make_baseline()
        r1 = run_lag_shift_control(df, cfg, baseline)
        r2 = run_lag_shift_control(df, cfg, baseline)
        assert r1.metrics["lag_r2"] == r2.metrics["lag_r2"]


# ---------------------------------------------------------------------------
# Tests: Placebo-feature control
# ---------------------------------------------------------------------------

class TestPlaceboFeatureControl:
    def test_returns_control_result(self):
        df = _make_dataset(120)
        cfg = load_pre_reg(PRE_REG_PATH)
        baseline = _make_baseline()
        result = run_placebo_feature_control(df, cfg, baseline, seed=42)
        assert isinstance(result, ControlResult)
        assert result.name == "placebo_feature"

    def test_placebo_coef_near_zero(self):
        """Noise feature should get near-zero weight."""
        df = _make_dataset(120)
        cfg = load_pre_reg(PRE_REG_PATH)
        baseline = _make_baseline()
        result = run_placebo_feature_control(df, cfg, baseline, seed=42)
        assert result.metrics["placebo_coef_abs"] < 0.5  # generous bound

    def test_deterministic(self):
        df = _make_dataset(120)
        cfg = load_pre_reg(PRE_REG_PATH)
        baseline = _make_baseline()
        r1 = run_placebo_feature_control(df, cfg, baseline, seed=42)
        r2 = run_placebo_feature_control(df, cfg, baseline, seed=42)
        assert r1.metrics["placebo_coef_abs"] == r2.metrics["placebo_coef_abs"]

    def test_has_required_metrics(self):
        df = _make_dataset(120)
        cfg = load_pre_reg(PRE_REG_PATH)
        baseline = _make_baseline()
        result = run_placebo_feature_control(df, cfg, baseline, seed=42)
        assert "placebo_coef_abs" in result.metrics
        assert "coef_threshold" in result.metrics


# ---------------------------------------------------------------------------
# Tests: Report structure
# ---------------------------------------------------------------------------

class TestNegativeControlReport:
    def test_all_pass(self):
        report = NegativeControlReport(
            baseline=_make_baseline(),
            controls=[
                ControlResult("a", True, "ok"),
                ControlResult("b", True, "ok"),
                ControlResult("c", True, "ok"),
            ],
        )
        assert report.all_passed
        assert report.verdict == "PASS"

    def test_one_fail(self):
        report = NegativeControlReport(
            baseline=_make_baseline(),
            controls=[
                ControlResult("a", True, "ok"),
                ControlResult("b", False, "bad"),
                ControlResult("c", True, "ok"),
            ],
        )
        assert not report.all_passed
        assert report.verdict == "FAIL"

    def test_summary_contains_controls(self):
        report = NegativeControlReport(
            baseline=_make_baseline(),
            controls=[
                ControlResult("randomized_labels", True, "ok", {"mean_r2": 0.01}),
                ControlResult("lag_shift", True, "ok", {"lag_r2": 0.02}),
                ControlResult("placebo_feature", True, "ok", {"placebo_coef_abs": 0.001}),
            ],
        )
        s = report.summary()
        assert "randomized_labels" in s
        assert "lag_shift" in s
        assert "placebo_feature" in s
        assert "PASS" in s

    def test_summary_shows_fail(self):
        report = NegativeControlReport(
            baseline=_make_baseline(),
            controls=[
                ControlResult("x", False, "bad"),
            ],
        )
        s = report.summary()
        assert "FAIL" in s


# ---------------------------------------------------------------------------
# Tests: Full suite integration
# ---------------------------------------------------------------------------

class TestRunNegativeControls:
    def test_returns_report(self):
        df = _make_dataset(120)
        report = run_negative_controls(df, PRE_REG_PATH, n_permutations=2)
        assert isinstance(report, NegativeControlReport)
        assert len(report.controls) == 3

    def test_three_control_names(self):
        df = _make_dataset(120)
        report = run_negative_controls(df, PRE_REG_PATH, n_permutations=2)
        names = {c.name for c in report.controls}
        assert names == {"randomized_labels", "lag_shift", "placebo_feature"}

    def test_baseline_populated(self):
        df = _make_dataset(120)
        report = run_negative_controls(df, PRE_REG_PATH, n_permutations=2)
        assert report.baseline.n_trades == 120
        assert report.baseline.r_squared > 0  # synthetic data has signal

    def test_deterministic(self):
        df = _make_dataset(120)
        r1 = run_negative_controls(df, PRE_REG_PATH, n_permutations=2, base_seed=42)
        r2 = run_negative_controls(df, PRE_REG_PATH, n_permutations=2, base_seed=42)
        for c1, c2 in zip(r1.controls, r2.controls):
            assert c1.passed == c2.passed
            assert c1.metrics == c2.metrics
