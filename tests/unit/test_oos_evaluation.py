"""Unit tests for OOS Test + Calibration (Task 13)."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.dataset_assembler import SCHEMA_COLUMNS
from ctl.oos_evaluation import (
    MIN_OOS_TRADES,
    MIN_SCORE_CORRELATION,
    MIN_TERCILE_SPREAD_R,
    MIN_TOP_TERCILE_AVG_R,
    BucketMetrics,
    CalibrationRow,
    CriterionResult,
    OOSResult,
    assign_oos_terciles,
    assign_quintiles,
    build_calibration_table,
    compute_bucket_table,
    evaluate_oos,
    score_oos,
)
from ctl.regression import ModelResult, train_model

REPO_ROOT = Path(__file__).resolve().parents[2]
PRE_REG_PATH = REPO_ROOT / "configs" / "pre_registration_v1.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(n: int = 120, seed: int = 42, date_start: str = "2019-01-01") -> pd.DataFrame:
    """Synthetic assembled dataset (same as test_regression.py)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(date_start, periods=n, freq="7D")

    bars_of_air = rng.integers(3, 15, size=n).astype(float)
    slope_20 = rng.uniform(5, 20, size=n)
    theoretical_r = 0.05 * bars_of_air + 0.02 * slope_20 + rng.normal(0, 0.5, size=n)

    clusters = ["IDX_FUT", "METALS_FUT", "ENERGY_FUT", "ETF_SECTOR"]
    cluster_col = [clusters[i % len(clusters)] for i in range(n)]
    sym_map = {"IDX_FUT": "/ES", "METALS_FUT": "/GC", "ENERGY_FUT": "/CL", "ETF_SECTOR": "XLE"}
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


@pytest.fixture(scope="module")
def trained_model():
    """Train a model once for the entire test module."""
    is_df = _make_dataset(n=120, seed=42)
    return train_model(is_df, PRE_REG_PATH)


@pytest.fixture(scope="module")
def oos_df():
    """OOS dataset with different seed and later dates."""
    return _make_dataset(n=40, seed=99, date_start="2025-01-01")


# ---------------------------------------------------------------------------
# Tests: feature-lock enforcement
# ---------------------------------------------------------------------------

class TestFeatureLock:
    def test_missing_feature_raises(self, trained_model, oos_df):
        bad_df = oos_df.drop(columns=["BarsOfAir"])
        with pytest.raises(ValueError, match="Feature lock violation"):
            score_oos(bad_df, trained_model)

    def test_missing_cluster_raises(self, trained_model, oos_df):
        bad_df = oos_df.drop(columns=["AssetCluster"])
        with pytest.raises(ValueError, match="AssetCluster"):
            score_oos(bad_df, trained_model)

    def test_extra_columns_ignored(self, trained_model, oos_df):
        extra_df = oos_df.copy()
        extra_df["ExtraFeature"] = 999.0
        scores = score_oos(extra_df, trained_model)
        scores_orig = score_oos(oos_df, trained_model)
        np.testing.assert_array_equal(scores, scores_orig)

    def test_scores_length_matches_input(self, trained_model, oos_df):
        scores = score_oos(oos_df, trained_model)
        assert len(scores) == len(oos_df)


# ---------------------------------------------------------------------------
# Tests: bucket assignment
# ---------------------------------------------------------------------------

class TestBucketAssignment:
    def test_tercile_uses_is_thresholds(self):
        scores = np.array([0.1, 0.5, 0.9, 1.2, 1.5])
        thresholds = (0.4, 1.0)  # IS-derived
        buckets = assign_oos_terciles(scores, thresholds)
        assert buckets[0] == "bottom"  # 0.1 < 0.4
        assert buckets[1] == "mid"     # 0.5 >= 0.4, < 1.0
        assert buckets[2] == "mid"     # 0.9 >= 0.4, < 1.0
        assert buckets[3] == "top"     # 1.2 >= 1.0
        assert buckets[4] == "top"     # 1.5 >= 1.0

    def test_all_above_top_threshold(self):
        scores = np.array([2.0, 3.0, 4.0])
        buckets = assign_oos_terciles(scores, (0.5, 1.0))
        assert (buckets == "top").all()

    def test_all_below_bottom_threshold(self):
        scores = np.array([-1.0, -0.5, 0.0])
        buckets = assign_oos_terciles(scores, (0.5, 1.0))
        assert (buckets == "bottom").all()

    def test_quintile_labels(self):
        scores = np.arange(20, dtype=float)
        labels, boundaries = assign_quintiles(scores)
        assert set(labels) == {"Q1", "Q2", "Q3", "Q4", "Q5"}
        assert len(boundaries) == 4

    def test_quintile_boundaries_ordered(self):
        scores = np.random.default_rng(42).normal(0, 1, 50)
        _, boundaries = assign_quintiles(scores)
        assert all(boundaries[i] <= boundaries[i + 1] for i in range(3))

    def test_quintile_empty(self):
        labels, boundaries = assign_quintiles(np.array([]))
        assert len(labels) == 0
        assert boundaries == []


# ---------------------------------------------------------------------------
# Tests: metric calculations
# ---------------------------------------------------------------------------

class TestMetricCalculations:
    def test_bucket_metrics_basic(self):
        r_vals = np.array([1.0, 2.0, -0.5, 0.3])
        buckets = np.array(["top", "top", "bottom", "bottom"])
        table = compute_bucket_table(r_vals, buckets, ["top", "bottom"])
        assert len(table) == 2
        top = table[0]
        assert top.label == "top"
        assert top.n_trades == 2
        assert top.avg_r == pytest.approx(1.5)
        assert top.win_rate == pytest.approx(1.0)
        bot = table[1]
        assert bot.n_trades == 2
        assert bot.avg_r == pytest.approx(-0.1)

    def test_empty_bucket(self):
        r_vals = np.array([1.0])
        buckets = np.array(["top"])
        table = compute_bucket_table(r_vals, buckets, ["top", "mid", "bottom"])
        mid = table[1]
        assert mid.n_trades == 0
        assert mid.avg_r == 0.0

    def test_win_rate_calculation(self):
        r_vals = np.array([1.0, -0.5, 0.5, -1.0, 0.0])
        buckets = np.array(["a", "a", "a", "a", "a"])
        table = compute_bucket_table(r_vals, buckets, ["a"])
        # Wins: 1.0, 0.5 (strictly > 0). 0.0 is NOT a win.
        assert table[0].win_rate == pytest.approx(2.0 / 5.0)


# ---------------------------------------------------------------------------
# Tests: monotonicity / spread logic
# ---------------------------------------------------------------------------

class TestMonotonicitySpread:
    def _make_tercile_table(self, top_r, mid_r, bot_r):
        return [
            BucketMetrics("top", 10, top_r, 0.7, top_r * 10),
            BucketMetrics("mid", 10, mid_r, 0.5, mid_r * 10),
            BucketMetrics("bottom", 10, bot_r, 0.3, bot_r * 10),
        ]

    def test_monotonic_passes(self):
        from ctl.oos_evaluation import _check_monotonicity
        table = self._make_tercile_table(2.0, 1.0, 0.0)
        result = _check_monotonicity(table)
        assert result.passed

    def test_mid_above_top_fails(self):
        from ctl.oos_evaluation import _check_monotonicity
        table = self._make_tercile_table(1.0, 1.5, 0.0)
        result = _check_monotonicity(table)
        assert not result.passed

    def test_bottom_above_mid_fails(self):
        from ctl.oos_evaluation import _check_monotonicity
        table = self._make_tercile_table(2.0, 0.5, 1.0)
        result = _check_monotonicity(table)
        assert not result.passed

    def test_spread_above_threshold_passes(self):
        from ctl.oos_evaluation import _check_tercile_spread
        table = self._make_tercile_table(1.5, 0.5, 0.0)
        result = _check_tercile_spread(table)
        assert result.passed
        assert result.value == pytest.approx(1.5)

    def test_spread_below_threshold_fails(self):
        from ctl.oos_evaluation import _check_tercile_spread
        table = self._make_tercile_table(0.5, 0.3, 0.0)
        result = _check_tercile_spread(table)
        assert not result.passed
        assert result.value == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Tests: pass/fail reason generation
# ---------------------------------------------------------------------------

class TestCriteriaReasons:
    def test_trade_count_pass(self):
        from ctl.oos_evaluation import _check_trade_count
        r = _check_trade_count(50)
        assert r.passed
        assert "50" in r.reason

    def test_trade_count_fail(self):
        from ctl.oos_evaluation import _check_trade_count
        r = _check_trade_count(20)
        assert not r.passed
        assert "20" in r.reason

    def test_top_tercile_kill(self):
        from ctl.oos_evaluation import _check_top_tercile_kill
        table = [BucketMetrics("top", 10, 0.3, 0.5, 3.0)]
        r = _check_top_tercile_kill(table)
        assert not r.passed
        assert "KILL" in r.reason

    def test_score_correlation_kill(self):
        from ctl.oos_evaluation import _check_score_correlation
        scores = np.array([1.0, 2.0, 3.0])
        r_vals = np.array([-1.0, -2.0, -3.0])  # Perfect negative correlation
        r = _check_score_correlation(scores, r_vals)
        assert not r.passed
        assert "KILL" in r.reason

    def test_score_correlation_passes(self):
        from ctl.oos_evaluation import _check_score_correlation
        scores = np.array([1.0, 2.0, 3.0, 4.0])
        r_vals = np.array([0.5, 1.5, 2.5, 3.5])
        r = _check_score_correlation(scores, r_vals)
        assert r.passed

    def test_quintile_calibration_pass(self):
        from ctl.oos_evaluation import _check_quintile_calibration
        table = [
            BucketMetrics(f"Q{i}", 10, i * 0.5, 0.5, i * 5.0)
            for i in range(1, 6)
        ]
        r = _check_quintile_calibration(table)
        assert r.passed

    def test_quintile_calibration_fail(self):
        from ctl.oos_evaluation import _check_quintile_calibration
        # Q3 avg_r > Q4 avg_r → not monotonic
        table = [
            BucketMetrics("Q1", 10, 0.1, 0.5, 1.0),
            BucketMetrics("Q2", 10, 0.3, 0.5, 3.0),
            BucketMetrics("Q3", 10, 1.0, 0.5, 10.0),
            BucketMetrics("Q4", 10, 0.5, 0.5, 5.0),
            BucketMetrics("Q5", 10, 1.5, 0.5, 15.0),
        ]
        r = _check_quintile_calibration(table)
        assert not r.passed


# ---------------------------------------------------------------------------
# Tests: calibration reliability table
# ---------------------------------------------------------------------------

class TestCalibrationTable:
    def test_five_rows(self):
        scores = np.arange(25, dtype=float)
        r_vals = scores * 0.1
        labels, boundaries = assign_quintiles(scores)
        table = build_calibration_table(scores, r_vals, labels, boundaries)
        assert len(table) == 5
        assert [r.quintile for r in table] == ["Q1", "Q2", "Q3", "Q4", "Q5"]

    def test_avg_realized_r_increases(self):
        # Perfectly calibrated: score == realized R.
        scores = np.arange(50, dtype=float)
        r_vals = scores.copy()
        labels, boundaries = assign_quintiles(scores)
        table = build_calibration_table(scores, r_vals, labels, boundaries)
        avg_rs = [r.avg_realized_r for r in table]
        assert all(avg_rs[i] <= avg_rs[i + 1] for i in range(4))

    def test_n_trades_sum(self):
        scores = np.arange(25, dtype=float)
        r_vals = scores * 0.1
        labels, boundaries = assign_quintiles(scores)
        table = build_calibration_table(scores, r_vals, labels, boundaries)
        assert sum(r.n_trades for r in table) == 25


# ---------------------------------------------------------------------------
# Tests: deterministic outputs
# ---------------------------------------------------------------------------

class TestDeterministic:
    def test_same_input_same_output(self, trained_model, oos_df):
        r1 = evaluate_oos(oos_df, trained_model)
        r2 = evaluate_oos(oos_df, trained_model)
        np.testing.assert_array_equal(r1.scores, r2.scores)
        np.testing.assert_array_equal(r1.tercile_buckets, r2.tercile_buckets)
        assert r1.tercile_spread == r2.tercile_spread
        assert r1.score_r_correlation == r2.score_r_correlation

    def test_different_oos_different_output(self, trained_model):
        oos1 = _make_dataset(n=40, seed=99, date_start="2025-01-01")
        oos2 = _make_dataset(n=40, seed=77, date_start="2025-06-01")
        r1 = evaluate_oos(oos1, trained_model)
        r2 = evaluate_oos(oos2, trained_model)
        assert not np.array_equal(r1.scores, r2.scores)


# ---------------------------------------------------------------------------
# Tests: integration — full pipeline
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_evaluate_oos_returns_result(self, trained_model, oos_df):
        result = evaluate_oos(oos_df, trained_model)
        assert isinstance(result, OOSResult)
        assert result.n_trades == 40

    def test_tercile_table_has_three_buckets(self, trained_model, oos_df):
        result = evaluate_oos(oos_df, trained_model)
        labels = [b.label for b in result.tercile_table]
        assert labels == ["top", "mid", "bottom"]

    def test_quintile_table_has_five_buckets(self, trained_model, oos_df):
        result = evaluate_oos(oos_df, trained_model)
        labels = [b.label for b in result.quintile_table]
        assert labels == ["Q1", "Q2", "Q3", "Q4", "Q5"]

    def test_calibration_table_has_five_rows(self, trained_model, oos_df):
        result = evaluate_oos(oos_df, trained_model)
        assert len(result.calibration_table) == 5

    def test_criteria_count(self, trained_model, oos_df):
        result = evaluate_oos(oos_df, trained_model)
        assert len(result.criteria) == 6  # 4 gate + 2 kill

    def test_verdict_string(self, trained_model, oos_df):
        result = evaluate_oos(oos_df, trained_model)
        assert result.verdict in ("ALL_CRITERIA_MET", "CRITERIA_NOT_MET")

    def test_summary_contains_sections(self, trained_model, oos_df):
        result = evaluate_oos(oos_df, trained_model)
        s = result.summary()
        assert "OOS Evaluation Report" in s
        assert "Tercile Performance" in s
        assert "Quintile Calibration" in s
        assert "Gate Criteria" in s
        assert "Verdict" in s

    def test_trade_count_per_bucket_sums(self, trained_model, oos_df):
        result = evaluate_oos(oos_df, trained_model)
        total_tercile = sum(b.n_trades for b in result.tercile_table)
        assert total_tercile == result.n_trades
        total_quintile = sum(b.n_trades for b in result.quintile_table)
        assert total_quintile == result.n_trades


# ---------------------------------------------------------------------------
# Tests: export
# ---------------------------------------------------------------------------

class TestExport:
    def test_to_dict_json_serializable(self, trained_model, oos_df):
        result = evaluate_oos(oos_df, trained_model)
        d = result.to_dict()
        serialized = json.dumps(d)
        assert len(serialized) > 0

    def test_save_json_roundtrip(self, trained_model, oos_df):
        result = evaluate_oos(oos_df, trained_model)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            result.save_json(path)
            with open(path) as f:
                loaded = json.load(f)
        assert loaded["n_trades"] == 40
        assert "criteria" in loaded
        assert len(loaded["criteria"]) == 6

    def test_save_bucket_csv(self, trained_model, oos_df):
        result = evaluate_oos(oos_df, trained_model)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "buckets.csv"
            result.save_bucket_csv(path)
            df = pd.read_csv(path)
        # 3 tercile + 5 quintile = 8 rows.
        assert len(df) == 8
        assert "group_type" in df.columns
        assert "label" in df.columns


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_oos(self, trained_model):
        empty_df = _make_dataset(n=120).head(0)
        result = evaluate_oos(empty_df, trained_model)
        assert result.n_trades == 0
        assert result.tercile_spread == 0.0
        assert len(result.criteria) == 1  # only trade count check

    def test_single_trade(self, trained_model):
        single = _make_dataset(n=40, seed=99, date_start="2025-01-01").head(1)
        result = evaluate_oos(single, trained_model)
        assert result.n_trades == 1
        assert result.score_r_correlation == 0.0  # can't compute with n=1

    def test_missing_target_raises(self, trained_model, oos_df):
        bad = oos_df.drop(columns=["TheoreticalR"])
        with pytest.raises(ValueError, match="TheoreticalR"):
            evaluate_oos(bad, trained_model)

    def test_novel_cluster_handled(self, trained_model, oos_df):
        """OOS data with a cluster not seen in IS should still score."""
        novel = oos_df.copy()
        novel["AssetCluster"] = "NOVEL_CLUSTER"
        # Should not raise — novel cluster gets all-zero dummies.
        scores = score_oos(novel, trained_model)
        assert len(scores) == len(novel)
