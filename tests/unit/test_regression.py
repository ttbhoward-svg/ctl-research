"""Unit tests for Phase 1a Elastic Net regression pipeline."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.dataset_assembler import SCHEMA_COLUMNS
from ctl.regression import (
    CoefficientTable,
    PreRegConfig,
    assign_terciles,
    check_signs,
    compute_diagnostics,
    load_pre_reg,
    prepare_features,
    train_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
PRE_REG_PATH = REPO_ROOT / "configs" / "pre_registration_v1.yaml"


def _make_dataset(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic assembled dataset for testing.

    Creates n trades with all SCHEMA_COLUMNS populated.
    Features are constructed so that higher BarsOfAir/Slope_20 predict
    higher TheoreticalR, making the Elastic Net learnable.
    """
    rng = np.random.default_rng(seed)

    # Date range spanning IS period.
    dates = pd.date_range("2019-01-01", periods=n, freq="7D")

    # Technical features with signal.
    bars_of_air = rng.integers(3, 15, size=n).astype(float)
    slope_20 = rng.uniform(5, 20, size=n)

    # Target: correlated with bars_of_air + slope_20 + noise.
    theoretical_r = 0.05 * bars_of_air + 0.02 * slope_20 + rng.normal(0, 0.5, size=n)

    # Clusters: rotate through a few.
    clusters = ["IDX_FUT", "METALS_FUT", "ENERGY_FUT", "ETF_SECTOR"]
    cluster_col = [clusters[i % len(clusters)] for i in range(n)]

    # Symbols corresponding to clusters.
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


# ---------------------------------------------------------------------------
# Tests: config loading
# ---------------------------------------------------------------------------

class TestLoadPreReg:
    def test_loads_successfully(self):
        cfg = load_pre_reg(PRE_REG_PATH)
        assert len(cfg.feature_names) == 9
        assert cfg.target == "TheoreticalR"
        assert cfg.cluster_column == "AssetCluster"
        assert cfg.random_seed == 42
        assert cfg.cv_folds == 5
        assert cfg.purge_gap_days == 30

    def test_expected_signs_all_positive(self):
        cfg = load_pre_reg(PRE_REG_PATH)
        for name, sign in cfg.expected_signs.items():
            assert sign == "positive", f"{name} expected sign is {sign}, not positive"

    def test_feature_names_match_schema(self):
        cfg = load_pre_reg(PRE_REG_PATH)
        for name in cfg.feature_names:
            assert name in SCHEMA_COLUMNS, f"{name} not in SCHEMA_COLUMNS"

    def test_invalid_feature_cap_raises(self):
        """YAML with != 9 candidates should raise ValueError."""
        bad_yaml = {
            "features": {
                "candidates": [{"name": "A", "expected_sign": "positive"}],
                "control": [{"name": "C", "type": "categorical"}],
            },
            "model": {
                "target": "TheoreticalR", "random_seed": 42,
                "cv_folds": 5, "purge_gap_days": 30,
            },
        }
        import yaml
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            yaml.dump(bad_yaml, f)
            path = Path(f.name)
        with pytest.raises(ValueError, match="Feature cap"):
            load_pre_reg(path)
        path.unlink()


# ---------------------------------------------------------------------------
# Tests: feature preparation
# ---------------------------------------------------------------------------

class TestPrepareFeatures:
    def test_output_shapes(self):
        df = _make_dataset(n=60)
        cfg = load_pre_reg(PRE_REG_PATH)
        X_cand, y, dates, names, cluster_dummies = prepare_features(df, cfg)
        assert X_cand.shape == (60, 9)
        assert len(y) == 60
        assert len(dates) == 60
        assert len(names) == 9
        assert cluster_dummies.shape[0] == 60

    def test_no_nans_after_fill(self):
        df = _make_dataset(n=60)
        cfg = load_pre_reg(PRE_REG_PATH)
        X_cand, y, _, _, _ = prepare_features(df, cfg)
        assert X_cand.isna().sum().sum() == 0

    def test_all_columns_float64(self):
        df = _make_dataset(n=60)
        cfg = load_pre_reg(PRE_REG_PATH)
        X_cand, _, _, _, _ = prepare_features(df, cfg)
        for col in X_cand.columns:
            assert X_cand[col].dtype == np.float64, f"{col} is {X_cand[col].dtype}"

    def test_missing_feature_raises(self):
        df = _make_dataset(n=20)
        df = df.drop(columns=["BarsOfAir"])
        cfg = load_pre_reg(PRE_REG_PATH)
        with pytest.raises(ValueError, match="BarsOfAir"):
            prepare_features(df, cfg)

    def test_missing_target_raises(self):
        df = _make_dataset(n=20)
        df = df.drop(columns=["TheoreticalR"])
        cfg = load_pre_reg(PRE_REG_PATH)
        with pytest.raises(ValueError, match="TheoreticalR"):
            prepare_features(df, cfg)


# ---------------------------------------------------------------------------
# Tests: sign check
# ---------------------------------------------------------------------------

class TestCheckSigns:
    def test_all_positive_match(self):
        coefs = np.array([0.5, 0.3, 0.1, 0.2, 0.4, 0.1, 0.05, 0.02, 0.3])
        names = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
        expected = {n: "positive" for n in names}
        table = check_signs(coefs, names, expected)
        assert table.n_matches == 9
        assert table.n_mismatches == 0
        assert not table.sign_flip_pause

    def test_one_negative_mismatch(self):
        coefs = np.array([0.5, -0.3, 0.1, 0.2, 0.4, 0.1, 0.05, 0.02, 0.3])
        names = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
        expected = {n: "positive" for n in names}
        table = check_signs(coefs, names, expected)
        assert table.n_mismatches == 1
        assert not table.sign_flip_pause  # Need 2+ for PAUSE

    def test_two_mismatches_triggers_pause(self):
        coefs = np.array([0.5, -0.3, -0.1, 0.2, 0.4, 0.1, 0.05, 0.02, 0.3])
        names = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
        expected = {n: "positive" for n in names}
        table = check_signs(coefs, names, expected)
        assert table.n_mismatches == 2
        assert table.sign_flip_pause

    def test_zeroed_not_counted_as_mismatch(self):
        coefs = np.array([0.5, 0.0, 0.0, 0.0, 0.4, 0.1, 0.0, 0.0, 0.3])
        names = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
        expected = {n: "positive" for n in names}
        table = check_signs(coefs, names, expected)
        assert table.n_zeroed == 5
        assert table.n_mismatches == 0
        assert not table.sign_flip_pause


# ---------------------------------------------------------------------------
# Tests: tercile assignment
# ---------------------------------------------------------------------------

class TestAssignTerciles:
    def test_three_equal_groups(self):
        scores = np.arange(9, dtype=float)
        buckets, thresholds = assign_terciles(scores)
        assert (buckets == "bottom").sum() == 3
        assert (buckets == "mid").sum() == 3
        assert (buckets == "top").sum() == 3

    def test_remainder_goes_to_top(self):
        scores = np.arange(10, dtype=float)
        buckets, _ = assign_terciles(scores)
        assert (buckets == "top").sum() >= 4  # 10 // 3 = 3, remainder = 1

    def test_empty_scores(self):
        buckets, thresholds = assign_terciles(np.array([]))
        assert len(buckets) == 0

    def test_thresholds_order(self):
        scores = np.arange(30, dtype=float)
        _, (low, high) = assign_terciles(scores)
        assert low <= high

    def test_all_same_scores(self):
        scores = np.ones(9)
        buckets, _ = assign_terciles(scores)
        # All scores equal — all should be "top" (>= high_cutoff where high == 1.0).
        assert len(buckets) == 9


# ---------------------------------------------------------------------------
# Tests: diagnostics
# ---------------------------------------------------------------------------

class TestDiagnostics:
    def test_correlation_matrix_shape(self):
        df = _make_dataset(n=60)
        cfg = load_pre_reg(PRE_REG_PATH)
        X_cand, _, _, names, _ = prepare_features(df, cfg)
        corr, flags = compute_diagnostics(X_cand, names)
        assert corr.shape == (9, 9)

    def test_no_self_correlation_flag(self):
        """Diagonal is 1.0 but should not be flagged."""
        df = _make_dataset(n=60)
        cfg = load_pre_reg(PRE_REG_PATH)
        X_cand, _, _, names, _ = prepare_features(df, cfg)
        _, flags = compute_diagnostics(X_cand, names)
        for f in flags:
            parts = f.split(" x ")
            assert parts[0] != parts[1]


# ---------------------------------------------------------------------------
# Tests: deterministic pipeline
# ---------------------------------------------------------------------------

class TestTrainModel:
    def test_deterministic_output(self):
        """Same data + same seed => same coefficients and scores."""
        df = _make_dataset(n=120, seed=42)
        r1 = train_model(df, PRE_REG_PATH)
        r2 = train_model(df, PRE_REG_PATH)
        np.testing.assert_array_equal(r1.scores, r2.scores)
        for e1, e2 in zip(r1.coef_table.entries, r2.coef_table.entries):
            assert e1.coefficient == e2.coefficient

    def test_result_fields(self):
        df = _make_dataset(n=120)
        result = train_model(df, PRE_REG_PATH)
        assert len(result.feature_names) == 9
        assert len(result.scores) == 120
        assert len(result.buckets) == 120
        assert result.diagnostics.alpha > 0
        assert 0 <= result.diagnostics.l1_ratio <= 1
        assert len(result.coef_table.entries) == 9

    def test_buckets_are_valid(self):
        df = _make_dataset(n=120)
        result = train_model(df, PRE_REG_PATH)
        valid = {"top", "mid", "bottom"}
        assert set(result.buckets).issubset(valid)

    def test_scaler_is_fitted(self):
        df = _make_dataset(n=120)
        result = train_model(df, PRE_REG_PATH)
        # Scaler should have mean_ and scale_ attributes.
        assert hasattr(result.scaler, "mean_")
        assert len(result.scaler.mean_) == 9
