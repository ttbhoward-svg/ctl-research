"""Unit tests for Drift Monitoring + Archive (Task 15)."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.archive import (
    DEFAULT_ARTIFACTS,
    ArchiveManifest,
    ArtifactEntry,
    build_manifest,
    file_sha256,
    load_manifest,
    save_manifest,
)
from ctl.drift_monitor import (
    DEFAULT_AVG_R_ALERT,
    DEFAULT_WIN_RATE_ALERT,
    PSI_OK,
    PSI_WATCH,
    BaselineProfile,
    DriftSnapshot,
    MetricDrift,
    build_baseline,
    check_drift,
    classify_psi,
    compute_psi,
    rolling_avg_r,
    rolling_win_rate,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_features(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "BarsOfAir": rng.integers(3, 15, n).astype(float),
        "Slope_20": rng.uniform(5, 20, n),
        "CleanPullback": rng.choice([0.0, 1.0], n),
    })


def _make_scores(n: int = 100, seed: int = 42) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.5, 0.3, n)


def _make_r_values(n: int = 100, seed: int = 42) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.5, 1.0, n)


# ---------------------------------------------------------------------------
# Tests: PSI computation
# ---------------------------------------------------------------------------

class TestComputePSI:
    def test_identical_distributions_near_zero(self):
        data = np.random.default_rng(42).normal(0, 1, 200)
        psi, edges = compute_psi(data, data)
        assert psi < 0.01  # near zero for identical data

    def test_shifted_distribution_higher_psi(self):
        rng = np.random.default_rng(42)
        baseline = rng.normal(0, 1, 500)
        shifted = rng.normal(2, 1, 500)  # big shift
        psi, _ = compute_psi(baseline, shifted)
        assert psi > PSI_WATCH  # should be ALERT level

    def test_returns_bin_edges(self):
        data = np.random.default_rng(42).normal(0, 1, 100)
        _, edges = compute_psi(data, data, n_bins=5)
        assert len(edges) >= 2

    def test_empty_baseline(self):
        psi, edges = compute_psi(np.array([]), np.array([1, 2, 3]))
        assert psi == 0.0

    def test_empty_current(self):
        psi, edges = compute_psi(np.array([1, 2, 3]), np.array([]))
        assert psi == 0.0

    def test_deterministic(self):
        data = np.random.default_rng(42).normal(0, 1, 100)
        other = np.random.default_rng(99).normal(0.5, 1, 100)
        psi1, _ = compute_psi(data, other)
        psi2, _ = compute_psi(data, other)
        assert psi1 == psi2

    def test_non_negative(self):
        rng = np.random.default_rng(42)
        psi, _ = compute_psi(rng.normal(0, 1, 100), rng.normal(1, 2, 100))
        assert psi >= 0.0


# ---------------------------------------------------------------------------
# Tests: PSI classification
# ---------------------------------------------------------------------------

class TestClassifyPSI:
    def test_ok(self):
        assert classify_psi(0.05) == "OK"

    def test_watch(self):
        assert classify_psi(0.15) == "WATCH"

    def test_alert(self):
        assert classify_psi(0.30) == "ALERT"

    def test_boundary_ok(self):
        assert classify_psi(0.0) == "OK"

    def test_boundary_watch(self):
        assert classify_psi(PSI_OK) == "WATCH"  # >= 0.10

    def test_boundary_alert(self):
        assert classify_psi(PSI_WATCH) == "ALERT"  # >= 0.25


# ---------------------------------------------------------------------------
# Tests: rolling metrics
# ---------------------------------------------------------------------------

class TestRollingMetrics:
    def test_rolling_avg_r_length(self):
        r = np.arange(30, dtype=float)
        result = rolling_avg_r(r, window=10)
        assert len(result) == 21  # 30 - 10 + 1

    def test_rolling_avg_r_values(self):
        r = np.ones(20)
        result = rolling_avg_r(r, window=5)
        np.testing.assert_allclose(result, 1.0)

    def test_rolling_win_rate_all_wins(self):
        r = np.ones(20)
        result = rolling_win_rate(r, window=5)
        np.testing.assert_allclose(result, 1.0)

    def test_rolling_win_rate_all_losses(self):
        r = -np.ones(20)
        result = rolling_win_rate(r, window=5)
        np.testing.assert_allclose(result, 0.0)

    def test_short_series_returns_single(self):
        r = np.array([1.0, 2.0, 3.0])
        result = rolling_avg_r(r, window=10)
        assert len(result) == 1
        assert result[0] == pytest.approx(2.0)

    def test_empty_series(self):
        result = rolling_avg_r(np.array([]), window=10)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: baseline profile
# ---------------------------------------------------------------------------

class TestBaselineProfile:
    def test_build_baseline(self):
        features = _make_features(100)
        scores = _make_scores(100)
        r_vals = _make_r_values(100)
        profile = build_baseline(features, scores, r_vals)
        assert profile.n_trades == 100
        assert len(profile.feature_bin_edges) == 3
        assert len(profile.score_bin_edges) >= 2
        assert profile.avg_r != 0.0

    def test_feature_stats_populated(self):
        features = _make_features(50)
        profile = build_baseline(features, np.zeros(50), np.zeros(50))
        for col in features.columns:
            assert col in profile.feature_stats
            stats = profile.feature_stats[col]
            assert "mean" in stats
            assert "std" in stats

    def test_to_dict_roundtrip(self):
        features = _make_features(50)
        scores = _make_scores(50)
        r_vals = _make_r_values(50)
        profile = build_baseline(features, scores, r_vals)
        d = profile.to_dict()
        restored = BaselineProfile.from_dict(d)
        assert restored.n_trades == profile.n_trades
        assert restored.avg_r == pytest.approx(profile.avg_r)
        assert set(restored.feature_bin_edges.keys()) == set(profile.feature_bin_edges.keys())

    def test_empty_data(self):
        profile = build_baseline(pd.DataFrame(), np.array([]), np.array([]))
        assert profile.n_trades == 0
        assert profile.avg_r == 0.0


# ---------------------------------------------------------------------------
# Tests: drift check — status transitions
# ---------------------------------------------------------------------------

class TestDriftCheck:
    def test_no_drift_all_ok(self):
        features = _make_features(200, seed=42)
        scores = _make_scores(200, seed=42)
        r_vals = np.ones(200) * 0.5  # stable positive R
        baseline = build_baseline(features, scores, r_vals)

        # Check the FULL baseline against itself → PSI ~ 0, outcome OK.
        snapshot = check_drift(features, scores, r_vals, baseline)
        assert snapshot.overall_status == "OK"

    def test_shifted_features_trigger_alert(self):
        features = _make_features(200, seed=42)
        baseline = build_baseline(features, _make_scores(200), _make_r_values(200))

        # Heavily shifted features.
        shifted = features.copy()
        shifted["BarsOfAir"] = shifted["BarsOfAir"] + 50  # huge shift
        snapshot = check_drift(shifted, None, None, baseline)
        psi_metrics = [m for m in snapshot.metrics if "psi" in m.name]
        assert any(m.status in ("WATCH", "ALERT") for m in psi_metrics)

    def test_negative_rolling_r_triggers_alert(self):
        features = _make_features(200)
        baseline = build_baseline(features, _make_scores(200), _make_r_values(200))

        bad_r = -np.ones(30)  # all losses
        snapshot = check_drift(None, None, bad_r, baseline)
        r_metric = next((m for m in snapshot.metrics if m.name == "rolling_avg_r"), None)
        assert r_metric is not None
        assert r_metric.status == "ALERT"

    def test_low_win_rate_triggers_alert(self):
        features = _make_features(200)
        baseline = build_baseline(features, _make_scores(200), _make_r_values(200))

        bad_r = np.full(30, -0.5)  # 0% win rate
        snapshot = check_drift(None, None, bad_r, baseline)
        wr_metric = next((m for m in snapshot.metrics if m.name == "rolling_win_rate"), None)
        assert wr_metric is not None
        assert wr_metric.status == "ALERT"

    def test_all_none_inputs(self):
        baseline = build_baseline(pd.DataFrame(), np.array([]), np.array([]))
        snapshot = check_drift(None, None, None, baseline)
        assert snapshot.overall_status == "OK"
        assert len(snapshot.metrics) == 0

    def test_snapshot_has_timestamp(self):
        baseline = build_baseline(pd.DataFrame(), np.array([]), np.array([]))
        snapshot = check_drift(None, None, None, baseline)
        assert len(snapshot.timestamp) > 0

    def test_snapshot_to_dict(self):
        features = _make_features(100)
        baseline = build_baseline(features, _make_scores(100), _make_r_values(100))
        snapshot = check_drift(features, _make_scores(50), np.ones(50), baseline)
        d = snapshot.to_dict()
        assert "timestamp" in d
        assert "overall_status" in d
        assert "metrics" in d

    def test_summary_contains_status(self):
        features = _make_features(100)
        baseline = build_baseline(features, _make_scores(100), _make_r_values(100))
        snapshot = check_drift(features, None, None, baseline)
        s = snapshot.summary()
        assert "Drift Monitor Snapshot" in s


# ---------------------------------------------------------------------------
# Tests: archive — file hashing
# ---------------------------------------------------------------------------

class TestFileHash:
    def test_known_content(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world")
            path = Path(f.name)
        h = file_sha256(path)
        path.unlink()
        # SHA-256 of "hello world" is known.
        assert h == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"

    def test_deterministic(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content")
            path = Path(f.name)
        h1 = file_sha256(path)
        h2 = file_sha256(path)
        path.unlink()
        assert h1 == h2


# ---------------------------------------------------------------------------
# Tests: archive — manifest generation
# ---------------------------------------------------------------------------

class TestArchiveManifest:
    def test_build_manifest_finds_existing_files(self):
        manifest = build_manifest(REPO_ROOT)
        assert manifest.n_found >= 1  # at least some default artifacts exist

    def test_missing_files_tracked(self):
        manifest = build_manifest(REPO_ROOT, extra_paths=["nonexistent_file.xyz"])
        missing = [a for a in manifest.extra_files if not a.exists]
        assert len(missing) == 1
        assert missing[0].path == "nonexistent_file.xyz"

    def test_custom_artifact_paths(self):
        manifest = build_manifest(
            REPO_ROOT,
            artifact_paths=["configs/phase1a.yaml"],
        )
        assert len(manifest.artifacts) == 1
        assert manifest.artifacts[0].path == "configs/phase1a.yaml"
        assert manifest.artifacts[0].exists

    def test_hash_populated_for_existing(self):
        manifest = build_manifest(REPO_ROOT, artifact_paths=["configs/phase1a.yaml"])
        entry = manifest.artifacts[0]
        assert len(entry.sha256) == 64  # SHA-256 hex digest

    def test_deterministic_hashes(self):
        m1 = build_manifest(REPO_ROOT, artifact_paths=["configs/phase1a.yaml"])
        m2 = build_manifest(REPO_ROOT, artifact_paths=["configs/phase1a.yaml"])
        assert m1.artifacts[0].sha256 == m2.artifacts[0].sha256

    def test_save_and_load_roundtrip(self):
        manifest = build_manifest(REPO_ROOT, artifact_paths=["configs/phase1a.yaml"])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            save_manifest(manifest, path)
            loaded = load_manifest(path)
        assert loaded["phase"] == "phase1a"
        assert loaded["n_found"] >= 1

    def test_to_dict_json_serializable(self):
        manifest = build_manifest(REPO_ROOT)
        d = manifest.to_dict()
        serialized = json.dumps(d)
        assert len(serialized) > 0

    def test_summary_contains_info(self):
        manifest = build_manifest(REPO_ROOT)
        s = manifest.summary()
        assert "Phase 1a Archive Manifest" in s
        assert "Found:" in s


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_psi_single_value_baseline(self):
        baseline = np.array([1.0])
        current = np.array([1.0, 2.0])
        psi, _ = compute_psi(baseline, current)
        assert psi >= 0.0

    def test_drift_check_with_empty_r(self):
        baseline = build_baseline(pd.DataFrame(), np.array([]), np.array([]))
        snapshot = check_drift(None, None, np.array([]), baseline)
        assert snapshot.overall_status == "OK"

    def test_baseline_with_nan_features(self):
        features = pd.DataFrame({"A": [1.0, np.nan, 3.0, 4.0, 5.0]})
        profile = build_baseline(features, np.zeros(5), np.zeros(5))
        assert "A" in profile.feature_bin_edges

    def test_manifest_empty_repo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = build_manifest(Path(tmpdir))
        assert manifest.n_found == 0
        assert manifest.n_missing == len(DEFAULT_ARTIFACTS)
