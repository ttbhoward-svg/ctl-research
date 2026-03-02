"""Tests for PL feature-model gap correction helpers."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.pl_gap_feature_model import apply_feature_bias_model, estimate_feature_bias_model  # noqa: E402
from ctl.roll_reconciliation import RollManifestEntry  # noqa: E402


class TestEstimateFeatureBiasModel:
    def test_builds_hierarchical_maps(self):
        l2 = pd.DataFrame(
            {
                "status": ["PASS", "PASS", "PASS", "WATCH", "PASS"],
                "canonical_date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2024-01-01"],
                "to_contract": ["PLJ3", "PLJ3", "PLN3", "PLN3", "PLJ4"],
                "canonical_gap": [2.0, 2.0, -1.0, -1.0, 10.0],
                "ts_gap": [1.0, 1.0, -2.0, -2.0, 0.0],
            }
        )
        model = estimate_feature_bias_model(l2, train_end_date="2023-12-31", min_rows=2)
        assert model.exact[("mid_2020_2023", "J", 1)] == 1.0
        assert model.exact[("mid_2020_2023", "N", -1)] == 1.0
        assert model.regime_month[("mid_2020_2023", "J")] == 1.0
        assert model.month["N"] == 1.0


class TestApplyFeatureBiasModel:
    def test_uses_priority_exact_then_fallback(self):
        model = estimate_feature_bias_model(
            pd.DataFrame(
                {
                    "status": ["PASS", "PASS", "PASS", "PASS"],
                    "canonical_date": ["2023-01-01", "2023-01-02", "2023-02-01", "2023-02-02"],
                    "to_contract": ["PLJ3", "PLJ3", "PLN3", "PLN3"],
                    "canonical_gap": [2.0, 2.0, 3.0, 3.0],
                    "ts_gap": [1.0, 1.0, 1.0, 1.0],
                }
            ),
            train_end_date="2023-12-31",
            min_rows=2,
        )
        manifest = [
            # exact key exists: mid_2020_2023, J, sign=+1 => bias 1.0
            RollManifestEntry("2024-01-01", "PLF4", "PLJ4", 100.0, 102.0, 2.0, 2.0),
            # month fallback N exists (exact sign might not): bias 2.0
            RollManifestEntry("2024-02-01", "PLJ4", "PLN4", 102.0, 105.0, 3.0, 5.0),
        ]
        out = apply_feature_bias_model(manifest, model, apply_start_date="2024-01-01")
        assert [round(x.gap, 6) for x in out] == [1.0, 1.0]

    def test_respects_apply_start_date(self):
        model = estimate_feature_bias_model(
            pd.DataFrame(
                {
                    "status": ["PASS", "PASS"],
                    "canonical_date": ["2023-01-01", "2023-01-02"],
                    "to_contract": ["PLJ3", "PLJ3"],
                    "canonical_gap": [2.0, 2.0],
                    "ts_gap": [1.0, 1.0],
                }
            ),
            train_end_date="2023-12-31",
            min_rows=2,
        )
        manifest = [
            RollManifestEntry("2023-12-01", "PLV3", "PLJ4", 100.0, 102.0, 2.0, 2.0),
            RollManifestEntry("2024-01-01", "PLF4", "PLJ4", 102.0, 104.0, 2.0, 4.0),
        ]
        out = apply_feature_bias_model(manifest, model, apply_start_date="2024-01-01")
        assert [round(x.gap, 6) for x in out] == [2.0, 1.0]
