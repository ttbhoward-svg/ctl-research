"""Tests for generalized PL month-gap correction helpers."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.pl_gap_generalized import apply_month_gap_biases, estimate_month_gap_biases  # noqa: E402
from ctl.roll_reconciliation import RollManifestEntry  # noqa: E402


class TestEstimateMonthGapBiases:
    def test_train_end_filters_rows(self):
        l2 = pd.DataFrame(
            {
                "status": ["PASS", "PASS", "PASS"],
                "canonical_date": ["2023-01-01", "2023-02-01", "2024-01-01"],
                "to_contract": ["PLJ3", "PLN3", "PLJ4"],
                "canonical_gap": [2.0, 1.0, 9.0],
                "ts_gap": [1.0, 0.0, 0.0],
            }
        )
        out = estimate_month_gap_biases(l2, train_end_date="2023-12-31")
        by_month = {x.month_code: x for x in out}
        assert set(by_month.keys()) == {"J", "N"}
        assert by_month["J"].median_signed_gap_delta == 1.0
        assert by_month["N"].median_signed_gap_delta == 1.0


class TestApplyMonthGapBiases:
    def test_applies_only_after_start_date(self):
        manifest = [
            RollManifestEntry("2023-12-01", "PLV3", "PLF4", 100.0, 103.0, 3.0, 3.0),
            RollManifestEntry("2024-01-01", "PLF4", "PLJ4", 103.0, 105.0, 2.0, 5.0),
        ]
        biases = [
            type("B", (), {"month_code": "F", "median_signed_gap_delta": 1.0}),
            type("B", (), {"month_code": "J", "median_signed_gap_delta": 0.5}),
        ]
        out = apply_month_gap_biases(manifest, biases, apply_start_date="2024-01-01")

        # first row unchanged (before apply_start), second corrected by J bias
        assert [round(x.gap, 6) for x in out] == [3.0, 1.5]
        assert [round(x.cumulative_adj, 6) for x in out] == [3.0, 4.5]
