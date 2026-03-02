"""Tests for PL gap-bias correction helpers."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.pl_gap_correction import apply_gap_bias, estimate_gap_bias  # noqa: E402
from ctl.roll_reconciliation import RollManifestEntry  # noqa: E402


class TestEstimateGapBias:
    def test_estimates_from_pass_watch(self):
        df = pd.DataFrame(
            {
                "status": ["PASS", "WATCH", "FAIL"],
                "canonical_gap": [2.0, 3.0, 10.0],
                "ts_gap": [1.0, 1.5, 2.0],
            }
        )
        est = estimate_gap_bias(df)
        # signed deltas considered: 1.0, 1.5 -> median 1.25
        assert est.n_rows == 2
        assert round(est.median_signed_gap_delta, 6) == 1.25

    def test_empty_returns_zero(self):
        est = estimate_gap_bias(pd.DataFrame())
        assert est.n_rows == 0
        assert est.median_signed_gap_delta == 0.0


class TestApplyGapBias:
    def test_applies_and_recomputes_cumulative(self):
        entries = [
            RollManifestEntry("2020-01-01", "A", "B", 100.0, 101.0, 1.0, 1.0),
            RollManifestEntry("2020-02-01", "B", "C", 110.0, 111.0, 1.0, 2.0),
        ]
        out = apply_gap_bias(entries, signed_gap_bias=0.5)
        assert len(out) == 2
        assert round(out[0].gap, 6) == 0.5
        assert round(out[1].gap, 6) == 0.5
        assert round(out[0].cumulative_adj, 6) == 0.5
        assert round(out[1].cumulative_adj, 6) == 1.0
        assert round(out[0].to_close, 6) == 100.5
