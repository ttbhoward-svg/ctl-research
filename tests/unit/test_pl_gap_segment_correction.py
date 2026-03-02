"""Tests for PL segment-based gap correction helpers."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.pl_gap_segment_correction import apply_segment_gap_bias, estimate_segment_gap_bias  # noqa: E402
from ctl.roll_reconciliation import RollManifestEntry  # noqa: E402


class TestEstimateSegmentGapBias:
    def test_estimates_per_segment(self):
        l2 = pd.DataFrame(
            {
                "status": ["PASS", "WATCH", "PASS"],
                "canonical_date": ["2019-01-01", "2021-01-01", "2025-01-01"],
                "canonical_gap": [2.0, 3.0, 4.0],
                "ts_gap": [1.0, 2.0, 5.0],
            }
        )
        segs = [
            ("pre", "2018-01-01", "2019-12-31"),
            ("mid", "2020-01-01", "2023-12-31"),
            ("post", "2024-01-01", "2026-12-31"),
        ]
        out = estimate_segment_gap_bias(l2, segs)
        assert len(out) == 3
        assert round(out[0].median_signed_gap_delta, 6) == 1.0
        assert round(out[1].median_signed_gap_delta, 6) == 1.0
        assert round(out[2].median_signed_gap_delta, 6) == -1.0


class TestApplySegmentGapBias:
    def test_applies_by_roll_date_segment(self):
        entries = [
            RollManifestEntry("2019-01-01", "A", "B", 100.0, 102.0, 2.0, 2.0),
            RollManifestEntry("2021-01-01", "B", "C", 110.0, 113.0, 3.0, 5.0),
            RollManifestEntry("2025-01-01", "C", "D", 120.0, 124.0, 4.0, 9.0),
        ]
        biases = [
            type("B", (), {"start": "2018-01-01", "end": "2019-12-31", "median_signed_gap_delta": 1.0}),
            type("B", (), {"start": "2020-01-01", "end": "2023-12-31", "median_signed_gap_delta": 1.0}),
            type("B", (), {"start": "2024-01-01", "end": "2026-12-31", "median_signed_gap_delta": -1.0}),
        ]
        out = apply_segment_gap_bias(entries, biases)
        # gap corrected = gap - bias => [1.0, 2.0, 5.0]
        assert [round(x.gap, 6) for x in out] == [1.0, 2.0, 5.0]
        # cumulative recomputed from corrected gaps.
        assert [round(x.cumulative_adj, 6) for x in out] == [1.0, 3.0, 8.0]
