"""Tests for PL window-specific gap correction helpers."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.pl_gap_window_correction import apply_window_biases, select_top_window_biases  # noqa: E402
from ctl.roll_reconciliation import RollManifestEntry  # noqa: E402


class TestSelectTopWindowBiases:
    def test_selects_top_k_by_abs(self):
        l2 = pd.DataFrame(
            {
                "status": ["PASS", "WATCH", "PASS"],
                "canonical_date": ["2025-01-01", "2025-02-01", "2025-03-01"],
                "from_contract": ["A", "B", "C"],
                "to_contract": ["B", "C", "D"],
                "canonical_gap": [5.0, 2.0, 1.0],
                "ts_gap": [0.0, 1.0, 1.2],
            }
        )
        out = select_top_window_biases(l2, top_k=2)
        assert len(out) == 2
        # largest abs signed deltas are 5.0 and 1.0
        assert out[0].from_contract == "A"
        assert out[1].from_contract == "B"


class TestApplyWindowBiases:
    def test_applies_only_matching_windows(self):
        manifest = [
            RollManifestEntry("2025-01-01", "A", "B", 100.0, 105.0, 5.0, 5.0),
            RollManifestEntry("2025-02-01", "B", "C", 110.0, 112.0, 2.0, 7.0),
        ]
        biases = [
            type("W", (), {"roll_date": "2025-01-01", "from_contract": "A", "to_contract": "B", "signed_gap_delta": 5.0}),
        ]
        out = apply_window_biases(manifest, biases)
        # first corrected to zero gap, second unchanged
        assert [round(x.gap, 6) for x in out] == [0.0, 2.0]
        assert [round(x.cumulative_adj, 6) for x in out] == [0.0, 2.0]
