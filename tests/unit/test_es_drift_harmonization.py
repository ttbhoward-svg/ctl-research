"""Tests for ES drift harmonization helpers."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.es_drift_harmonization import (  # noqa: E402
    apply_regime_offsets,
    derive_regime_offsets,
    summarize_top_drift_intervals,
)
from ctl.roll_reconciliation import DriftExplanationResult, StepExplanation  # noqa: E402


class TestSummarizeTopDriftIntervals:
    def test_returns_top_n_by_contribution(self):
        explanation = DriftExplanationResult(
            symbol="ES",
            overall_mean_drift=1.0,
            overall_max_drift=2.0,
            n_intervals=3,
            intervals=[
                StepExplanation("a", "b", "PASS", 1.0, 2.0, 10.0),
                StepExplanation("b", "c", "WATCH", 2.0, 3.0, 30.0),
                StepExplanation("c", "d", "PASS", 3.0, 4.0, 20.0),
            ],
        )
        out = summarize_top_drift_intervals(explanation, top_n=2)
        assert len(out) == 2
        assert out[0].drift_contribution_pct == 30.0
        assert out[1].drift_contribution_pct == 20.0


class TestRegimeOffsets:
    def test_derive_and_apply_offsets(self):
        can = pd.DataFrame({
            "Date": ["2019-01-01", "2019-01-02", "2024-01-01", "2024-01-02"],
            "Close": [100.0, 100.0, 200.0, 200.0],
        })
        ts = pd.DataFrame({
            "Date": ["2019-01-01", "2019-01-02", "2024-01-01", "2024-01-02"],
            "Close": [90.0, 90.0, 195.0, 195.0],
        })
        offsets = derive_regime_offsets(
            can,
            ts,
            [("pre", "2018-01-01", "2019-12-31"), ("post", "2024-01-01", "2024-12-31")],
        )
        assert [round(x.median_signed_diff, 6) for x in offsets] == [10.0, 5.0]

        corrected = apply_regime_offsets(can, offsets)
        assert corrected["Close"].tolist() == [90.0, 90.0, 195.0, 195.0]
