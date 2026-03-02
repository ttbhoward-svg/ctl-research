"""Tests for PL interval-level basis analysis."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.pl_basis_analysis import (  # noqa: E402
    build_interval_basis_report,
    save_interval_basis_report,
)


def _make_drift_df() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    close_can = [100, 101, 102, 103, 104, 105, 106, 110, 111, 112]
    close_ts = [99, 100, 101, 102, 103, 104, 105, 107, 108, 109]
    return pd.DataFrame({"Date": dates, "close_can": close_can, "close_ts": close_ts})


def _make_l2_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "status": ["FAIL", "WATCH", "FAIL"],
            "canonical_date": ["2024-01-03", "2024-01-07", None],
            "ts_date": [None, "2024-01-08", "2024-01-09"],
        }
    )


def _make_explanation() -> dict:
    return {
        "intervals": [
            {
                "interval_start": "2024-01-01",
                "interval_end": "2024-01-05",
                "roll_status": "WATCH",
                "drift_contribution_pct": 60.0,
            },
            {
                "interval_start": "2024-01-06",
                "interval_end": "2024-01-10",
                "roll_status": "WATCH",
                "drift_contribution_pct": 40.0,
            },
        ]
    }


class TestBuildIntervalBasisReport:
    def test_builds_ranked_rows(self):
        report = build_interval_basis_report(
            drift_df=_make_drift_df(),
            l2_detail_df=_make_l2_df(),
            explanation=_make_explanation(),
            top_n=2,
            roll_window_days=2,
        )
        assert len(report) == 2
        assert report.iloc[0]["drift_contribution_pct"] >= report.iloc[1]["drift_contribution_pct"]
        assert "median_signed_diff" in report.columns
        assert "pct_can_above_ts" in report.columns

    def test_top_n_limits_rows(self):
        report = build_interval_basis_report(
            drift_df=_make_drift_df(),
            l2_detail_df=_make_l2_df(),
            explanation=_make_explanation(),
            top_n=1,
        )
        assert len(report) == 1

    def test_empty_inputs_return_empty(self):
        report = build_interval_basis_report(
            drift_df=pd.DataFrame(),
            l2_detail_df=pd.DataFrame(),
            explanation={},
        )
        assert report.empty


class TestSaveIntervalBasisReport:
    def test_save_csv(self, tmp_path):
        report = build_interval_basis_report(
            drift_df=_make_drift_df(),
            l2_detail_df=_make_l2_df(),
            explanation=_make_explanation(),
            top_n=2,
        )
        out = tmp_path / "pl_report.csv"
        saved = save_interval_basis_report(report, out)
        assert saved == out
        assert out.exists()

    def test_save_none_on_empty(self, tmp_path):
        out = tmp_path / "empty.csv"
        saved = save_interval_basis_report(pd.DataFrame(), out)
        assert saved is None
        assert not out.exists()
