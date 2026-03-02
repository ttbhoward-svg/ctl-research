"""Unit tests for promotion-priority helpers."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.canonical_acceptance import (  # noqa: E402
    AcceptanceThresholds,
    FuturesAcceptanceInput,
    evaluate_futures_acceptance,
)
from ctl.promotion_priority import (  # noqa: E402
    build_priority_row,
    extract_mtfa_rates,
    load_latest_run_summary,
    rank_priority,
)


def _acc(symbol: str, mean_gap: float, mean_drift: float):
    inp = FuturesAcceptanceInput(
        symbol=symbol,
        n_canonical=32,
        n_ts=32,
        n_paired=32,
        n_matched=0,
        n_watch=32,
        n_fail=0,
        unmatched_canonical=0,
        unmatched_ts=0,
        mean_gap_diff=mean_gap,
        max_gap_diff=5.0,
        mean_drift=mean_drift,
        max_drift=50.0,
        strict_status="WATCH",
        policy_status="WATCH",
    )
    return evaluate_futures_acceptance(inp, AcceptanceThresholds())


class TestLoadLatestRunSummary:
    def test_none_when_empty(self, tmp_path):
        assert load_latest_run_summary(tmp_path) is None

    def test_loads_latest_sorted_file(self, tmp_path):
        a = tmp_path / "20260101_010101_portfolio_run.json"
        b = tmp_path / "20260101_010102_portfolio_run.json"
        a.write_text(json.dumps({"timestamp": "a"}))
        b.write_text(json.dumps({"timestamp": "b"}))
        loaded = load_latest_run_summary(tmp_path)
        assert loaded["timestamp"] == "b"


class TestExtractMtfaRates:
    def test_extracts_per_symbol(self):
        summary = {
            "symbol_run_results": [
                {"symbol": "ES", "mtfa_weekly_rate": 0.1, "mtfa_monthly_rate": 0.2},
                {"symbol": "CL", "mtfa_weekly_rate": 0.7, "mtfa_monthly_rate": 0.3},
            ]
        }
        out = extract_mtfa_rates(summary)
        assert out["ES"]["mtfa_weekly_rate"] == 0.1
        assert out["CL"]["mtfa_monthly_rate"] == 0.3

    def test_empty_when_none(self):
        assert extract_mtfa_rates(None) == {}


class TestBuildPriorityRow:
    def test_drift_only_blocker_scores_positive(self):
        acc = _acc("ES", mean_gap=0.5, mean_drift=7.33)
        row = build_priority_row("ES", acc)
        assert row.priority_score > 0.0
        assert row.drift_excess > 0.0
        assert row.gap_excess == 0.0

    def test_gap_and_drift_raise_score(self):
        a = _acc("ES", mean_gap=0.5, mean_drift=7.33)
        b = _acc("PL", mean_gap=1.66, mean_drift=8.28)
        row_a = build_priority_row("ES", a)
        row_b = build_priority_row("PL", b)
        assert row_b.priority_score > row_a.priority_score

    def test_mtfa_rates_attached(self):
        acc = _acc("CL", mean_gap=0.5, mean_drift=4.0)
        row = build_priority_row("CL", acc, mtfa={"mtfa_weekly_rate": 0.6, "mtfa_monthly_rate": 0.25})
        assert row.mtfa_weekly_rate == 0.6
        assert row.mtfa_monthly_rate == 0.25


class TestRankPriority:
    def test_highest_score_first(self):
        es = build_priority_row("ES", _acc("ES", 0.5, 7.33))
        pl = build_priority_row("PL", _acc("PL", 1.66, 8.28))
        ranked = rank_priority([es, pl])
        assert ranked[0].symbol == "PL"
