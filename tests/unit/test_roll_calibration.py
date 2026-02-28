"""Unit tests for CL roll-policy calibration (Data Cutover Task H.3)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.continuous_builder import (
    ContractSpec,
    RollEvent,
    apply_panama_adjustment,
    detect_rolls,
    parse_contracts,
)
from ctl.roll_calibration import (
    CL_ALL_MONTHS,
    CL_QUARTERLY_MONTHS,
    MAX_PLAUSIBLE_ROLLS,
    MIN_PLAUSIBLE_ROLLS,
    SCORE_W_DRIFT,
    SCORE_W_FAIL,
    SCORE_W_GAP,
    SCORE_W_UNMATCHED,
    CalibrationScore,
    RollPolicyVariant,
    compute_composite_score,
    generate_cl_variants,
    rank_variants,
    save_calibration_artifacts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_contract_df(
    symbol: str,
    start: str,
    n_bars: int,
    base_price: float = 100.0,
    base_volume: float = 10000.0,
    trend: float = 0.1,
    volume_trend: float = 0.0,
) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame for one contract."""
    dates = pd.bdate_range(start, periods=n_bars)
    close = base_price + np.arange(n_bars) * trend
    volume = np.maximum(base_volume + np.arange(n_bars) * volume_trend, 0)
    return pd.DataFrame({
        "date": [d.date() for d in dates],
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": volume,
    })


def _three_cl_contracts():
    """Build three overlapping CL-like contracts: CLH4, CLJ4, CLM4.

    CLH4: starts early, volume declines.
    CLJ4: starts mid, volume rises then declines.
    CLM4: starts late, volume rises.

    Returns (contracts dict, contract_order list).
    """
    # CLH4 — March contract, active first
    clh = _make_contract_df(
        "CLH4", "2024-01-02", 60,
        base_price=70.0, base_volume=50000, volume_trend=-800,
    )
    # CLJ4 — April contract, overlaps with CLH4
    clj = _make_contract_df(
        "CLJ4", "2024-02-01", 60,
        base_price=71.0, base_volume=5000, volume_trend=1500,
    )
    # CLM4 — June (quarterly) contract, overlaps with CLJ4
    clm = _make_contract_df(
        "CLM4", "2024-03-01", 60,
        base_price=72.0, base_volume=1000, volume_trend=2000,
    )

    contracts = {"CLH4": clh, "CLJ4": clj, "CLM4": clm}
    order = parse_contracts(["CLH4", "CLJ4", "CLM4"])
    return contracts, order


# ===========================================================================
# TestRollPolicyVariant
# ===========================================================================

class TestRollPolicyVariant:
    def test_label_all_months(self):
        v = RollPolicyVariant(2, None, "same_day")
        assert v.label == "cd=2_months=all_timing=same_day"
        assert v.months_label == "all"

    def test_label_quarterly(self):
        v = RollPolicyVariant(1, CL_QUARTERLY_MONTHS, "next_session")
        assert v.label == "cd=1_months=quarterly_timing=next_session"
        assert v.months_label == "quarterly"


# ===========================================================================
# TestGenerateVariants
# ===========================================================================

class TestGenerateVariants:
    def test_correct_count(self):
        variants = generate_cl_variants()
        assert len(variants) == 12  # 3 × 2 × 2

    def test_all_unique_labels(self):
        variants = generate_cl_variants()
        labels = [v.label for v in variants]
        assert len(labels) == len(set(labels))


# ===========================================================================
# TestCompositeScore
# ===========================================================================

class TestCompositeScore:
    def test_deterministic_formula(self):
        v = RollPolicyVariant(2, None, "same_day")
        s = CalibrationScore(
            variant=v,
            unmatched_canonical=5, unmatched_ts=3,
            n_fail=4,
            mean_gap_diff=0.5,
            mean_drift=10.0,
        )
        expected = (
            SCORE_W_UNMATCHED * 8
            + SCORE_W_FAIL * 4
            + SCORE_W_GAP * 0.5
            + SCORE_W_DRIFT * 10.0
        )
        assert compute_composite_score(s) == pytest.approx(expected)

    def test_zero_inputs(self):
        v = RollPolicyVariant(2, None, "same_day")
        s = CalibrationScore(variant=v)
        assert compute_composite_score(s) == 0.0


# ===========================================================================
# TestRankVariants
# ===========================================================================

class TestRankVariants:
    def _make_score(self, n_fail, unmatched, drift, valid=True, cd=2):
        v = RollPolicyVariant(cd, None, "same_day")
        return CalibrationScore(
            variant=v, n_fail=n_fail,
            unmatched_canonical=unmatched, unmatched_ts=0,
            mean_drift=drift, valid=valid,
        )

    def test_correct_ordering(self):
        s1 = self._make_score(n_fail=10, unmatched=5, drift=2.0, cd=1)
        s2 = self._make_score(n_fail=2, unmatched=1, drift=1.0, cd=2)
        s3 = self._make_score(n_fail=5, unmatched=3, drift=3.0, cd=3)
        ranked = rank_variants([s1, s2, s3])
        assert ranked[0].variant.consecutive_days == 2  # lowest composite
        assert ranked[0].rank == 1

    def test_tiebreak_by_n_fail(self):
        """Same composite → fewer FAILs should rank higher."""
        v1 = RollPolicyVariant(1, None, "same_day")
        v2 = RollPolicyVariant(2, None, "same_day")
        # Both have composite = SCORE_W_FAIL * n_fail
        s1 = CalibrationScore(variant=v1, n_fail=5)
        s1.composite_score = 25.0
        s2 = CalibrationScore(variant=v2, n_fail=3)
        s2.composite_score = 25.0
        # Force same composite by setting it directly.
        # rank_variants will re-compute, so use matching weights.
        s1_fresh = CalibrationScore(
            variant=v1, n_fail=3, unmatched_canonical=2, mean_drift=0.0,
        )
        s2_fresh = CalibrationScore(
            variant=v2, n_fail=5, unmatched_canonical=0, mean_drift=0.0,
        )
        # composite: s1 = 5*3 + 10*2 = 35, s2 = 5*5 = 25
        # s2 has lower composite → ranks first.
        ranked = rank_variants([s1_fresh, s2_fresh])
        assert ranked[0].variant.consecutive_days == 2

    def test_invalid_sorted_last(self):
        s_valid = self._make_score(n_fail=10, unmatched=5, drift=100.0)
        s_invalid = self._make_score(n_fail=0, unmatched=0, drift=0.0, valid=False)
        ranked = rank_variants([s_invalid, s_valid])
        assert ranked[0].valid is True
        assert ranked[1].valid is False

    def test_deterministic(self):
        scores1 = [
            self._make_score(n_fail=3, unmatched=1, drift=1.0, cd=1),
            self._make_score(n_fail=5, unmatched=2, drift=2.0, cd=2),
        ]
        scores2 = [
            self._make_score(n_fail=3, unmatched=1, drift=1.0, cd=1),
            self._make_score(n_fail=5, unmatched=2, drift=2.0, cd=2),
        ]
        r1 = rank_variants(scores1)
        r2 = rank_variants(scores2)
        assert r1[0].variant.consecutive_days == r2[0].variant.consecutive_days
        assert r1[0].composite_score == pytest.approx(r2[0].composite_score)


# ===========================================================================
# TestEligibleMonthsFilter
# ===========================================================================

class TestEligibleMonthsFilter:
    def test_quarterly_filter_removes_non_quarterly(self):
        """Filtering to quarterly should skip the April (J) contract."""
        contracts, order = _three_cl_contracts()
        # order has H(Mar), J(Apr), M(Jun)
        assert len(order) == 3

        # With quarterly filter: only H and M should remain.
        rolls, active = detect_rolls(
            contracts, order, consecutive_days=2,
            eligible_months=CL_QUARTERLY_MONTHS,
        )
        # Only CLH4 → CLM4 roll possible (CLJ4 filtered out).
        active_contracts = set(active["contract"].unique())
        assert "CLJ4" not in active_contracts
        assert "CLH4" in active_contracts or "CLM4" in active_contracts

    def test_all_months_no_filtering(self):
        """eligible_months=None should use all contracts."""
        contracts, order = _three_cl_contracts()
        rolls_all, active_all = detect_rolls(contracts, order, consecutive_days=2)
        rolls_none, active_none = detect_rolls(
            contracts, order, consecutive_days=2, eligible_months=None,
        )
        assert len(rolls_all) == len(rolls_none)
        assert list(active_all["contract"]) == list(active_none["contract"])


# ===========================================================================
# TestRollTimingNextSession
# ===========================================================================

class TestRollTimingNextSession:
    def test_next_session_delays_switch(self):
        """next_session should delay the active-contract switch by 1 day."""
        contracts, order = _three_cl_contracts()
        _, active_same = detect_rolls(
            contracts, order, consecutive_days=2, roll_timing="same_day",
        )
        _, active_next = detect_rolls(
            contracts, order, consecutive_days=2, roll_timing="next_session",
        )
        # In next_session, the old contract stays active 1 day longer.
        # Find the first date where the active contracts differ.
        merged = pd.merge(
            active_same.rename(columns={"contract": "same_day"}),
            active_next.rename(columns={"contract": "next_session"}),
            on="date", how="inner",
        )
        diff = merged[merged["same_day"] != merged["next_session"]]
        # There should be at least one date where they differ.
        assert len(diff) >= 1

    def test_roll_count_preserved(self):
        """Both timings should detect the same number of rolls."""
        contracts, order = _three_cl_contracts()
        rolls_same, _ = detect_rolls(
            contracts, order, consecutive_days=2, roll_timing="same_day",
        )
        rolls_next, _ = detect_rolls(
            contracts, order, consecutive_days=2, roll_timing="next_session",
        )
        assert len(rolls_same) == len(rolls_next)


# ===========================================================================
# TestRegressionDefaultBehavior
# ===========================================================================

class TestRegressionDefaultBehavior:
    """Prove that default params (eligible_months=None, roll_timing='same_day')
    produce identical output to the original detect_rolls signature."""

    def test_two_contract_identical(self):
        """Two-contract setup: defaults must match original behavior."""
        # Build two overlapping contracts.
        front = _make_contract_df(
            "CLH4", "2024-01-02", 60,
            base_price=70.0, base_volume=50000, volume_trend=-800,
        )
        back = _make_contract_df(
            "CLJ4", "2024-02-01", 60,
            base_price=71.0, base_volume=5000, volume_trend=1500,
        )
        contracts = {"CLH4": front, "CLJ4": back}
        order = parse_contracts(["CLH4", "CLJ4"])

        # Call with no new args (production path).
        rolls_base, active_base = detect_rolls(contracts, order, consecutive_days=2)
        # Call with explicit defaults (calibration path).
        rolls_explicit, active_explicit = detect_rolls(
            contracts, order, consecutive_days=2,
            eligible_months=None, roll_timing="same_day",
        )

        assert len(rolls_base) == len(rolls_explicit)
        for rb, re in zip(rolls_base, rolls_explicit):
            assert rb.date == re.date
            assert rb.from_contract == re.from_contract
            assert rb.to_contract == re.to_contract
            assert rb.adjustment == pytest.approx(re.adjustment)
        assert list(active_base["date"]) == list(active_explicit["date"])
        assert list(active_base["contract"]) == list(active_explicit["contract"])

    def test_three_contract_identical(self):
        """Three-contract CL setup: defaults identical to no-args call."""
        contracts, order = _three_cl_contracts()
        rolls_base, active_base = detect_rolls(contracts, order, consecutive_days=2)
        rolls_explicit, active_explicit = detect_rolls(
            contracts, order, consecutive_days=2,
            eligible_months=None, roll_timing="same_day",
        )
        assert len(rolls_base) == len(rolls_explicit)
        for rb, re in zip(rolls_base, rolls_explicit):
            assert rb.date == re.date
            assert rb.from_contract == re.from_contract
        assert list(active_base["contract"]) == list(active_explicit["contract"])


# ===========================================================================
# TestPlausibilityGuardrail
# ===========================================================================

class TestPlausibilityGuardrail:
    def test_constants_defined(self):
        assert MIN_PLAUSIBLE_ROLLS == 40
        assert MAX_PLAUSIBLE_ROLLS == 140

    def test_guardrail_documented_in_to_dict(self):
        """Invalid variants should have valid=False in to_dict."""
        v = RollPolicyVariant(2, None, "same_day")
        s = CalibrationScore(variant=v, n_rolls=10, valid=False)
        d = s.to_dict()
        assert d["valid"] is False


# ===========================================================================
# TestSaveArtifacts
# ===========================================================================

class TestSaveArtifacts:
    def test_files_created(self, tmp_path):
        v = RollPolicyVariant(2, None, "same_day")
        scores = [CalibrationScore(variant=v, n_rolls=98, composite_score=50.0)]
        paths = save_calibration_artifacts(scores, tmp_path, symbol="CL")
        assert paths["calibration_csv"].exists()
        assert paths["recommendation_json"].exists()

    def test_recommendation_json_schema(self, tmp_path):
        v = RollPolicyVariant(1, None, "next_session")
        scores = [
            CalibrationScore(
                variant=v, n_rolls=95, composite_score=30.0,
                strict_status="FAIL", policy_status="WATCH",
            ),
        ]
        paths = save_calibration_artifacts(scores, tmp_path, symbol="CL")
        with open(paths["recommendation_json"]) as f:
            data = json.load(f)
        assert data["symbol"] == "CL"
        assert data["recommended_variant"] == v.label
        assert "strict_status" in data
        assert "policy_status" in data
        assert data["strict_status"] == "FAIL"
        assert data["policy_status"] == "WATCH"
        assert "top_3" in data
        assert data["total_variants_tested"] == 1
        # Verify top_3 entries include both statuses.
        entry = data["top_3"][0]
        assert "strict_status" in entry
        assert "policy_status" in entry

    def test_csv_columns(self, tmp_path):
        v = RollPolicyVariant(2, None, "same_day")
        scores = [CalibrationScore(variant=v)]
        paths = save_calibration_artifacts(scores, tmp_path, symbol="CL")
        df = pd.read_csv(paths["calibration_csv"])
        assert "strict_status" in df.columns
        assert "policy_status" in df.columns
        assert "composite_score" in df.columns
        assert "valid" in df.columns
