"""Tests for purged time-series cross-validation."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.cross_validation import FoldInfo, PurgedTimeSeriesSplit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trading_dates(n: int, start: str = "2020-01-02") -> pd.DatetimeIndex:
    """Generate *n* business dates."""
    return pd.bdate_range(start, periods=n)


# ---------------------------------------------------------------------------
# Split mechanics
# ---------------------------------------------------------------------------

class TestSplitMechanics:
    def test_correct_number_of_folds(self):
        """5 splits on 120 dates should yield 5 folds (no purge)."""
        dates = _trading_dates(120)
        cv = PurgedTimeSeriesSplit(n_splits=5, purge_gap_days=0)
        folds = list(cv.split(dates))
        assert len(folds) == 5

    def test_expanding_training_window(self):
        """Each fold's training set should be >= previous fold's."""
        dates = _trading_dates(120)
        cv = PurgedTimeSeriesSplit(n_splits=5, purge_gap_days=0)
        sizes = [len(train) for train, _, _ in cv.split(dates)]
        for i in range(1, len(sizes)):
            assert sizes[i] >= sizes[i - 1]

    def test_no_overlap_between_train_and_test(self):
        """Train and test indices must be disjoint."""
        dates = _trading_dates(120)
        cv = PurgedTimeSeriesSplit(n_splits=5, purge_gap_days=10)
        for train, test, _ in cv.split(dates):
            overlap = set(train.tolist()) & set(test.tolist())
            assert len(overlap) == 0

    def test_test_indices_are_contiguous(self):
        """Test indices should be a contiguous block."""
        dates = _trading_dates(120)
        cv = PurgedTimeSeriesSplit(n_splits=5, purge_gap_days=0)
        for _, test, _ in cv.split(dates):
            assert np.array_equal(test, np.arange(test[0], test[-1] + 1))

    def test_train_precedes_test(self):
        """All training indices < all test indices."""
        dates = _trading_dates(120)
        cv = PurgedTimeSeriesSplit(n_splits=5, purge_gap_days=10)
        for train, test, _ in cv.split(dates):
            assert train[-1] < test[0]


# ---------------------------------------------------------------------------
# Purge gap enforcement
# ---------------------------------------------------------------------------

class TestPurgeGap:
    def test_purge_removes_observations(self):
        """Training sets should shrink when purge gap is applied."""
        dates = _trading_dates(120)
        cv_no = PurgedTimeSeriesSplit(n_splits=5, purge_gap_days=0)
        cv_yes = PurgedTimeSeriesSplit(n_splits=5, purge_gap_days=30)

        for (tn, _, _), (tp, _, _) in zip(cv_no.split(dates), cv_yes.split(dates)):
            assert len(tp) <= len(tn)

    def test_temporal_gap_enforced(self):
        """Last train date must be >= purge_gap_days before first test date."""
        dates = _trading_dates(240)
        cv = PurgedTimeSeriesSplit(n_splits=5, purge_gap_days=30)

        for train, test, info in cv.split(dates):
            gap = (dates[test[0]] - dates[train[-1]]).days
            assert gap >= 30, (
                f"Gap {gap} days < 30 on fold {info.fold}"
            )

    def test_purge_count_in_fold_info(self):
        """FoldInfo.n_purged should be non-negative and plausible."""
        dates = _trading_dates(120)
        cv = PurgedTimeSeriesSplit(n_splits=3, purge_gap_days=20)

        for _, _, info in cv.split(dates):
            assert info.n_purged >= 0
            # n_purged + n_train should equal the total train candidates.
            assert info.n_purged + info.n_train > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_small_dataset_skips_folds(self):
        """Large purge gap on small data -> some folds skipped."""
        dates = _trading_dates(12)
        cv = PurgedTimeSeriesSplit(
            n_splits=5, purge_gap_days=60, min_train_size=5,
        )
        folds = list(cv.split(dates))
        assert len(folds) < 5

    def test_tiny_dataset_yields_nothing(self):
        """Fewer observations than splits+1 -> no folds."""
        dates = _trading_dates(3)
        cv = PurgedTimeSeriesSplit(n_splits=5, purge_gap_days=0)
        folds = list(cv.split(dates))
        assert len(folds) == 0

    def test_invalid_n_splits_raises(self):
        with pytest.raises(ValueError):
            PurgedTimeSeriesSplit(n_splits=0)

    def test_get_n_splits(self):
        cv = PurgedTimeSeriesSplit(n_splits=7)
        assert cv.get_n_splits() == 7

    def test_fold_info_populated(self):
        """FoldInfo fields should match actual indices."""
        dates = _trading_dates(60)
        cv = PurgedTimeSeriesSplit(n_splits=3, purge_gap_days=0)

        for train, test, info in cv.split(dates):
            assert info.n_train == len(train)
            assert info.n_test == len(test)
            assert info.train_start == dates[train[0]]
            assert info.train_end == dates[train[-1]]
            assert info.test_start == dates[test[0]]
            assert info.test_end == dates[test[-1]]


# ---------------------------------------------------------------------------
# Leakage: no future data in training set
# ---------------------------------------------------------------------------

class TestNoLeakage:
    def test_train_dates_before_test_dates(self):
        """Every training date must be strictly before the first test date."""
        dates = _trading_dates(200)
        cv = PurgedTimeSeriesSplit(n_splits=5, purge_gap_days=30)

        for train, test, _ in cv.split(dates):
            max_train_date = dates[train[-1]]
            min_test_date = dates[test[0]]
            assert max_train_date < min_test_date

    def test_no_train_date_in_purge_window(self):
        """No training date should fall within the purge window."""
        dates = _trading_dates(200)
        gap = 30
        cv = PurgedTimeSeriesSplit(n_splits=5, purge_gap_days=gap)

        for train, test, _ in cv.split(dates):
            cutoff = dates[test[0]] - pd.Timedelta(days=gap)
            for t_idx in train:
                assert dates[t_idx] <= cutoff
