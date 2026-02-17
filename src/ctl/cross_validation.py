"""Purged walk-forward time-series cross-validation.

Implements purged CV per locked spec:
  - model.yaml: 5 folds, 30-day purge gap
  - Phase Gate Checklist v2: purge >= max holding period
  - Synthesis v2: TimeSeriesSplit with 30-day purge gap

The purge removes training observations whose dates fall within the
purge gap of the test window start, preventing look-ahead bias from
overlapping trade holding periods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Fold metadata
# ---------------------------------------------------------------------------

@dataclass
class FoldInfo:
    """Metadata for one CV fold."""

    fold: int
    n_train: int
    n_purged: int
    n_test: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


# ---------------------------------------------------------------------------
# Purged time-series splitter
# ---------------------------------------------------------------------------

class PurgedTimeSeriesSplit:
    """Walk-forward expanding-window CV with purge gap.

    Divides the data into ``n_splits + 1`` contiguous blocks.  Fold *k*
    trains on blocks 0..k (expanding) and tests on block k+1.  Training
    observations within ``purge_gap_days`` calendar days of the first
    test observation are removed to prevent leakage from overlapping
    holding periods.

    Parameters
    ----------
    n_splits : int
        Number of folds (default 5 per model.yaml).
    purge_gap_days : int
        Calendar days to purge from training end before test start
        (default 30 per phase1a.yaml).
    min_train_size : int
        Minimum training samples after purge; fold skipped if not met.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap_days: int = 30,
        min_train_size: int = 10,
    ):
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        self.n_splits = n_splits
        self.purge_gap_days = purge_gap_days
        self.min_train_size = min_train_size

    def split(
        self,
        dates: pd.Series | pd.DatetimeIndex | np.ndarray,
    ) -> Generator[Tuple[np.ndarray, np.ndarray, FoldInfo], None, None]:
        """Yield (train_indices, test_indices, fold_info) per fold.

        Parameters
        ----------
        dates : array-like of datetime
            Sorted ascending.  One per observation (e.g., trade trigger
            date).
        """
        dates = pd.DatetimeIndex(dates)
        n = len(dates)
        block_size = n // (self.n_splits + 1)

        if block_size < 1:
            return

        for fold in range(self.n_splits):
            test_start = (fold + 1) * block_size
            test_end = (fold + 2) * block_size if fold < self.n_splits - 1 else n

            test_idx = np.arange(test_start, test_end)

            # Purge: remove training obs within purge_gap of test start.
            first_test_date = dates[test_start]
            purge_cutoff = first_test_date - pd.Timedelta(days=self.purge_gap_days)

            all_train = np.arange(0, test_start)
            train_mask = dates[all_train] <= purge_cutoff
            train_idx = all_train[train_mask]

            n_purged = int(np.sum(~train_mask))

            if len(train_idx) < self.min_train_size:
                continue

            info = FoldInfo(
                fold=fold,
                n_train=len(train_idx),
                n_purged=n_purged,
                n_test=len(test_idx),
                train_start=dates[train_idx[0]],
                train_end=dates[train_idx[-1]],
                test_start=dates[test_idx[0]],
                test_end=dates[test_idx[-1]],
            )

            yield train_idx, test_idx, info

    def get_n_splits(self) -> int:
        """Return the configured number of folds (some may be skipped)."""
        return self.n_splits
