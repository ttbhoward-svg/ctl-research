"""Overlap window computation and enforcement for parity runs (H.5).

Provides utilities to compute the date intersection between two OHLCV
DataFrames, align both to that intersection, and validate that the
overlap meets minimum bar-count requirements.

This prevents coverage-mismatch contamination in cross-provider
parity comparisons where one series may have a longer or shorter
date range than the other.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd


def compute_overlap_window(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    date_col: str = "Date",
) -> Tuple[pd.Timestamp, pd.Timestamp, int]:
    """Return (start, end, n_overlap) for the date intersection.

    Parameters
    ----------
    df_a, df_b : pd.DataFrame
        DataFrames each containing a ``date_col`` column.
    date_col : str
        Column name for dates (default ``"Date"``).

    Returns
    -------
    tuple of (pd.Timestamp, pd.Timestamp, int)
        ``(overlap_start, overlap_end, n_overlap_bars)``.
        If there is no overlap, returns ``(NaT, NaT, 0)``.
    """
    if df_a.empty or df_b.empty:
        return (pd.NaT, pd.NaT, 0)

    dates_a = pd.to_datetime(df_a[date_col])
    dates_b = pd.to_datetime(df_b[date_col])

    overlap_start = max(dates_a.min(), dates_b.min())
    overlap_end = min(dates_a.max(), dates_b.max())

    if overlap_start > overlap_end:
        return (pd.NaT, pd.NaT, 0)

    # Count actual overlapping dates via inner join.
    set_a = set(dates_a)
    set_b = set(dates_b)
    n_overlap = len(set_a & set_b)

    if n_overlap == 0:
        return (pd.NaT, pd.NaT, 0)

    return (overlap_start, overlap_end, n_overlap)


def align_to_overlap(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    date_col: str = "Date",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter both frames to their overlapping date range.

    Parameters
    ----------
    df_a, df_b : pd.DataFrame
        DataFrames each containing a ``date_col`` column.
    date_col : str
        Column name for dates (default ``"Date"``).

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        Copies of the input frames filtered to ``[overlap_start, overlap_end]``.
        If there is no overlap, returns empty DataFrames (preserving columns).
    """
    start, end, n_overlap = compute_overlap_window(df_a, df_b, date_col)

    if n_overlap == 0:
        return (
            df_a.iloc[0:0].copy(),
            df_b.iloc[0:0].copy(),
        )

    dates_a = pd.to_datetime(df_a[date_col])
    dates_b = pd.to_datetime(df_b[date_col])

    mask_a = (dates_a >= start) & (dates_a <= end)
    mask_b = (dates_b >= start) & (dates_b <= end)

    return (
        df_a.loc[mask_a].copy().reset_index(drop=True),
        df_b.loc[mask_b].copy().reset_index(drop=True),
    )


def validate_min_overlap(n_overlap: int, min_bars: int) -> None:
    """Raise ValueError if overlap is insufficient.

    Parameters
    ----------
    n_overlap : int
        Number of overlapping bars.
    min_bars : int
        Minimum required overlap bars.

    Raises
    ------
    ValueError
        If ``n_overlap < min_bars``.
    """
    if n_overlap < min_bars:
        raise ValueError(
            f"Insufficient overlap: {n_overlap} bars, "
            f"minimum required is {min_bars}."
        )
