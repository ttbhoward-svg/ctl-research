"""Normalization modes for parity inputs (Data Cutover Task H.4).

Enforces explicit adjustment-basis declarations so that parity
comparisons never silently mix split-adjusted and unadjusted data.

Supported modes:

- ``"raw"``                  — pass-through; schema coercion only.
- ``"split_adjusted"``       — apply split factor column (equities/ETFs).
- ``"total_return_adjusted"``— reserved; raises ``NotImplementedError``.

Supported asset classes: ``"futures"``, ``"equity"``, ``"etf"``.

See docs/governance/cutover_h2_h3_decision_log.md for policy rationale.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

NormalizationMode = Literal["raw", "split_adjusted", "total_return_adjusted"]

AssetClass = Literal["futures", "equity", "etf"]

#: Canonical output columns (always in this order).
CANONICAL_COLUMNS = ("Date", "Open", "High", "Low", "Close", "Volume")

#: Recognised volume column aliases.
_VOLUME_ALIASES = {"vol", "volume", "vol.", "tvol"}

#: OHLC price columns that receive split adjustment.
_PRICE_COLUMNS = ("Open", "High", "Low", "Close")

#: Case-insensitive mapping for OHLC columns (lowercase → canonical).
_OHLC_CANONICAL = {"open": "Open", "high": "High", "low": "Low", "close": "Close"}


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def coerce_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the ``Date`` column to tz-naive ``datetime64[ns]``.

    - Accepts ``Date``, ``date``, ``DATE``, ``Datetime``, ``datetime``, ``timestamp``, ``ts_event``.
    - Strips timezone info (converts UTC → naive).
    - Truncates to midnight (date-only).

    Returns a copy with the normalised ``Date`` column.
    """
    df = df.copy()

    # Find the date column (case-insensitive search).
    date_col = None
    for col in df.columns:
        if col.lower() in ("date", "datetime", "timestamp", "ts_event"):
            date_col = col
            break

    if date_col is None:
        raise ValueError(
            f"No date column found.  Columns present: {list(df.columns)}. "
            "Expected one of: Date, date, Datetime, timestamp, ts_event."
        )

    # Parse and normalise.
    parsed = pd.to_datetime(df[date_col], utc=False)
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_localize(None)
    parsed = parsed.dt.normalize()  # midnight

    if date_col != "Date":
        df = df.drop(columns=[date_col])
    df["Date"] = parsed
    return df


def coerce_volume_column(df: pd.DataFrame) -> pd.DataFrame:
    """Rename volume-like columns to canonical ``Volume``.

    Returns a copy.
    """
    df = df.copy()
    if "Volume" in df.columns:
        return df
    for col in df.columns:
        if col.lower() in _VOLUME_ALIASES:
            df = df.rename(columns={col: "Volume"})
            return df
    return df


def coerce_ohlc_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename case-variant OHLC columns to canonical form.

    Handles ``open`` → ``Open``, ``high`` → ``High``, etc.
    Returns a copy.
    """
    df = df.copy()
    rename_map = {}
    for col in df.columns:
        low = col.lower()
        if low in _OHLC_CANONICAL and col != _OHLC_CANONICAL[low]:
            rename_map[col] = _OHLC_CANONICAL[low]
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def validate_ohlcv_schema(df: pd.DataFrame) -> None:
    """Validate that *df* has all canonical OHLCV columns.

    Raises
    ------
    ValueError
        With an actionable message listing missing columns.
    """
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Present columns: {list(df.columns)}. "
            f"Expected: {list(CANONICAL_COLUMNS)}."
        )


# ---------------------------------------------------------------------------
# Core normaliser
# ---------------------------------------------------------------------------

def normalize_ohlcv(
    df: pd.DataFrame,
    asset_class: AssetClass,
    mode: NormalizationMode = "raw",
    source: str = "",
) -> pd.DataFrame:
    """Normalise an OHLCV DataFrame to canonical schema and adjustment basis.

    Parameters
    ----------
    df : pd.DataFrame
        Input OHLCV data.
    asset_class : AssetClass
        ``"futures"``, ``"equity"``, or ``"etf"``.
    mode : NormalizationMode
        ``"raw"`` — schema coercion only (pass-through prices).
        ``"split_adjusted"`` — apply ``split_factor`` column to OHLC
        (equities/ETFs only).
        ``"total_return_adjusted"`` — reserved.
    source : str
        Provider label for error messages (e.g. ``"databento"``).

    Returns
    -------
    pd.DataFrame
        With columns ``Date, Open, High, Low, Close, Volume`` in canonical
        order, ``Date`` tz-naive at midnight.

    Raises
    ------
    ValueError
        If validation fails or incompatible mode/asset_class is requested.
    NotImplementedError
        If ``total_return_adjusted`` mode is requested.
    """
    # ---- Mode guards ----
    if mode == "total_return_adjusted":
        raise NotImplementedError(
            "total_return_adjusted mode is not yet implemented.  "
            "Use 'raw' or 'split_adjusted'."
        )

    if mode == "split_adjusted" and asset_class == "futures":
        raise ValueError(
            f"Cannot apply split_adjusted mode to futures data "
            f"(source={source!r}).  Futures use 'raw' (continuous "
            f"back-adjustment is handled separately by the roll pipeline)."
        )

    # ---- Schema coercion ----
    out = coerce_date_column(df)
    out = coerce_ohlc_columns(out)
    out = coerce_volume_column(out)

    # ---- Split adjustment (equities/ETFs only) ----
    if mode == "split_adjusted":
        if "split_factor" not in out.columns:
            raise ValueError(
                f"split_adjusted mode requires a 'split_factor' column, "
                f"but it is missing (source={source!r}).  "
                f"Columns present: {list(out.columns)}.  "
                f"Provide explicit split factors or use mode='raw'."
            )
        factor = out["split_factor"].astype(float)
        for col in _PRICE_COLUMNS:
            if col in out.columns:
                out[col] = out[col] * factor

    # ---- Final validation ----
    validate_ohlcv_schema(out)

    # ---- Canonical column order + types ----
    out = out[list(CANONICAL_COLUMNS)].copy()
    out = out.sort_values("Date").reset_index(drop=True)
    for col in _PRICE_COLUMNS:
        out[col] = out[col].astype(float)
    out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce").fillna(0).astype(float)

    return out
