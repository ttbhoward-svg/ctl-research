"""DataProvider interface and canonical schema enforcement.

Every data provider must subclass ``DataProvider`` and return
DataFrames conforming to the canonical schema defined here.

Canonical columns
-----------------
- ``timestamp``   : datetime64[ns, UTC] — bar open time, timezone-aware UTC
- ``Open``        : float64
- ``High``        : float64
- ``Low``         : float64
- ``Close``       : float64
- ``Volume``      : float64
- ``symbol``      : str — normalised symbol (e.g. "/ES", "XLE")
- ``timeframe``   : str — "1D", "1W", "1M", "4H", etc.
- ``provider``    : str — originating provider name
- ``session_type``: str — "electronic", "pit", "combined", "regular"
- ``roll_method`` : str — "back_adjusted", "front_month", "continuous", "none"
- ``close_type``  : str — "settlement", "last_trade", "auction", "unknown"
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Canonical schema
# ---------------------------------------------------------------------------

CANONICAL_COLUMNS: List[str] = [
    "timestamp",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "symbol",
    "timeframe",
    "provider",
    "session_type",
    "roll_method",
    "close_type",
]

#: Price columns that must be float64.
_PRICE_COLS = ("Open", "High", "Low", "Close", "Volume")

#: Valid values for constrained string columns.
SESSION_TYPES = ("electronic", "pit", "combined", "regular")
ROLL_METHODS = ("back_adjusted", "front_month", "continuous", "none")
CLOSE_TYPES = ("settlement", "last_trade", "auction", "unknown")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

@dataclass
class ProviderMeta:
    """Metadata attached to every provider response."""

    provider: str
    session_type: str = "electronic"
    roll_method: str = "back_adjusted"
    close_type: str = "settlement"

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.session_type not in SESSION_TYPES:
            errors.append(
                f"session_type '{self.session_type}' not in {SESSION_TYPES}"
            )
        if self.roll_method not in ROLL_METHODS:
            errors.append(
                f"roll_method '{self.roll_method}' not in {ROLL_METHODS}"
            )
        if self.close_type not in CLOSE_TYPES:
            errors.append(
                f"close_type '{self.close_type}' not in {CLOSE_TYPES}"
            )
        return errors


# ---------------------------------------------------------------------------
# Schema validation / normalisation
# ---------------------------------------------------------------------------

def validate_canonical(df: pd.DataFrame) -> List[str]:
    """Validate that *df* conforms to the canonical OHLCV schema.

    Returns a list of error strings (empty if valid).
    """
    errors: List[str] = []

    # Column presence.
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")
        return errors  # can't validate further

    # Timestamp dtype.
    ts = df["timestamp"]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        errors.append(
            f"timestamp dtype is {ts.dtype}, expected datetime64"
        )
    elif ts.dt.tz is None:
        errors.append("timestamp must be timezone-aware (UTC)")

    # Price columns dtype.
    for col in _PRICE_COLS:
        if not pd.api.types.is_float_dtype(df[col]):
            errors.append(f"{col} dtype is {df[col].dtype}, expected float64")

    # OHLC sanity.
    if len(df) > 0:
        if (df["High"] < df["Low"]).any():
            errors.append("High < Low on one or more bars")
        if (df["High"] < df["Open"]).any():
            errors.append("High < Open on one or more bars")
        if (df["High"] < df["Close"]).any():
            errors.append("High < Close on one or more bars")
        if (df["Low"] > df["Open"]).any():
            errors.append("Low > Open on one or more bars")
        if (df["Low"] > df["Close"]).any():
            errors.append("Low > Close on one or more bars")

    # Constrained string columns.
    for col, valid in [
        ("session_type", SESSION_TYPES),
        ("roll_method", ROLL_METHODS),
        ("close_type", CLOSE_TYPES),
    ]:
        vals = df[col].dropna().unique()
        bad = [v for v in vals if v not in valid]
        if bad:
            errors.append(f"{col} contains invalid values: {bad}")

    return errors


def normalize_to_canonical(
    df: pd.DataFrame,
    meta: ProviderMeta,
    symbol: str,
    timeframe: str,
    timestamp_col: str = "Date",
    tz: str = "UTC",
) -> pd.DataFrame:
    """Normalize a raw OHLCV DataFrame to canonical schema.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with at least Open/High/Low/Close/Volume and a date column.
    meta : ProviderMeta
        Provider metadata to stamp on every row.
    symbol : str
        Normalised symbol name.
    timeframe : str
        Timeframe label (e.g. "1D").
    timestamp_col : str
        Name of the source date/time column.
    tz : str
        Timezone to localize naive timestamps to.

    Returns
    -------
    pd.DataFrame
        Canonical schema DataFrame.

    Raises
    ------
    ValueError
        If required price columns are missing or metadata is invalid.
    """
    meta_errors = meta.validate()
    if meta_errors:
        raise ValueError(f"Invalid provider metadata: {meta_errors}")

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing price columns: {missing}")
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found")

    out = pd.DataFrame()

    # Timestamp: ensure datetime64[ns, UTC].
    ts = pd.to_datetime(df[timestamp_col])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(tz)
    else:
        ts = ts.dt.tz_convert("UTC")
    out["timestamp"] = ts.reset_index(drop=True)

    # Price columns.
    for col in required:
        out[col] = df[col].values.astype(np.float64)

    # Metadata columns.
    n = len(df)
    out["symbol"] = symbol
    out["timeframe"] = timeframe
    out["provider"] = meta.provider
    out["session_type"] = meta.session_type
    out["roll_method"] = meta.roll_method
    out["close_type"] = meta.close_type

    out = out[CANONICAL_COLUMNS]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Abstract provider
# ---------------------------------------------------------------------------

class DataProvider(ABC):
    """Abstract base class for all data providers.

    Subclasses must implement ``get_ohlcv`` to fetch and return
    canonical-schema DataFrames.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short provider identifier (e.g. ``"databento"``)."""

    @abstractmethod
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a single symbol.

        Parameters
        ----------
        symbol : str
            Normalised symbol (e.g. ``"/ES"``, ``"XLE"``).
        timeframe : str
            Timeframe label (e.g. ``"1D"``, ``"1W"``).
        start : str
            Start date (inclusive), ISO format ``"YYYY-MM-DD"``.
        end : str
            End date (inclusive), ISO format ``"YYYY-MM-DD"``.

        Returns
        -------
        pd.DataFrame
            Canonical schema DataFrame.  Must pass ``validate_canonical``.
        """

    def get_ohlcv_multi(
        self,
        symbols: Sequence[str],
        timeframe: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch OHLCV for multiple symbols (default: serial calls).

        Subclasses may override for batch-optimised fetching.
        """
        frames = []
        for sym in symbols:
            frames.append(self.get_ohlcv(sym, timeframe, start, end))
        if not frames:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)
        return pd.concat(frames, ignore_index=True)
