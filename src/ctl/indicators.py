"""Technical indicator calculations for the B1 pipeline.

All formulas match B1_Strategy_Logic_Spec_v2.md Section 1.3-1.4.
Pandas implementations chosen so that:
  - EMA uses ewm(span=N, adjust=False)  -> matches TradeStation XAverage()
  - SMA uses rolling(N).mean()
  - ATR uses Wilder smoothing (ewm(alpha=1/N, adjust=False))
"""

from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average.  Seed = first value, then recursive.

    Spec §1.3: multiplier = 2/(period+1), EMA[0]=Close[0].
    pandas ewm(span=period, adjust=False) produces this exactly.
    """
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(period).mean()


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range using Wilder smoothing.

    TR = max(H-L, |H-prevC|, |L-prevC|)
    ATR = Wilder EMA of TR = ewm(alpha=1/period, adjust=False)
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
) -> pd.Series:
    """Williams %R.  Range: -100 (oversold) to 0 (overbought).

    Spec §1.4: ((Highest(High,10) - Close) / (Highest(High,10) - Lowest(Low,10))) * -100
    """
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    denom = hh - ll
    # Avoid division by zero on flat bars.
    denom = denom.replace(0, float("nan"))
    return ((hh - close) / denom) * -100.0


def slope_pct(ema_series: pd.Series, lookback: int) -> pd.Series:
    """Percentage change in an EMA over *lookback* bars.

    Spec §2 C1: Slope_20 = ((EMA10[N] / EMA10[N-20]) - 1) * 100
    """
    return (ema_series / ema_series.shift(lookback) - 1.0) * 100.0
