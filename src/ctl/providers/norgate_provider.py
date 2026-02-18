"""Norgate Data provider stub.

This is a placeholder implementation. The ``get_ohlcv`` method raises
``NotImplementedError`` until the Norgate Data Updater and ``norgatedata``
Python package are configured.

When implemented, this provider will:
- Use the ``norgatedata`` package for local database access
- Fetch futures continuous contracts and equity OHLCV
- Map CTL symbols to Norgate symbol conventions
- Return back-adjusted combined-session last-trade bars
"""

from __future__ import annotations

import pandas as pd

from ctl.providers.base import DataProvider, ProviderMeta


class NorgateProvider(DataProvider):
    """Norgate Data local database provider (stub).

    Parameters
    ----------
    database_path : str, optional
        Path to Norgate local database. Defaults to system install location.
    """

    def __init__(self, database_path: str = "") -> None:
        self._database_path = database_path
        self._meta = ProviderMeta(
            provider="norgate",
            session_type="combined",
            roll_method="back_adjusted",
            close_type="last_trade",
        )

    @property
    def name(self) -> str:
        return "norgate"

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch OHLCV from Norgate local database.

        Raises
        ------
        NotImplementedError
            Always — this is a stub awaiting Norgate integration.
        """
        raise NotImplementedError(
            f"NorgateProvider.get_ohlcv not yet implemented. "
            f"Requested: {symbol} {timeframe} {start}–{end}"
        )
