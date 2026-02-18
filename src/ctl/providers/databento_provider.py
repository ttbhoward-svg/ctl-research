"""Databento data provider stub.

This is a placeholder implementation. The ``get_ohlcv`` method raises
``NotImplementedError`` until a Databento API key and client are configured.

When implemented, this provider will:
- Connect to the Databento API (``databento`` Python client)
- Fetch CME/CBOT/NYMEX/COMEX continuous futures via ``DBNStore``
- Map CTL symbols (e.g. "/ES") to Databento instrument IDs
- Return back-adjusted electronic-session settlement bars
"""

from __future__ import annotations

import pandas as pd

from ctl.providers.base import DataProvider, ProviderMeta


class DatabentoProvider(DataProvider):
    """Databento continuous futures provider (stub).

    Parameters
    ----------
    api_key : str, optional
        Databento API key. Required for live data.
    dataset : str
        Databento dataset identifier (e.g. ``"GLBX.MDP3"``).
    """

    def __init__(
        self,
        api_key: str = "",
        dataset: str = "GLBX.MDP3",
    ) -> None:
        self._api_key = api_key
        self._dataset = dataset
        self._meta = ProviderMeta(
            provider="databento",
            session_type="electronic",
            roll_method="back_adjusted",
            close_type="settlement",
        )

    @property
    def name(self) -> str:
        return "databento"

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch OHLCV from Databento.

        Raises
        ------
        NotImplementedError
            Always — this is a stub awaiting API integration.
        """
        raise NotImplementedError(
            f"DatabentoProvider.get_ohlcv not yet implemented. "
            f"Requested: {symbol} {timeframe} {start}–{end}"
        )
