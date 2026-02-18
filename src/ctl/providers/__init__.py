"""Data provider abstraction layer.

Defines a canonical OHLCV schema and a ``DataProvider`` interface
that all concrete providers (Databento, Norgate, etc.) must implement.
"""

from ctl.providers.base import (
    CANONICAL_COLUMNS,
    CLOSE_TYPES,
    ROLL_METHODS,
    SESSION_TYPES,
    DataProvider,
    ProviderMeta,
    validate_canonical,
)

__all__ = [
    "CANONICAL_COLUMNS",
    "CLOSE_TYPES",
    "DataProvider",
    "ProviderMeta",
    "ROLL_METHODS",
    "SESSION_TYPES",
    "validate_canonical",
]
