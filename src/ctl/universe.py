"""Universe configuration for Phase 1a.

Loads the symbol universe from configs/symbols_phase1a.yaml and provides
lookup helpers used by all downstream modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Set

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "symbols_phase1a.yaml"

# Canonical 11-cluster taxonomy (immutable within a phase).
CLUSTER_NAMES = (
    "IDX_FUT",
    "METALS_FUT",
    "ENERGY_FUT",
    "RATES_FUT",
    "GRAINS_FUT",
    "SOFTS_FUT",
    "LIVESTOCK_FUT",
    "FX_FUT",
    "ETF_SECTOR",
    "EQ_COMMODITY_LINKED",
    "EQ_MACRO_BELLWETHER",
)

# Tick values per futures contract (for slippage stress tests).
TICK_VALUES: Dict[str, float] = {
    "/PA": 5.0,
    "/GC": 10.0,
    "/SI": 25.0,
    "/HG": 12.50,
    "/PL": 5.0,
    "/ES": 12.50,
    "/NQ": 5.0,
    "/YM": 5.0,
    "/RTY": 5.0,
    "/CL": 10.0,
    "/NG": 10.0,
    "/ZB": 31.25,
    "/ZN": 15.625,
    "/ZC": 12.50,
    "/ZS": 12.50,
}

# Equities/ETFs default tick value.
EQUITY_TICK = 0.01


@dataclass
class SymbolInfo:
    """Metadata for a single symbol in the universe."""

    symbol: str
    cluster: str
    status: Literal["tradable", "research_only"]
    tick_value: float = 0.0

    @property
    def is_future(self) -> bool:
        return self.symbol.startswith("/")

    @property
    def is_tradable(self) -> bool:
        return self.status == "tradable"


@dataclass
class Universe:
    """Phase 1a universe: 29 symbols across 8 active clusters."""

    symbols: Dict[str, SymbolInfo] = field(default_factory=dict)

    # --- factory -----------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_CONFIG) -> "Universe":
        """Load universe from the canonical YAML config."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        symbols: Dict[str, SymbolInfo] = {}
        for status_key in ("tradable", "research_only"):
            clusters = raw.get(status_key, {})
            for cluster_key, sym_list in clusters.items():
                cluster_name = cluster_key.upper()
                for sym in sym_list:
                    canon = _canonicalise(sym)
                    tick = TICK_VALUES.get(canon, EQUITY_TICK)
                    symbols[canon] = SymbolInfo(
                        symbol=canon,
                        cluster=cluster_name,
                        status=status_key,
                        tick_value=tick,
                    )

        return cls(symbols=symbols)

    # --- queries -----------------------------------------------------------

    @property
    def all_symbols(self) -> List[str]:
        return sorted(self.symbols.keys())

    @property
    def tradable(self) -> List[str]:
        return sorted(s for s, info in self.symbols.items() if info.is_tradable)

    @property
    def research_only(self) -> List[str]:
        return sorted(
            s for s, info in self.symbols.items() if not info.is_tradable
        )

    @property
    def clusters(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for info in self.symbols.values():
            out.setdefault(info.cluster, []).append(info.symbol)
        return {k: sorted(v) for k, v in sorted(out.items())}

    def futures(self) -> List[str]:
        return sorted(s for s, info in self.symbols.items() if info.is_future)

    def equities_and_etfs(self) -> List[str]:
        return sorted(
            s for s, info in self.symbols.items() if not info.is_future
        )


def _canonicalise(sym: str) -> str:
    """Normalise symbol names.

    Futures keep their slash prefix (/ES).
    Equities/ETFs are stored without $ prefix (XOM not $XOM).
    """
    return sym.lstrip("$")
