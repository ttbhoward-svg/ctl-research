"""Symbol mapping + hash tracking for multi-provider data cutover (Task B).

Loads the canonical symbol map from ``configs/symbol_map_v1.yaml`` and provides
lookup / validation helpers for translating CTL canonical symbols across
providers (TradeStation, Databento, Norgate, IBKR).

The file's SHA-256 hash is tracked for reproducibility and integrated with
the archive manifest system via ``archive.DEFAULT_ARTIFACTS``.

See docs/notes/TaskB_assumptions.md for design rationale.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAP_PATH = REPO_ROOT / "configs" / "symbol_map_v1.yaml"

#: Phase 1a has exactly 29 symbols (15 futures + 7 ETFs + 7 equities).
EXPECTED_SYMBOL_COUNT = 29

#: Providers that every symbol must map to.
PROVIDERS = ("tradestation", "databento", "norgate", "ibkr")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_symbol_map(path: Path = DEFAULT_MAP_PATH) -> dict:
    """Load the symbol map YAML and return the raw dict."""
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Hash
# ---------------------------------------------------------------------------

def map_sha256(path: Path = DEFAULT_MAP_PATH) -> str:
    """Return the SHA-256 hex digest of the symbol map file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_symbol_map(smap: dict) -> List[str]:
    """Validate that *smap* contains all 29 symbols with complete mappings.

    Returns a list of error strings (empty if valid).
    """
    errors: List[str] = []
    symbols = smap.get("symbols", {})

    if len(symbols) != EXPECTED_SYMBOL_COUNT:
        errors.append(
            f"Expected {EXPECTED_SYMBOL_COUNT} symbols, found {len(symbols)}"
        )

    for ctl_sym, entry in symbols.items():
        if not isinstance(entry, dict):
            errors.append(f"{ctl_sym}: entry is not a dict")
            continue

        # Required metadata fields.
        for key in ("asset_class", "cluster", "status"):
            if key not in entry:
                errors.append(f"{ctl_sym}: missing '{key}'")

        # Provider mappings.
        providers = entry.get("providers", {})
        if not isinstance(providers, dict):
            errors.append(f"{ctl_sym}: 'providers' is not a dict")
            continue

        for p in PROVIDERS:
            if p not in providers:
                errors.append(
                    f"{ctl_sym}: missing provider mapping for '{p}'"
                )
            elif not providers[p]:
                errors.append(
                    f"{ctl_sym}: empty mapping for provider '{p}'"
                )

    return errors


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

def get_provider_symbol(
    smap: dict,
    ctl_symbol: str,
    provider: str,
) -> Optional[str]:
    """Translate a CTL canonical symbol to a provider-specific symbol.

    Returns ``None`` if the symbol or provider is not found.
    """
    entry = smap.get("symbols", {}).get(ctl_symbol)
    if entry is None:
        return None
    return entry.get("providers", {}).get(provider)


def get_ctl_symbol(
    smap: dict,
    provider: str,
    provider_symbol: str,
) -> Optional[str]:
    """Reverse-lookup: find the CTL canonical symbol for a provider symbol.

    Returns ``None`` if no match is found.
    """
    for ctl_sym, entry in smap.get("symbols", {}).items():
        if entry.get("providers", {}).get(provider) == provider_symbol:
            return ctl_sym
    return None


def all_ctl_symbols(smap: dict) -> List[str]:
    """Return a sorted list of all CTL canonical symbols in the map."""
    return sorted(smap.get("symbols", {}).keys())


def provider_symbols(smap: dict, provider: str) -> Dict[str, str]:
    """Return ``{ctl_symbol: provider_symbol}`` for a given provider."""
    out: Dict[str, str] = {}
    for ctl_sym, entry in smap.get("symbols", {}).items():
        ps = entry.get("providers", {}).get(provider)
        if ps:
            out[ctl_sym] = ps
    return out
