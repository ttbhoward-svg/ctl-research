"""Provider manifest loader and availability checks.

Defines a per-symbol provider policy for Phase 1a:
- primary provider (intended execution/research source)
- fallback provider
- reference provider

This enables controlled, auditable provider promotion without changing
strategy/gating semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml

from ctl.universe import Universe

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "configs" / "cutover" / "provider_manifest_v1.yaml"

DEFAULT_DB_CONTINUOUS_DIR = REPO_ROOT / "data" / "processed" / "databento" / "cutover_v1" / "continuous"
DEFAULT_TS_DIR = REPO_ROOT / "data" / "raw" / "tradestation" / "cutover_v1"
DEFAULT_NORGATE_DIR = REPO_ROOT / "data" / "raw" / "norgate" / "cutover_v1"

PROVIDERS = {"databento", "tradestation", "norgate", "yfinance", "none"}


@dataclass(frozen=True)
class SymbolProviderPolicy:
    symbol: str
    primary: str
    fallback: str
    reference: str
    notes: str = ""


@dataclass(frozen=True)
class ProviderManifest:
    cycle_id: str
    version: str
    symbols: Dict[str, SymbolProviderPolicy]


@dataclass(frozen=True)
class ProviderAvailability:
    symbol: str
    primary: str
    primary_available: bool
    fallback: str
    fallback_available: bool
    reference: str
    reference_available: bool


def _base_symbol(symbol: str) -> str:
    return symbol.lstrip("/$")


def load_provider_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> ProviderManifest:
    with open(path) as f:
        raw = yaml.safe_load(f)

    symbols_raw = raw.get("symbols", {})
    policies: Dict[str, SymbolProviderPolicy] = {}
    for symbol, spec in symbols_raw.items():
        policies[symbol] = SymbolProviderPolicy(
            symbol=symbol,
            primary=str(spec.get("primary", "none")),
            fallback=str(spec.get("fallback", "none")),
            reference=str(spec.get("reference", "none")),
            notes=str(spec.get("notes", "")),
        )

    return ProviderManifest(
        cycle_id=str(raw.get("cycle_id", "")),
        version=str(raw.get("version", "")),
        symbols=policies,
    )


def validate_provider_manifest(
    manifest: ProviderManifest,
    expected_symbols: List[str] | None = None,
) -> List[str]:
    errors: List[str] = []

    if not manifest.cycle_id:
        errors.append("cycle_id is required")
    if not manifest.version:
        errors.append("version is required")

    if expected_symbols is None:
        expected_symbols = Universe.from_yaml().all_symbols

    expected = set(expected_symbols)
    actual = set(manifest.symbols.keys())
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        errors.append(f"missing symbols: {missing}")
    if extra:
        errors.append(f"unexpected symbols: {extra}")

    for sym, policy in manifest.symbols.items():
        for label, provider in (
            ("primary", policy.primary),
            ("fallback", policy.fallback),
            ("reference", policy.reference),
        ):
            if provider not in PROVIDERS:
                errors.append(f"{sym}: invalid {label} provider '{provider}'")
        if policy.primary == "none":
            errors.append(f"{sym}: primary provider cannot be 'none'")

    return errors


def _provider_available(
    symbol: str,
    provider: str,
    db_continuous_dir: Path = DEFAULT_DB_CONTINUOUS_DIR,
    ts_dir: Path = DEFAULT_TS_DIR,
    norgate_dir: Path = DEFAULT_NORGATE_DIR,
) -> bool:
    if provider == "none":
        return True

    base = _base_symbol(symbol)
    if provider in {"databento", "yfinance"}:
        return (db_continuous_dir / f"{base}_continuous.csv").is_file()
    if provider == "tradestation":
        return any(ts_dir.glob(f"TS_{base}*_1D*.csv"))
    if provider == "norgate":
        return any(norgate_dir.glob(f"NG_{base}_1D*.csv"))
    return False


def evaluate_manifest_availability(
    manifest: ProviderManifest,
    db_continuous_dir: Path = DEFAULT_DB_CONTINUOUS_DIR,
    ts_dir: Path = DEFAULT_TS_DIR,
    norgate_dir: Path = DEFAULT_NORGATE_DIR,
) -> List[ProviderAvailability]:
    rows: List[ProviderAvailability] = []
    for sym in sorted(manifest.symbols):
        p = manifest.symbols[sym]
        rows.append(
            ProviderAvailability(
                symbol=sym,
                primary=p.primary,
                primary_available=_provider_available(
                    sym, p.primary, db_continuous_dir, ts_dir, norgate_dir
                ),
                fallback=p.fallback,
                fallback_available=_provider_available(
                    sym, p.fallback, db_continuous_dir, ts_dir, norgate_dir
                ),
                reference=p.reference,
                reference_available=_provider_available(
                    sym, p.reference, db_continuous_dir, ts_dir, norgate_dir
                ),
            )
        )
    return rows
