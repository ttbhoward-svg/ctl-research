"""Operating profile loader and status comparison logic (H.7).

Provides typed dataclasses for the locked operating profile YAML and
pure functions to compare re-derived acceptance decisions against the
expected statuses recorded in the profile.

Usage
-----
>>> from ctl.operating_profile import load_operating_profile, check_symbol_status
>>> profile = load_operating_profile("configs/cutover/operating_profile_v1.yaml")
>>> result = check_symbol_status("ES", "WATCH", "WATCH")
>>> result.passed
True
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SymbolSetting:
    """Per-symbol operational settings from the locked profile."""

    tick_size: float
    max_day_delta: int
    expected_status: str
    notes: str = ""


@dataclass(frozen=True)
class PolicyConstraints:
    """Policy constraint flags."""

    thresholds_locked: bool = True
    strategy_logic_locked: bool = True


@dataclass(frozen=True)
class OperatingProfile:
    """Complete locked operating profile."""

    cycle_id: str
    locked_date: str
    portfolio_recommendation: str
    gating_universe: List[str]
    non_gating_symbols: List[str]
    symbol_settings: Dict[str, SymbolSetting]
    policy_constraints: PolicyConstraints
    portfolio_scope: str = "futures_only"


@dataclass
class SymbolCheckResult:
    """Result of comparing one symbol's expected vs actual decision."""

    symbol: str
    expected: str
    actual: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "expected": self.expected,
            "actual": self.actual,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass
class PortfolioCheckResult:
    """Aggregate result of all symbol checks."""

    passed: bool
    recommendation: str
    symbol_results: List[SymbolCheckResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "recommendation": self.recommendation,
            "symbol_results": [r.to_dict() for r in self.symbol_results],
        }


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

_REQUIRED_TOP_KEYS = {
    "cycle_id",
    "locked_date",
    "portfolio_recommendation",
    "gating_universe",
    "symbol_settings",
}


def load_operating_profile(path: Path) -> OperatingProfile:
    """Parse operating profile YAML and construct typed objects.

    Parameters
    ----------
    path : Path
        Path to the YAML configuration file.

    Returns
    -------
    OperatingProfile

    Raises
    ------
    ValueError
        If required keys are missing or symbol_settings are malformed.
    """
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)

    missing = _REQUIRED_TOP_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Missing required keys in profile: {sorted(missing)}")

    # Parse symbol settings.
    raw_settings = data["symbol_settings"]
    if not isinstance(raw_settings, dict):
        raise ValueError("symbol_settings must be a mapping")

    symbol_settings: Dict[str, SymbolSetting] = {}
    for sym, cfg in raw_settings.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"symbol_settings[{sym}] must be a mapping")
        symbol_settings[sym] = SymbolSetting(
            tick_size=float(cfg["tick_size"]),
            max_day_delta=int(cfg["max_day_delta"]),
            expected_status=str(cfg["expected_status"]),
            notes=str(cfg.get("notes", "")),
        )

    # Parse policy constraints (optional section with defaults).
    raw_policy = data.get("policy_constraints", {})
    policy = PolicyConstraints(
        thresholds_locked=bool(raw_policy.get("thresholds_locked", True)),
        strategy_logic_locked=bool(raw_policy.get("strategy_logic_locked", True)),
    )

    return OperatingProfile(
        cycle_id=str(data["cycle_id"]),
        locked_date=str(data["locked_date"]),
        portfolio_recommendation=str(data["portfolio_recommendation"]),
        gating_universe=list(data["gating_universe"]),
        non_gating_symbols=list(data.get("non_gating_symbols", [])),
        symbol_settings=symbol_settings,
        policy_constraints=policy,
        portfolio_scope=str(data.get("portfolio_scope", "futures_only")),
    )


# ---------------------------------------------------------------------------
# Status comparison
# ---------------------------------------------------------------------------

def check_symbol_status(
    symbol: str,
    expected: str,
    actual: str,
) -> SymbolCheckResult:
    """Compare expected and actual decision strings for one symbol.

    Parameters
    ----------
    symbol : str
        Symbol label.
    expected : str
        Expected decision from the locked profile.
    actual : str
        Re-derived decision from live diagnostics.

    Returns
    -------
    SymbolCheckResult
    """
    passed = expected == actual
    detail = "" if passed else f"expected {expected}, got {actual}"
    return SymbolCheckResult(
        symbol=symbol,
        expected=expected,
        actual=actual,
        passed=passed,
        detail=detail,
    )


def check_portfolio(
    profile: OperatingProfile,
    symbol_results: List[SymbolCheckResult],
) -> PortfolioCheckResult:
    """Check whether all gating symbols match their expected status.

    Parameters
    ----------
    profile : OperatingProfile
        Locked operating profile.
    symbol_results : list of SymbolCheckResult
        Per-symbol comparison results.

    Returns
    -------
    PortfolioCheckResult
    """
    all_passed = all(r.passed for r in symbol_results)
    return PortfolioCheckResult(
        passed=all_passed,
        recommendation=profile.portfolio_recommendation,
        symbol_results=list(symbol_results),
    )


# ---------------------------------------------------------------------------
# TS file discovery (CUSTOM variant)
# ---------------------------------------------------------------------------

def discover_ts_custom_file(
    symbol: str,
    ts_dir: Path,
    variant: str,
) -> Optional[Path]:
    """Discover a TradeStation CUSTOM file for *symbol* and *variant*.

    Glob pattern: ``TS_{symbol}_CUSTOM_{variant}_1D_*.csv``

    Follows the same first-sorted-match convention as
    ``parity_prep.discover_ts_file``.

    Parameters
    ----------
    symbol : str
        Root symbol (e.g. ``"ES"``).
    ts_dir : Path
        Directory containing TS CSV files.
    variant : str
        Variant tag (e.g. ``"ADJ"`` or ``"UNADJ"``).

    Returns
    -------
    Path or None
    """
    ts_dir = Path(ts_dir)
    matches = sorted(ts_dir.glob(f"TS_{symbol}_CUSTOM_{variant}_1D_*.csv"))
    if not matches:
        return None
    if len(matches) > 1:
        logger.warning(
            "Multiple TS CUSTOM %s files for %s: %s — using %s",
            variant,
            symbol,
            [m.name for m in matches],
            matches[0].name,
        )
    return matches[0]
