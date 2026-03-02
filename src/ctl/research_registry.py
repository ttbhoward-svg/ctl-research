"""Research-tier ticker registry loader."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESEARCH_REGISTRY = REPO_ROOT / "configs" / "cutover" / "research_ticker_registry_v1.yaml"


@dataclass(frozen=True)
class ResearchSymbol:
    symbol: str
    enabled: bool = True
    slippage_per_side: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ResearchTickerRegistry:
    cycle_id: str
    registry_id: str
    profile_path: str
    symbols: List[ResearchSymbol]

    def enabled_symbols(self) -> List[str]:
        return [s.symbol for s in self.symbols if s.enabled]

    def slippage_map(self) -> Dict[str, float]:
        return {s.symbol: float(s.slippage_per_side) for s in self.symbols if s.enabled}

    def to_dict(self) -> dict:
        return {
            "cycle_id": self.cycle_id,
            "registry_id": self.registry_id,
            "profile_path": self.profile_path,
            "symbols": [s.to_dict() for s in self.symbols],
        }


def load_research_registry(path: Path = DEFAULT_RESEARCH_REGISTRY) -> ResearchTickerRegistry:
    """Load and validate research ticker registry YAML."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    cycle_id = str(raw.get("cycle_id", "")).strip()
    registry_id = str(raw.get("registry_id", "")).strip()
    profile_path = str(raw.get("profile_path", "")).strip()
    symbol_rows = raw.get("symbols", [])

    errors: List[str] = []
    if not cycle_id:
        errors.append("missing cycle_id")
    if not registry_id:
        errors.append("missing registry_id")
    if not profile_path:
        errors.append("missing profile_path")
    if not isinstance(symbol_rows, list):
        errors.append("symbols must be a list")

    symbols: List[ResearchSymbol] = []
    seen = set()
    if isinstance(symbol_rows, list):
        for i, row in enumerate(symbol_rows):
            if not isinstance(row, dict):
                errors.append(f"symbols[{i}] must be a mapping")
                continue
            sym = str(row.get("symbol", "")).strip().upper()
            if not sym:
                errors.append(f"symbols[{i}] missing symbol")
                continue
            if sym in seen:
                errors.append(f"duplicate symbol: {sym}")
                continue
            seen.add(sym)
            symbols.append(
                ResearchSymbol(
                    symbol=sym,
                    enabled=bool(row.get("enabled", True)),
                    slippage_per_side=float(row.get("slippage_per_side", 0.0)),
                    notes=str(row.get("notes", "")),
                )
            )

    if errors:
        raise ValueError("invalid research registry: " + "; ".join(errors))

    return ResearchTickerRegistry(
        cycle_id=cycle_id,
        registry_id=registry_id,
        profile_path=profile_path,
        symbols=symbols,
    )
