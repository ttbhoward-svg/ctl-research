"""Provider connectivity and configuration checks (Data Cutover Task E).

Validates local provider setup without making network calls:

- Environment variable presence (``DATABENTO_API_KEY``, ``NORGATE_DATABASE_PATH``)
- Provider stub instantiation and metadata validation
- Symbol map loading and validation
- Sample ``.csv.zst`` schema smoke check

Each check returns PASS / FAIL / SKIP.

See docs/notes/TaskE_assumptions.md for design rationale.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ctl.providers.base import ProviderMeta, validate_canonical
from ctl.providers.databento_provider import DatabentoProvider
from ctl.providers.norgate_provider import NorgateProvider
from ctl.symbol_map import (
    DEFAULT_MAP_PATH,
    load_symbol_map,
    validate_symbol_map,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Default location for Databento outrights.
DEFAULT_OUTRIGHTS_DIR = (
    REPO_ROOT / "data" / "raw" / "databento" / "cutover_v1" / "outrights_only"
)

#: Symbols to smoke-check.
SMOKE_CHECK_SYMBOLS = ("ES", "CL", "PA")

#: Expected columns in Databento OHLCV-1D CSV files.
EXPECTED_CSV_COLUMNS = (
    "ts_event", "rtype", "publisher_id", "instrument_id",
    "open", "high", "low", "close", "volume", "symbol",
)

#: Environment variable names.
ENV_DATABENTO_KEY = "DATABENTO_API_KEY"
ENV_NORGATE_PATH = "NORGATE_DATABASE_PATH"

# Status values.
PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Result of a single connectivity check."""

    name: str
    status: str  # PASS, FAIL, or SKIP
    detail: str = ""


@dataclass
class ConnectivityReport:
    """Aggregate result from all connectivity checks."""

    checks: List[CheckResult] = field(default_factory=list)

    @property
    def any_fail(self) -> bool:
        return any(c.status == FAIL for c in self.checks)

    @property
    def n_pass(self) -> int:
        return sum(1 for c in self.checks if c.status == PASS)

    @property
    def n_fail(self) -> int:
        return sum(1 for c in self.checks if c.status == FAIL)

    @property
    def n_skip(self) -> int:
        return sum(1 for c in self.checks if c.status == SKIP)

    def summary_dict(self) -> dict:
        return {
            "n_pass": self.n_pass,
            "n_fail": self.n_fail,
            "n_skip": self.n_skip,
            "all_ok": not self.any_fail,
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail}
                for c in self.checks
            ],
        }

    def summary_text(self) -> str:
        lines = []
        for c in self.checks:
            tag = f"[{c.status}]"
            lines.append(f"  {tag:8s} {c.name}: {c.detail}")
        header = f"Connectivity: {self.n_pass} PASS, {self.n_fail} FAIL, {self.n_skip} SKIP"
        return header + "\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_env_var(var_name: str) -> CheckResult:
    """Check whether an environment variable is set and non-empty."""
    val = os.environ.get(var_name, "")
    if val:
        return CheckResult(
            name=f"env_{var_name}",
            status=PASS,
            detail=f"{var_name} is set ({len(val)} chars)",
        )
    return CheckResult(
        name=f"env_{var_name}",
        status=SKIP,
        detail=f"{var_name} not set or empty",
    )


def check_provider_stub(provider_name: str) -> CheckResult:
    """Instantiate a provider stub and validate its metadata."""
    try:
        if provider_name == "databento":
            p = DatabentoProvider()
            meta = p._meta
        elif provider_name == "norgate":
            p = NorgateProvider()
            meta = p._meta
        else:
            return CheckResult(
                name=f"provider_{provider_name}",
                status=FAIL,
                detail=f"Unknown provider: {provider_name}",
            )

        errors = meta.validate()
        if errors:
            return CheckResult(
                name=f"provider_{provider_name}",
                status=FAIL,
                detail=f"Metadata errors: {errors}",
            )
        return CheckResult(
            name=f"provider_{provider_name}",
            status=PASS,
            detail=(
                f"Instantiated OK; session={meta.session_type}, "
                f"roll={meta.roll_method}, close={meta.close_type}"
            ),
        )
    except Exception as exc:
        return CheckResult(
            name=f"provider_{provider_name}",
            status=FAIL,
            detail=str(exc),
        )


def check_symbol_map(path: Path = DEFAULT_MAP_PATH) -> CheckResult:
    """Load and validate the symbol map YAML."""
    try:
        smap = load_symbol_map(path)
    except FileNotFoundError:
        return CheckResult(
            name="symbol_map",
            status=FAIL,
            detail=f"File not found: {path}",
        )
    except Exception as exc:
        return CheckResult(
            name="symbol_map",
            status=FAIL,
            detail=f"Load error: {exc}",
        )

    errors = validate_symbol_map(smap)
    if errors:
        return CheckResult(
            name="symbol_map",
            status=FAIL,
            detail=f"{len(errors)} validation errors: {errors[:3]}",
        )

    n_symbols = len(smap.get("symbols", {}))
    return CheckResult(
        name="symbol_map",
        status=PASS,
        detail=f"Loaded {n_symbols} symbols, all valid",
    )


def check_sample_csv(
    root_symbol: str,
    outrights_dir: Path = DEFAULT_OUTRIGHTS_DIR,
) -> CheckResult:
    """Read one sample .csv.zst file for *root_symbol* and verify schema."""
    sym_dir = outrights_dir / root_symbol
    if not sym_dir.is_dir():
        return CheckResult(
            name=f"sample_csv_{root_symbol}",
            status=SKIP,
            detail=f"Directory not found: {sym_dir}",
        )

    zst_files = sorted(sym_dir.glob("*.csv.zst"))
    if not zst_files:
        return CheckResult(
            name=f"sample_csv_{root_symbol}",
            status=SKIP,
            detail=f"No .csv.zst files in {sym_dir}",
        )

    sample = zst_files[0]
    try:
        df = pd.read_csv(sample, nrows=5)
    except Exception as exc:
        return CheckResult(
            name=f"sample_csv_{root_symbol}",
            status=FAIL,
            detail=f"Read error on {sample.name}: {exc}",
        )

    missing = [c for c in EXPECTED_CSV_COLUMNS if c not in df.columns]
    if missing:
        return CheckResult(
            name=f"sample_csv_{root_symbol}",
            status=FAIL,
            detail=f"Missing columns in {sample.name}: {missing}",
        )

    return CheckResult(
        name=f"sample_csv_{root_symbol}",
        status=PASS,
        detail=f"{sample.name}: {len(df)} rows, schema OK",
    )


# ---------------------------------------------------------------------------
# Full suite
# ---------------------------------------------------------------------------

def run_connectivity_checks(
    outrights_dir: Path = DEFAULT_OUTRIGHTS_DIR,
    symbol_map_path: Path = DEFAULT_MAP_PATH,
    smoke_symbols: tuple = SMOKE_CHECK_SYMBOLS,
) -> ConnectivityReport:
    """Run all connectivity checks and return a report.

    Parameters
    ----------
    outrights_dir : Path
        Directory containing ``{SYMBOL}/`` subdirectories with ``.csv.zst`` files.
    symbol_map_path : Path
        Path to the symbol map YAML.
    smoke_symbols : tuple of str
        Root symbols to smoke-check.

    Returns
    -------
    ConnectivityReport
    """
    report = ConnectivityReport()

    # 1. Environment variables.
    report.checks.append(check_env_var(ENV_DATABENTO_KEY))
    report.checks.append(check_env_var(ENV_NORGATE_PATH))

    # 2. Provider stubs.
    report.checks.append(check_provider_stub("databento"))
    report.checks.append(check_provider_stub("norgate"))

    # 3. Symbol map.
    report.checks.append(check_symbol_map(symbol_map_path))

    # 4. Sample CSV schema smoke checks.
    for sym in smoke_symbols:
        report.checks.append(check_sample_csv(sym, outrights_dir))

    return report
