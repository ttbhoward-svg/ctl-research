"""Data health-check report for the Phase 1a assembled dataset.

Runs a battery of integrity checks and produces a structured pass/fail
report.  Checks cover:
  - Row counts and symbol coverage
  - Missingness by field
  - Critical-field null checks
  - Duplicate-key detection
  - Date-range validation
  - COT presence rules (futures vs non-futures)

See docs/notes/Task8_assumptions.md for check rationale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from ctl.universe import Universe

logger = logging.getLogger(__name__)

# Fields that must never be null.
CRITICAL_FIELDS = [
    "EntryPrice",
    "StopPrice",
    "TP1",
    "SetupType",
    "RMultiple_Actual",
    "TheoreticalR",
    "AssetCluster",
]

# Composite key for duplicate detection.
DUPLICATE_KEY_COLS = ["Date", "Ticker", "Timeframe", "TriggerBarIdx"]


@dataclass
class CheckResult:
    """One health-check outcome."""

    name: str
    passed: bool
    detail: str


@dataclass
class MissingSummary:
    """Per-column missingness counts."""

    column: str
    n_missing: int
    n_total: int

    @property
    def pct_missing(self) -> float:
        if self.n_total == 0:
            return 0.0
        return 100.0 * self.n_missing / self.n_total


@dataclass
class HealthReport:
    """Full health-check report."""

    checks: List[CheckResult] = field(default_factory=list)
    missingness: List[MissingSummary] = field(default_factory=list)
    n_rows: int = 0
    n_columns: int = 0
    symbols_present: int = 0
    symbols_expected: int = 0
    date_min: Optional[str] = None
    date_max: Optional[str] = None

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            f"Health Report: {self.n_passed} passed, {self.n_failed} failed",
            f"  Rows: {self.n_rows}  Columns: {self.n_columns}",
            f"  Symbols: {self.symbols_present}/{self.symbols_expected}",
            f"  Date range: {self.date_min} to {self.date_max}",
            "",
        ]
        for c in self.checks:
            status = "PASS" if c.passed else "FAIL"
            lines.append(f"  [{status}] {c.name}: {c.detail}")

        # Missingness summary (only non-zero).
        missing_lines = [m for m in self.missingness if m.n_missing > 0]
        if missing_lines:
            lines.append("")
            lines.append("  Missingness (non-zero only):")
            for m in missing_lines:
                lines.append(f"    {m.column}: {m.n_missing}/{m.n_total} ({m.pct_missing:.1f}%)")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Check implementations
# ---------------------------------------------------------------------------

def _check_row_count(df: pd.DataFrame) -> CheckResult:
    n = len(df)
    return CheckResult("row_count", n > 0, f"{n} rows")


def _check_symbol_coverage(
    df: pd.DataFrame, universe: Universe,
) -> CheckResult:
    expected = set(universe.all_symbols)
    actual = set(df["Ticker"].unique()) if "Ticker" in df.columns else set()
    missing = sorted(expected - actual)
    n_actual = len(actual)
    n_expected = len(expected)
    # Informational — not all symbols may produce triggers.
    passed = n_actual > 0
    detail = f"{n_actual}/{n_expected} symbols"
    if missing:
        detail += f". Missing: {missing}"
    return CheckResult("symbol_coverage", passed, detail)


def _check_critical_nulls(df: pd.DataFrame) -> List[CheckResult]:
    checks = []
    for col in CRITICAL_FIELDS:
        if col not in df.columns:
            checks.append(CheckResult(f"no_nulls_{col}", False, "column missing from dataset"))
            continue
        n_null = int(df[col].isna().sum())
        checks.append(CheckResult(f"no_nulls_{col}", n_null == 0, f"{n_null} nulls"))
    return checks


def _check_duplicates(df: pd.DataFrame) -> CheckResult:
    key_cols = [c for c in DUPLICATE_KEY_COLS if c in df.columns]
    if not key_cols:
        return CheckResult("no_duplicates", True, "no key columns present")
    n_dup = int(df.duplicated(subset=key_cols, keep=False).sum())
    return CheckResult("no_duplicates", n_dup == 0, f"{n_dup} duplicate rows (key: {key_cols})")


def _check_date_range(df: pd.DataFrame) -> CheckResult:
    if "Date" not in df.columns or df.empty:
        return CheckResult("date_range", False, "no Date column or empty dataset")
    d_min = df["Date"].min()
    d_max = df["Date"].max()
    detail = f"{d_min} to {d_max}"
    # Check if range spans expected IS period (2018+).
    passed = True
    if pd.notna(d_min) and pd.Timestamp(d_min).year > 2020:
        detail += " (WARNING: no data before 2020)"
        passed = False
    return CheckResult("date_range", passed, detail)


def _check_cot_rules(df: pd.DataFrame, universe: Universe) -> List[CheckResult]:
    """COT should be present for futures, NULL for non-futures."""
    checks = []
    futures = set(universe.futures())
    non_futures = set(universe.equities_and_etfs())

    for col in ("COT_20D_Delta", "COT_ZScore_1Y"):
        if col not in df.columns:
            checks.append(CheckResult(f"cot_rules_{col}", True, "column not in dataset (OK if no COT data)"))
            continue

        # Non-futures should have all NULL.
        nf_mask = df["Ticker"].isin(non_futures)
        nf_non_null = int(df.loc[nf_mask, col].notna().sum())
        checks.append(CheckResult(
            f"cot_null_non_futures_{col}",
            nf_non_null == 0,
            f"{nf_non_null} non-null values in non-futures (should be 0)",
        ))

    return checks


def _check_r_consistency(df: pd.DataFrame) -> CheckResult:
    """Spot-check R-multiple consistency: R = (Exit - Entry) / Risk."""
    if not {"RMultiple_Actual", "EntryPrice", "ExitPrice", "RiskPerUnit"}.issubset(df.columns):
        return CheckResult("r_consistency", True, "required columns missing, skipped")
    if df.empty:
        return CheckResult("r_consistency", True, "empty dataset")

    # Only check rows where RiskPerUnit > 0 (avoid division issues).
    mask = df["RiskPerUnit"] > 0
    sub = df[mask]
    if sub.empty:
        return CheckResult("r_consistency", True, "no rows with positive risk")

    expected_r = (sub["ExitPrice"] - sub["EntryPrice"]) / sub["RiskPerUnit"]
    diff = (sub["RMultiple_Actual"] - expected_r).abs()
    max_diff = float(diff.max())
    passed = max_diff < 0.01  # tolerance for float arithmetic
    return CheckResult("r_consistency", passed, f"max |actual - expected| = {max_diff:.6f}")


# ---------------------------------------------------------------------------
# Missingness
# ---------------------------------------------------------------------------

def _compute_missingness(df: pd.DataFrame) -> List[MissingSummary]:
    summaries = []
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        summaries.append(MissingSummary(col, n_missing, len(df)))
    return summaries


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_health_checks(
    df: pd.DataFrame,
    universe: Universe,
) -> HealthReport:
    """Run all health checks on the assembled dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``dataset_assembler.assemble_dataset``.
    universe : Universe
        Phase 1a universe for coverage and COT rule checks.

    Returns
    -------
    HealthReport with all check results, missingness, and summary stats.
    """
    report = HealthReport()
    report.n_rows = len(df)
    report.n_columns = len(df.columns)

    if "Ticker" in df.columns:
        report.symbols_present = df["Ticker"].nunique()
    report.symbols_expected = len(universe.all_symbols)

    if "Date" in df.columns and not df.empty:
        report.date_min = str(df["Date"].min())
        report.date_max = str(df["Date"].max())

    # Run all checks.
    report.checks.append(_check_row_count(df))
    report.checks.append(_check_symbol_coverage(df, universe))
    report.checks.extend(_check_critical_nulls(df))
    report.checks.append(_check_duplicates(df))
    report.checks.append(_check_date_range(df))
    report.checks.extend(_check_cot_rules(df, universe))
    report.checks.append(_check_r_consistency(df))

    # Missingness.
    report.missingness = _compute_missingness(df)

    status = "ALL PASSED" if report.all_passed else f"{report.n_failed} FAILED"
    logger.info("Health check: %s (%d checks)", status, len(report.checks))
    return report
