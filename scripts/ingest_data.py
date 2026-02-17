#!/usr/bin/env python3
"""Phase 1a Task 1 — Data ingestion and sanitisation.

Usage:
    python scripts/ingest_data.py                 # load all + sanitise
    python scripts/ingest_data.py --sanitise-only  # sanitise already-loaded data
    python scripts/ingest_data.py --sbsw-only      # download SBSW from yfinance

Expects TradeStation CSV files in data/raw/ with naming convention:
    {SYMBOL}_{timeframe}.csv
    e.g. PA_daily.csv, ES_weekly.csv, XLE_monthly.csv, GS_h4.csv

For futures, strip the leading slash: /PA -> PA_daily.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src/ to path so we can import ctl package.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ctl.data_loader import (
    download_yfinance,
    load_all_tradestation,
    save_processed,
)
from ctl.data_sanitizer import print_report, sanitise_universe
from ctl.universe import Universe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest")


def main() -> None:
    parser = argparse.ArgumentParser(description="CTL Phase 1a data ingestion")
    parser.add_argument("--sanitise-only", action="store_true", help="Only run sanitiser on processed data")
    parser.add_argument("--sbsw-only", action="store_true", help="Only download SBSW via yfinance")
    args = parser.parse_args()

    universe = Universe.from_yaml()
    logger.info(
        "Universe loaded: %d symbols (%d tradable, %d research_only)",
        len(universe.all_symbols),
        len(universe.tradable),
        len(universe.research_only),
    )

    if args.sbsw_only:
        logger.info("Downloading SBSW from yfinance...")
        sbsw_data = download_yfinance("SBSW", start="2015-01-01")
        save_processed({"SBSW": sbsw_data})
        logger.info("SBSW download complete.")
        return

    if args.sanitise_only:
        logger.info("Sanitise-only mode: loading from processed files...")
        # Load from processed dir.
        from ctl.data_loader import PROCESSED_DIR, load_processed

        data = {}
        for sym in universe.all_symbols:
            data[sym] = {}
            for tf in ("daily", "weekly", "monthly", "h4"):
                try:
                    data[sym][tf] = load_processed(sym, tf, PROCESSED_DIR)
                except FileNotFoundError:
                    pass
        reports = sanitise_universe(data, universe)
        print(print_report(reports))
        return

    # Full pipeline: load TradeStation + yfinance, save, sanitise.
    logger.info("Loading TradeStation CSV exports from %s ...", REPO_ROOT / "data" / "raw")
    data = load_all_tradestation(universe)

    # Download SBSW via yfinance.
    if "SBSW" in universe.all_symbols:
        logger.info("Downloading SBSW from yfinance...")
        sbsw_data = download_yfinance("SBSW", start="2015-01-01")
        data["SBSW"] = sbsw_data

    # Save processed files.
    written = save_processed(data)
    logger.info("Saved %d processed files.", len(written))

    # Run sanitiser.
    logger.info("Running data sanitiser...")
    reports = sanitise_universe(data, universe)
    report_text = print_report(reports)
    print(report_text)

    # Write report to file.
    report_path = REPO_ROOT / "outputs" / "reports" / "data_sanitiser_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text)
    logger.info("Report saved to %s", report_path)

    # Summary.
    total_errors = sum(r.error_count for r in reports.values())
    if total_errors > 0:
        logger.error(
            "DATA ERRORS FOUND: %d errors. Fix before proceeding to Task 2.",
            total_errors,
        )
        sys.exit(1)
    else:
        logger.info("Data ingestion complete. No blocking errors.")


if __name__ == "__main__":
    main()
