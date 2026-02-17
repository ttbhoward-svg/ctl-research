#!/usr/bin/env python3
"""Phase 1a runner placeholder.

Wire this script to your ingestion -> signal detection -> simulation -> modeling pipeline.
"""

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    print("CTL Phase 1a runner")
    print(f"Repo root: {root}")
    print("Next: implement ingestion and simulator entrypoints.")


if __name__ == "__main__":
    main()
