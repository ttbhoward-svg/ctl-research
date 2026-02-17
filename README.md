# CTL Research Infrastructure

Systematic research pipeline for CTL setup validation, scoring, and risk-governed deployment.

## Current Scope
- Phase 1a: B1 strategy, fixed universe, frozen feature set, strict governance gates.

## Repo Conventions
- Specs and governance docs are source-of-truth and versioned in `docs/`.
- Raw/processed data and heavy artifacts are excluded from git.
- Every model run should produce a model card and reproducibility metadata.

## Quick Start
1. Create virtual environment.
2. Install dependencies.
3. Populate `configs/symbols_phase1a.yaml`.
4. Run `scripts/run_phase1a.py`.

