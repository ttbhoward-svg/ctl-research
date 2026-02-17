# CTL Project Handoff

## Current State
- Repo initialized and pushed to GitHub via SSH.
- Pre-build reconciliation checklist: **COMPLETE** (10/10 items resolved, verified 2026-02-17).
- Final Lock Summary: **SIGNED** (all SHA-256 hashes verified).
- Phase 1a coding: **AUTHORIZED** — Task 1 (data ingestion) in progress.

## Canonical Docs (read first)
1. docs/governance/Final_Lock_Summary.md (authoritative canonical values)
2. docs/governance/PreBuild_Reconciliation_Checklist.md (completed)
3. docs/governance/CTL_Phase_Gate_Checklist_One_Page.md
4. docs/specs/CTL_Dashboard_Build_Spec_v1.1.md

## Locked Source Specs (in repo)
- docs/specs/locked/B1_Strategy_Logic_Spec_v2.md
- docs/specs/locked/CTL_Phase1a_Project_Tracker_v3.md
- docs/specs/locked/CTL_Phase_Gate_Checklist_v2.md
- docs/specs/locked/CTL_Research_Infrastructure_Synthesis_v2.md
- docs/specs/locked/CTL_Final_PreBuild_Status.md

## External Final Spec Folder (original source)
- /Users/ttbhoward/Downloads/Final spec ready to build/

## Immediate Next Steps
1. ~~Reconcile spec drift using PreBuild_Reconciliation_Checklist.md.~~ DONE
2. ~~Generate final lock summary.~~ DONE
3. ~~Freeze hashes and begin Phase 1a coding.~~ AUTHORIZED
4. **Current: Phase 1a Task 1 — Data ingestion infrastructure**

## Coding Scope (Phase 1a only)
- B1 strategy in Python
- 29-symbol universe (28 tradable + 1 research_only)
- Frozen feature set and gate criteria
- Model card + negative controls + calibration + slippage/degradation tests

## Non-Negotiables
- No tuning on OOS
- Costs ON
- Purged time-series CV
- Gate criteria binary pass/iterate
