# CTL Project Handoff

## Current State
- Repo initialized and pushed to GitHub via SSH.
- Canonical specs and governance docs prepared.
- Pre-build reconciliation checklist exists and should be completed before coding Phase 1a.

## Canonical Docs (read first)
1. docs/governance/PreBuild_Reconciliation_Checklist.md
2. docs/governance/Final_Lock_Summary_Template.md
3. docs/governance/CTL_Phase_Gate_Checklist_One_Page.md
4. docs/specs/CTL_Dashboard_Build_Spec_v1.1.md

## External Final Spec Folder (source docs)
- /Users/ttbhoward/Downloads/Final Spec ready for building/

## Immediate Next Steps
1. Reconcile spec drift using PreBuild_Reconciliation_Checklist.md.
2. Generate final lock summary from Final_Lock_Summary_Template.md.
3. Freeze hashes and begin Phase 1a coding.

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

