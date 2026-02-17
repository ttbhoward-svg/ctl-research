# Pre-Build Reconciliation Checklist (Final Lock Pass)

## Purpose
Resolve cross-document parameter drift before coding starts. Update all listed files, then mark this checklist complete.

## Canonical Source Order
1. `/Users/ttbhoward/Downloads/Final Spec ready for building/CTL_Phase_Gate_Checklist_v2.md`
2. `/Users/ttbhoward/Downloads/Final Spec ready for building/CTL_Final_PreBuild_Status 2026-02-17.md`
3. `/Users/ttbhoward/Downloads/Final Spec ready for building/CTL_Phase1a_Project_Tracker_v3.md`
4. `/Users/ttbhoward/Downloads/Final Spec ready for building/CTL_Research_Infrastructure_Synthesis_v2.md`
5. `/Users/ttbhoward/Downloads/Final Spec ready for building/B1_Strategy_Logic_Spec_v2.md`

---

## 1) SAN Cap + Drawdown Brakes (Conflict)
### Canonical (recommended)
- Phase A SAN cap: **$11,000**
- Phase A DD brakes: **10% / 15% / 20%**
- Phase B DD brakes: **5% / 10% / 15%**

### Conflicts found
- `/Users/ttbhoward/Downloads/Final Spec ready for building/CTL_Research_Infrastructure_Synthesis_v2.md:898` references **$7,500 SAN cap** and older brake framing.
- `/Users/ttbhoward/Downloads/Final Spec ready for building/CTL_Final_PreBuild_Status 2026-02-17.md:198-204` uses **$11,000** and current brakes.

### Action
- Replace all older SAN/brace values in synthesis with canonical values above.

---

## 2) OOS Spread Threshold for Capital Decisions (Conflict)
### Canonical (recommended)
- Gate threshold: **Top minus bottom tercile >= +1.0R** for Phase 1a pass.
- Capital infusion prerequisite should reference same strict threshold unless explicitly downgraded by future gate revision.

### Conflicts found
- `/Users/ttbhoward/Downloads/Final Spec ready for building/CTL_Phase_Gate_Checklist_v2.md:44` uses **+1.0R**.
- `/Users/ttbhoward/Downloads/Final Spec ready for building/CTL_Research_Infrastructure_Synthesis_v2.md:338` mentions **>=0.8R** in infusion section.

### Action
- Standardize to one number (recommend +1.0R) across all docs.

---

## 3) Feature Cap Phrase (Consistency)
### Canonical
- **"Phase 1a feature cap = 9 candidates + 1 cluster control (frozen)."**

### Potential drift points
- Gate checklist and tracker use this wording.
- Verify synthesis and logic spec use identical phrasing where phase scope is defined.

### Action
- Replace any variant phrasing (`8 features`, `10 features`, etc.) in Phase 1a sections unless explicitly marked as historical context.

---

## 4) Stop/Exit Fill Convention (Consistency)
### Canonical (choose one and enforce everywhere)
Option A (recommended for simplicity):
- Stop breach condition evaluated at bar close.
- If breached, exit at **next-bar open + slippage**.

### Drift risk
- Different docs describe stop handling in slightly different wording.

### Action
- Add one explicit “Execution Convention” paragraph in:
  - `B1_Strategy_Logic_Spec_v2.md`
  - `CTL_Phase_Gate_Checklist_v2.md`
  - `CTL_Phase1a_Project_Tracker_v3.md`

---

## 5) Purge Gap + Hold Period Link (Consistency)
### Canonical
- Purge gap = **30 trading days** (or `max holding period`, whichever is larger).

### Drift risk
- Some historical drafts used 44-day gap.

### Action
- Confirm no remaining references to non-canonical purge gap in final doc set.

---

## 6) Universe Lock Integrity (Consistency)
### Canonical
- Ring 1 / Phase 1a = **29 symbols (28 tradable + 1 research_only)**.

### Action
- Verify every universe table and gate reference uses this same count.
- Keep `research_only` handling and promotion/demotion rules identical across checklist/tracker/synthesis.

---

## 7) Research Column Governance (Clarity)
### Canonical
- Research columns are **hypothesis generation only**.
- No promotion to scoring feature without fresh pre-registered OOS validation in a later phase.

### Action
- Ensure this rule is explicitly repeated in:
  - Gate checklist
  - Project tracker tasking
  - Synthesis feature expansion section

---

## 8) Entry Degradation Tolerances (Optional tighten decision)
### Current
- <=25% total R degradation, <=5pp win-rate drop, <=30% MAR degradation.

### Decision point
- Keep as-is OR tighten for production promotion (e.g., 20% / 4pp / 20%).

### Action
- If keeping current values, mark explicitly as **Phase 1a exploratory tolerance** or **hard production gate tolerance**.

---

## 9) Model Type Naming Consistency
### Canonical
- Primary Phase 1a model: **Elastic Net** (with scaling).
- Any mention of pure Lasso should be contextual/historical unless intentionally used for comparison.

### Action
- Align naming in all docs so implementation and governance match.

---

## 10) Final Lock Procedure
1. Apply all edits.
2. Regenerate a one-page “Final Lock Summary” with canonical values.
3. Freeze document hashes.
4. Start coding only after lock summary is signed off.

---

## Completion Log
- [ ] Item 1 resolved
- [ ] Item 2 resolved
- [ ] Item 3 resolved
- [ ] Item 4 resolved
- [ ] Item 5 resolved
- [ ] Item 6 resolved
- [ ] Item 7 resolved
- [ ] Item 8 decision logged
- [ ] Item 9 resolved
- [ ] Item 10 completed

