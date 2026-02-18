#!/usr/bin/env bash
set -euo pipefail

EXPECTED_REPO="${1:-$HOME/ctl-research}"
EXPECTED_REMOTE="git@github.com:ttbhoward-svg/ctl-research.git"

fail=0

say_ok() { printf "[OK] %s\n" "$1"; }
say_warn() { printf "[WARN] %s\n" "$1"; }
say_fail() { printf "[FAIL] %s\n" "$1"; fail=1; }

printf "Workspace Alignment Check\n"
printf "Expected repo: %s\n\n" "$EXPECTED_REPO"

# 1) Path exists
if [[ -d "$EXPECTED_REPO" ]]; then
  say_ok "Repo path exists"
else
  say_fail "Repo path missing: $EXPECTED_REPO"
fi

# 2) Git root check
if git -C "$EXPECTED_REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  ROOT="$(git -C "$EXPECTED_REPO" rev-parse --show-toplevel)"
  if [[ "$ROOT" == "$EXPECTED_REPO" ]]; then
    say_ok "Git root matches expected path"
  else
    say_fail "Git root mismatch. Expected $EXPECTED_REPO but found $ROOT"
  fi
else
  say_fail "Not a git repo: $EXPECTED_REPO"
fi

# 3) Remote check
if git -C "$EXPECTED_REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  REMOTE="$(git -C "$EXPECTED_REPO" remote get-url origin 2>/dev/null || true)"
  if [[ "$REMOTE" == "$EXPECTED_REMOTE" || "$REMOTE" == "https://github.com/ttbhoward-svg/ctl-research.git" ]]; then
    say_ok "Origin remote is correct: $REMOTE"
  else
    say_fail "Unexpected origin remote: ${REMOTE:-<none>}"
  fi
fi

# 4) Branch/status
if git -C "$EXPECTED_REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  BRANCH="$(git -C "$EXPECTED_REPO" branch --show-current)"
  say_ok "Current branch: ${BRANCH:-detached}"
  if [[ -n "$(git -C "$EXPECTED_REPO" status --porcelain)" ]]; then
    say_warn "Working tree has uncommitted changes"
  else
    say_ok "Working tree clean"
  fi
fi

# 5) Required project files
REQ_FILES=(
  "RESUME_ON_ANY_MACHINE.md"
  "SESSION_LOG.md"
  "HANDOFF.md"
  "docs/governance/phase1a_completion_snapshot.md"
)
for rel in "${REQ_FILES[@]}"; do
  if [[ -f "$EXPECTED_REPO/$rel" ]]; then
    say_ok "Found $rel"
  else
    say_fail "Missing required file: $rel"
  fi
done

# 6) Claude CLI availability
if command -v claude >/dev/null 2>&1; then
  say_ok "Claude CLI found: $(command -v claude)"
else
  say_warn "Claude CLI not in PATH (Codex/Git can still work)"
fi

# 7) SSH auth to GitHub (non-fatal)
if ssh -T git@github.com -o BatchMode=yes 2>&1 | grep -qi "successfully authenticated"; then
  say_ok "SSH auth to GitHub works"
else
  say_warn "SSH auth check did not confirm success (may still work interactively)"
fi

# 8) Human check hints for Codex/Claude UI
printf "\nManual UI checks:\n"
printf "%s\n" "- In Codex app, project path should be: $EXPECTED_REPO"
printf "%s\n" "- In Claude, run: pwd && git rev-parse --show-toplevel && git remote -v"

if [[ $fail -eq 0 ]]; then
  printf "\nRESULT: ALIGNED\n"
  exit 0
else
  printf "\nRESULT: NOT ALIGNED (fix failures above)\n"
  exit 1
fi
