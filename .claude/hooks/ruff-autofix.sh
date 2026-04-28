#!/usr/bin/env bash
# PostToolUse hook: auto-format and auto-fix Python files after Write/Edit.
# Reads the hook payload on stdin, extracts the edited file path, and runs ruff
# only when (a) the file is *.py and (b) ruff is available. Anything else is a
# no-op so the hook never blocks unrelated edits.
#
# Exit 0 always — we surface ruff output so Claude sees it, but we never want
# the hook itself to fail a tool call. Ruff's own output (if any) is the signal.

set -u

payload="$(cat || true)"

# Cheap JSON extraction without requiring jq. Looks for "file_path":"..." in the
# tool_input. Falls back silently if the shape is unexpected.
file_path="$(printf '%s' "$payload" \
  | grep -o '"file_path"[[:space:]]*:[[:space:]]*"[^"]*"' \
  | head -n1 \
  | sed 's/.*"file_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')"

[ -z "${file_path:-}" ] && exit 0
[ -f "$file_path" ] || exit 0
case "$file_path" in
  *.py) ;;
  *) exit 0 ;;
esac

command -v ruff >/dev/null 2>&1 || exit 0

# Run inside the file's project root if it has one; otherwise current dir is fine.
ruff format --quiet "$file_path" 2>&1 || true
ruff check --fix --quiet "$file_path" 2>&1 || true

exit 0
