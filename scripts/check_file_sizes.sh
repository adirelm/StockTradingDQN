#!/usr/bin/env bash
# Enforce the lecturer's file-size rule (submission_guidelines §3.2):
#   "Every code file must not exceed 150 lines of code.
#    Empty lines and comment lines do NOT count."
#
# A "code line" is a non-blank line whose first non-whitespace
# character is not '#'. Docstrings count as code lines (they're
# regular string statements, not comments) which is consistent with
# the lecturer's "comment lines don't count" wording: he means '#'
# comments specifically.
set -euo pipefail

LIMIT=150
violations=()
while IFS= read -r -d '' file; do
    code_lines=$(grep -cvE '^\s*$|^\s*#' "$file" || true)
    if [ "$code_lines" -gt "$LIMIT" ]; then
        violations+=("$file: $code_lines code lines")
    fi
done < <(find src tests scripts analysis -name '*.py' -print0)

if [ ${#violations[@]} -gt 0 ]; then
    echo "Files exceeding ${LIMIT}-line code limit (blank + '#' comment lines excluded):" >&2
    printf '  %s\n' "${violations[@]}" >&2
    exit 1
fi
echo "All Python files within ${LIMIT}-line code limit (per submission guidelines §3.2)."
