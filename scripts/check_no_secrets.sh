#!/usr/bin/env bash
# §7.4 — secret-pattern scan over the staged tree.
#
# Auditable evidence (not just claim) that no obvious secret has landed.
# Runs at every commit via .pre-commit-config.yaml and complements the
# .gitignore secret-pattern block. The patterns below are deliberately
# conservative — false positives would block development for no gain.
#
# What we detect:
#   - AWS access-key / secret-key shapes
#   - Slack / GitHub / Anthropic / OpenAI token prefixes
#   - Common "password = 'literal'" / "api_key = 'literal'" assignments
#
# Allowlist:
#   - .gitignore / .env-example (documentation of patterns)
#   - .secrets.baseline (kept for reference / future detect-secrets)
#   - tests/ (test fixtures may legitimately use placeholder strings)
#   - uv.lock (hash strings look like high-entropy secrets)
#   - results/, assets/, *.png, *.ipynb (binary / generated)

set -euo pipefail

# NOTE: each alternative is anchored later via "^($EXCLUDE_DIRS)" so we
# must use trailing `/` (or `$`) on directory prefixes — without it,
# `\.git` matches `.github/` and silently exempts workflow files from
# the scan. `.github/` is intentionally NOT excluded: CI workflows are
# exactly where a real GITHUB_TOKEN / API key would accidentally land.
# Pass-5 §7 finding F7.12 caught this.
EXCLUDE_DIRS='\.git/|\.venv/|\.pytest_cache/|\.ruff_cache/|__pycache__/|results/|assets/|notebooks/|tests/|uv\.lock|\.secrets\.baseline|\.gitignore|\.env-example|.*\.png'

PATTERNS=(
    'AKIA[0-9A-Z]{16}'                   # AWS access key
    'aws_secret_access_key\s*=\s*[A-Za-z0-9/+]{40}'
    'xox[bpaors]-[A-Za-z0-9-]{10,}'      # Slack tokens
    'ghp_[A-Za-z0-9]{36}'                # GitHub PAT
    'sk-ant-[A-Za-z0-9-]{20,}'           # Anthropic key
    'sk-[A-Za-z0-9]{48}'                 # OpenAI key
    "(password|api_key|api_token|secret_key)\s*=\s*['\"][A-Za-z0-9_/+=-]{12,}['\"]"
)

found=0
for pat in "${PATTERNS[@]}"; do
    if matches=$(git ls-files | grep -vE "^($EXCLUDE_DIRS)" \
                  | xargs -I {} grep -EHn "$pat" {} 2>/dev/null); then
        if [ -n "$matches" ]; then
            echo "Possible secret matching pattern: $pat"
            echo "$matches"
            found=1
        fi
    fi
done

if [ "$found" -eq 1 ]; then
    echo
    echo "Secret-pattern scan FAILED. If a match is a false positive, narrow"
    echo "the pattern in scripts/check_no_secrets.sh or move the file to"
    echo "an excluded directory."
    exit 1
fi

exit 0
