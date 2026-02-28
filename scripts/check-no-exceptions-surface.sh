#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

pattern='(^|[^[:alnum:]_])throw[[:space:]]+|(^|[^[:alnum:]_])try[[:space:]]*\{|(^|[^[:alnum:]_])catch[[:space:]]*\('

mapfile -t hits < <(grep -RInE "$pattern" include src || true)
if [ "${#hits[@]}" -eq 0 ]; then
    echo "No-exceptions surface check: OK (no exception keywords found)."
    exit 0
fi

allowed_prefixes=(
    "src/core/error.cpp:"
)

violations=()
for hit in "${hits[@]}"; do
    allowed=false
    for prefix in "${allowed_prefixes[@]}"; do
        if [[ "$hit" == "$prefix"* ]]; then
            allowed=true
            break
        fi
    done
    if [ "$allowed" = false ]; then
        violations+=("$hit")
    fi
done

if [ "${#violations[@]}" -ne 0 ]; then
    echo "Disallowed exception keywords in include/src:"
    printf '  %s\n' "${violations[@]}"
    exit 1
fi

echo "No-exceptions surface check: OK (${#hits[@]} allowed occurrence(s))."
