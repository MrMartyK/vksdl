#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---check}"
if [[ "${MODE}" != "--check" && "${MODE}" != "--fix" ]]; then
    echo "Usage: $0 [--check|--fix]"
    exit 2
fi
if ! command -v git >/dev/null 2>&1; then
    echo "git not found"
    exit 1
fi

CLANG_FORMAT_BIN="${CLANG_FORMAT_BIN:-clang-format-18}"
if [[ "${CLANG_FORMAT_BIN}" =~ ^[A-Za-z]:/ ]] && command -v cygpath >/dev/null 2>&1; then
    CLANG_FORMAT_BIN="$(cygpath -u "${CLANG_FORMAT_BIN}")"
fi
if [[ -f "${CLANG_FORMAT_BIN}" ]]; then
    :
elif ! command -v "${CLANG_FORMAT_BIN}" >/dev/null 2>&1; then
    CLANG_FORMAT_BIN="clang-format"
fi
if [[ -f "${CLANG_FORMAT_BIN}" ]]; then
    :
elif ! command -v "${CLANG_FORMAT_BIN}" >/dev/null 2>&1; then
    echo "clang-format not found (tried clang-format-18 and clang-format)"
    exit 1
fi

mapfile -t FILES < <(
    (
        {
            git diff --name-only --diff-filter=ACMR
            git diff --cached --name-only --diff-filter=ACMR
        } |
        sort -u |
        grep -E '\.(c|cc|cpp|cxx|h|hpp|hh|hxx|mm)$' |
        grep -Ev '^(third_party/|vcpkg/|build/|build-|\.playwright-mcp/|\.github/)'
    ) || true
)

if [[ "${#FILES[@]}" -eq 0 ]]; then
    BASE_REF="${FORMAT_BASE_REF:-origin/main}"
    if git rev-parse --verify "${BASE_REF}" >/dev/null 2>&1; then
        BASE_COMMIT="$(git merge-base "${BASE_REF}" HEAD)"
        DIFF_RANGE="${BASE_COMMIT}...HEAD"
    else
        DIFF_RANGE="HEAD~1..HEAD"
    fi

    mapfile -t FILES < <(
        (
            git diff --name-only --diff-filter=ACMR "${DIFF_RANGE}" |
            grep -E '\.(c|cc|cpp|cxx|h|hpp|hh|hxx|mm)$' |
            grep -Ev '^(third_party/|vcpkg/|build/|build-|\.playwright-mcp/|\.github/)'
        ) || true
    )
fi

if [[ "${#FILES[@]}" -eq 0 ]]; then
    echo "No files to format"
    exit 0
fi

if [[ "${MODE}" == "--fix" ]]; then
    echo "Formatting ${#FILES[@]} files with ${CLANG_FORMAT_BIN}"
    "${CLANG_FORMAT_BIN}" -i "${FILES[@]}"
    echo "Format complete"
    exit 0
fi

FAILED=0
echo "Checking ${#FILES[@]} files with ${CLANG_FORMAT_BIN}"
for f in "${FILES[@]}"; do
    if ! "${CLANG_FORMAT_BIN}" --dry-run --Werror "${f}" >/dev/null 2>&1; then
        echo "Needs formatting: ${f}"
        FAILED=1
    fi
done

if [[ "${FAILED}" -ne 0 ]]; then
    echo "clang-format check failed"
    exit 1
fi

echo "clang-format check passed"
