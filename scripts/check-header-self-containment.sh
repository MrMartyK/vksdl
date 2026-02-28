#!/usr/bin/env bash
# Verifies each public header compiles as a standalone translation unit.
# Requires: a C++20 compiler, Vulkan headers, SDL3 headers on include path.
# Usage: CXX=g++-14 bash scripts/check-header-self-containment.sh [build_dir]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${1:-${REPO_ROOT}/build}"
CXX="${CXX:-c++}"

# Extract -I and -isystem flags from compile_commands.json.
INCLUDE_FLAGS="-I${REPO_ROOT}/include"
if [ -f "${BUILD_DIR}/compile_commands.json" ]; then
    extra=$(grep -oE '(-I|-isystem )[^ "]+' "${BUILD_DIR}/compile_commands.json" \
            | sort -u \
            | grep -v 'spirv-reflect' \
            || true)
    if [ -n "$extra" ]; then
        INCLUDE_FLAGS="${INCLUDE_FLAGS} ${extra}"
    fi
fi

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

FAILED=0
TOTAL=0

for header in "${REPO_ROOT}"/include/vksdl/*.hpp; do
    name="$(basename "$header" .hpp)"
    TOTAL=$((TOTAL + 1))

    # vksdl.hpp is the umbrella -- skip it, it includes everything.
    if [ "$name" = "vksdl" ]; then
        continue
    fi

    src="${TMPDIR}/${name}_test.cpp"
    echo "#include <vksdl/${name}.hpp>" > "$src"
    echo "int main() { return 0; }" >> "$src"

    printf "  %-45s " "vksdl/${name}.hpp"
    # shellcheck disable=SC2086
    if $CXX -std=c++20 -fsyntax-only $INCLUDE_FLAGS "$src" 2>"${TMPDIR}/${name}.err"; then
        echo "ok"
    else
        echo "FAIL"
        cat "${TMPDIR}/${name}.err"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "Checked ${TOTAL} headers, ${FAILED} failed."
if [ "$FAILED" -ne 0 ]; then
    exit 1
fi
echo "Header self-containment check: OK"
