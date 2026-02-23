#!/usr/bin/env bash
# Verifies version is consistent across all canonical locations.
# Exit 0 if in sync, exit 1 if mismatched.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Extract versions
CMAKE_VER=$(grep -oP 'project\(vksdl VERSION \K[0-9]+\.[0-9]+\.[0-9]+' "$REPO_ROOT/CMakeLists.txt")
VCPKG_VER=$(grep -oP '"version": "\K[0-9]+\.[0-9]+\.[0-9]+' "$REPO_ROOT/vcpkg.json")
HPP_VER=$(grep -oP 'VKSDL_VERSION_STRING "\K[0-9]+\.[0-9]+\.[0-9]+' "$REPO_ROOT/include/vksdl/version.hpp")

MISMATCH=0

if [ "$CMAKE_VER" != "$VCPKG_VER" ]; then
    echo "Version mismatch: CMakeLists.txt=${CMAKE_VER} vs vcpkg.json=${VCPKG_VER}"
    MISMATCH=1
fi

if [ "$CMAKE_VER" != "$HPP_VER" ]; then
    echo "Version mismatch: CMakeLists.txt=${CMAKE_VER} vs version.hpp=${HPP_VER}"
    MISMATCH=1
fi

if [ "$MISMATCH" -eq 0 ]; then
    echo "Version sync OK: ${CMAKE_VER}"
else
    echo ""
    echo "Run: ./scripts/bump-version.sh <version> to fix"
    exit 1
fi
