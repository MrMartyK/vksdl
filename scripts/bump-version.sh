#!/usr/bin/env bash
# Usage: ./scripts/bump-version.sh <major.minor.patch>
# Updates version in all 3 canonical locations + reminds about CHANGELOG.
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <major.minor.patch>"
    echo "Example: $0 0.2.0"
    exit 1
fi

VERSION="$1"

if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: version must be in format major.minor.patch (e.g. 0.2.0)"
    exit 1
fi

MAJOR=$(echo "$VERSION" | cut -d. -f1)
MINOR=$(echo "$VERSION" | cut -d. -f2)
PATCH=$(echo "$VERSION" | cut -d. -f3)

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# 1. CMakeLists.txt
sed -i "s/project(vksdl VERSION [0-9]*\.[0-9]*\.[0-9]*/project(vksdl VERSION ${VERSION}/" \
    "$REPO_ROOT/CMakeLists.txt"

# 2. vcpkg.json
sed -i "s/\"version\": \"[0-9]*\.[0-9]*\.[0-9]*\"/\"version\": \"${VERSION}\"/" \
    "$REPO_ROOT/vcpkg.json"

# 3. version.hpp
cat > "$REPO_ROOT/include/vksdl/version.hpp" << EOF
#pragma once

#define VKSDL_VERSION_MAJOR ${MAJOR}
#define VKSDL_VERSION_MINOR ${MINOR}
#define VKSDL_VERSION_PATCH ${PATCH}

#define VKSDL_VERSION_STRING "${VERSION}"
EOF

echo "Version updated to ${VERSION} in:"
echo "  - CMakeLists.txt"
echo "  - vcpkg.json"
echo "  - include/vksdl/version.hpp"
echo ""
echo "Remember to:"
echo "  1. Update CHANGELOG.md with a ## [${VERSION}] section"
echo "  2. Commit and tag: git tag v${VERSION}"
