#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

pattern='vkDeviceWaitIdle[[:space:]]*\(|vkQueueWaitIdle[[:space:]]*\(|vkWaitForFences[[:space:]]*\(|vkWaitSemaphores[[:space:]]*\(|VK_QUERY_RESULT_WAIT_BIT'
tag='VKSDL_BLOCKING_WAIT:'

total=0
untagged=0

while IFS=: read -r file line text; do
    [ -z "$file" ] && continue
    total=$((total + 1))

    start=$((line > 8 ? line - 8 : 1))
    end=$((line + 1))
    if ! sed -n "${start},${end}p" "$file" | grep -q "$tag"; then
        echo "Untagged blocking wait: ${file}:${line}:${text}"
        untagged=$((untagged + 1))
    fi
done < <(grep -RInE "$pattern" src || true)

echo "Blocking waits found: ${total}"
if [ "$untagged" -ne 0 ]; then
    echo "Blocking wait check failed: ${untagged} untagged occurrence(s)."
    exit 1
fi

echo "Blocking wait check: OK"
