param(
    [string]$BaseRef = "origin/main",
    [string]$ToolchainFile = "vcpkg/scripts/buildsystems/vcpkg.cmake",
    [string]$ClangFormatBin = ""
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Action
    )
    Write-Host "==> $Name"
    & $Action
    Write-Host "PASS: $Name"
}

function Invoke-Cmd {
    param([string]$Cmd)
    Write-Host "   $Cmd"
    Invoke-Expression $Cmd
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $Cmd"
    }
}

$range = "HEAD~1..HEAD"
git rev-parse --verify $BaseRef *> $null
if ($LASTEXITCODE -eq 0) {
    $mergeBase = (git merge-base $BaseRef HEAD).Trim()
    if (-not [string]::IsNullOrWhiteSpace($mergeBase)) {
        $range = "$mergeBase..HEAD"
    }
}

Invoke-Step "clang-format check" {
    $fmtCmd = "powershell -NoProfile -ExecutionPolicy Bypass -File scripts/clang-format.ps1 -Check"
    if (-not [string]::IsNullOrWhiteSpace($ClangFormatBin)) {
        $fmtCmd += " -ClangFormatBin `"$ClangFormatBin`""
    }
    $fmtCmd += " -BaseRef `"$BaseRef`""
    Invoke-Cmd $fmtCmd
}

Invoke-Step "configure + build + full tests" {
    Invoke-Cmd "cmake -S . -B build-verify-local -G Ninja -DCMAKE_TOOLCHAIN_FILE=`"$ToolchainFile`" -DVKSDL_BUILD_TESTS=ON -DVKSDL_BUILD_EXAMPLES=ON"
    Invoke-Cmd "cmake --build build-verify-local -j 8"
    Invoke-Cmd "ctest --test-dir build-verify-local --output-on-failure"
}

Invoke-Step "no-exceptions build" {
    Invoke-Cmd "cmake -S . -B build-noexc-verify-local -G Ninja -DCMAKE_TOOLCHAIN_FILE=`"$ToolchainFile`" -DVKSDL_BUILD_TESTS=OFF -DVKSDL_BUILD_EXAMPLES=OFF -DVKSDL_ENABLE_EXCEPTIONS=OFF -DCMAKE_CXX_FLAGS=-fno-exceptions"
    Invoke-Cmd "cmake --build build-noexc-verify-local -j 8"
}

Invoke-Step "commit message policy" {
    $messages = git log --format=%B $range
    if ($messages -match "(?im)(co-authored-by:|\bphase\b)") {
        throw "Commit message policy check failed in range $range"
    }
    if ($messages -match "[^\u0000-\u007F]") {
        throw "Commit messages contain non-ASCII characters in range $range"
    }
}

Invoke-Step "no markdown in range" {
    $mdFiles = git diff --name-only $range -- '*.md'
    if (-not [string]::IsNullOrWhiteSpace(($mdFiles -join ""))) {
        throw "Markdown files found in commit range ${range}:`n$($mdFiles -join "`n")"
    }
}

Write-Host "All verify-local gates passed."
