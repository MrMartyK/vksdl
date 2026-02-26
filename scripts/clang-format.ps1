param(
    [switch]$Check,
    [switch]$Fix,
    [string]$ClangFormatBin = "",
    [string]$BaseRef = "origin/main"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

if (($Check -and $Fix) -or (-not $Check -and -not $Fix)) {
    Write-Host "Usage: ./scripts/clang-format.ps1 -Check | -Fix"
    exit 2
}

if ([string]::IsNullOrWhiteSpace($ClangFormatBin)) {
    $ClangFormatBin = "clang-format-18"
}

if (-not (Test-Path $ClangFormatBin) -and -not (Get-Command $ClangFormatBin -ErrorAction SilentlyContinue)) {
    $ClangFormatBin = "clang-format"
}

if (-not (Test-Path $ClangFormatBin) -and -not (Get-Command $ClangFormatBin -ErrorAction SilentlyContinue)) {
    Write-Error "clang-format not found (tried clang-format-18 and clang-format)"
}

$files = @(
    git diff --name-only --diff-filter=ACMR
    git diff --cached --name-only --diff-filter=ACMR
) |
    Sort-Object -Unique |
    Where-Object {
        $_ -match '\.(c|cc|cpp|cxx|h|hpp|hh|hxx|mm)$' -and
        $_ -notmatch '^(third_party/|vcpkg/|build/|build-|\.playwright-mcp/|\.github/)'
    }

if (-not $files -or $files.Count -eq 0) {
    $diffRange = "HEAD~1..HEAD"
    git rev-parse --verify $BaseRef *> $null
    if ($LASTEXITCODE -eq 0) {
        $baseCommit = (git merge-base $BaseRef HEAD).Trim()
        if (-not [string]::IsNullOrWhiteSpace($baseCommit)) {
            $diffRange = "$baseCommit...HEAD"
        }
    }

    $files = git diff --name-only --diff-filter=ACMR $diffRange |
        Where-Object {
            $_ -match '\.(c|cc|cpp|cxx|h|hpp|hh|hxx|mm)$' -and
            $_ -notmatch '^(third_party/|vcpkg/|build/|build-|\.playwright-mcp/|\.github/)'
        }
}

if (-not $files -or $files.Count -eq 0) {
    Write-Host "No files to format"
    exit 0
}

if ($Fix) {
    Write-Host "Formatting $($files.Count) files with $ClangFormatBin"
    & $ClangFormatBin -i $files
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "Format complete"
    exit 0
}

Write-Host "Checking $($files.Count) files with $ClangFormatBin"
$failed = @()
foreach ($file in $files) {
    $xml = & $ClangFormatBin --output-replacements-xml $file
    if ($LASTEXITCODE -ne 0 -or ($xml -join "`n") -match "<replacement ") {
        $failed += $file
    }
}

if ($failed.Count -gt 0) {
    foreach ($f in $failed) { Write-Host "Needs formatting: $f" }
    Write-Host "clang-format check failed"
    exit 1
}

Write-Host "clang-format check passed"
exit 0
