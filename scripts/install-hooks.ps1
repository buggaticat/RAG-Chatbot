param()

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$hooksPath = Join-Path $repoRoot ".githooks"
$hookFile = Join-Path $hooksPath "pre-push"

if (-not (Test-Path $hookFile)) {
    throw "Expected hook file not found: $hookFile"
}

git config --local core.hooksPath ".githooks" | Out-Null

Write-Host "Installed local Git hooks path: .githooks"
Write-Host "Pre-push hook will now block pushes when targeted tests fail."
