# Regenerate requirements.lock for all supported platforms (Linux Docker, macOS, Windows).
# Requires Docker Desktop. Run from the repository root in PowerShell.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

docker run --rm `
  -v "${Root}:/work" `
  -w /work `
  python:3.12.13-slim `
  bash -c "pip install --no-cache-dir uv==0.9.7 && uv pip compile requirements.txt --python-version 3.12 --universal --generate-hashes -o requirements.lock"

Write-Host "Wrote requirements.lock (universal). Rebuild: docker compose --profile batch build --no-cache"
