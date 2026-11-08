$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
docker compose stop
if ($LASTEXITCODE -ne 0) { throw "docker compose stop failed with exit code $LASTEXITCODE" }
Write-Host "Paper-trading services stopped. Docker volume and SQLite trading data were preserved."
