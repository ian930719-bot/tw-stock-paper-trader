$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$LogRoot = Join-Path $env:LOCALAPPDATA "TwStockPaperTrader"
$LogFile = Join-Path $LogRoot "startup.log"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
Start-Transcript -Path $LogFile -Append | Out-Null
try {
    Set-Location $ProjectRoot
    if (-not (Test-Path (Join-Path $ProjectRoot ".env"))) {
        Copy-Item (Join-Path $ProjectRoot ".env.example") (Join-Path $ProjectRoot ".env")
        Write-Output "Created .env from .env.example. Automatic simulated entries remain disabled."
    }
    $dockerDesktop = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"
    if (-not (docker info 2>$null)) {
        if (Test-Path $dockerDesktop) {
            Write-Output "Starting Docker Desktop..."
            Start-Process -FilePath $dockerDesktop
        }
    }
    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        try {
            docker info *> $null
            if ($LASTEXITCODE -eq 0) { $ready = $true; break }
        } catch { }
        Start-Sleep -Seconds 5
    }
    if (-not $ready) { throw "Docker Engine did not become ready within 5 minutes. Open Docker Desktop and inspect its status." }
    docker compose up -d --build
    if ($LASTEXITCODE -ne 0) { throw "docker compose up failed with exit code $LASTEXITCODE" }
    docker compose ps
    Write-Output "Paper-trading services started. PAPER_AUTO_ENTRIES in .env controls simulated BUY orders."
} catch {
    Write-Error $_
    exit 1
} finally {
    Stop-Transcript | Out-Null
}
