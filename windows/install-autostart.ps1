$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$StartScript = Join-Path $PSScriptRoot "start-paper-trader.ps1"
$TaskName = "TW Stock Paper Trader"
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker CLI not found. Install Docker Desktop first: https://www.docker.com/products/docker-desktop/"
}
if (-not (Test-Path (Join-Path $ProjectRoot "docker-compose.yml"))) {
    throw "docker-compose.yml not found. Run this script from the cloned project folder."
}
if (-not (Test-Path (Join-Path $ProjectRoot ".env"))) {
    Copy-Item (Join-Path $ProjectRoot ".env.example") (Join-Path $ProjectRoot ".env")
    Write-Host "Created .env from .env.example. PAPER_AUTO_ENTRIES defaults to false."
}
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument ('-NoProfile -ExecutionPolicy Bypass -File "' + $StartScript + '"') -WorkingDirectory $ProjectRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero)
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description "Starts the local Taiwan-stock paper-trading Docker Compose services after this Windows user signs in." -Force | Out-Null
Write-Host ""
Write-Host "Installed scheduled task: $TaskName"
Write-Host "It runs when you sign in to Windows (not before sign-in)."
Write-Host "Docker Desktop will be started by the script if its engine is not ready."
Write-Host "To start now, run: .\windows\start-paper-trader.ps1"
Write-Host "Startup log: $env:LOCALAPPDATA\TwStockPaperTrader\startup.log"
