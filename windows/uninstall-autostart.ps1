$ErrorActionPreference = "Stop"
$TaskName = "TW Stock Paper Trader"
$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($task) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed scheduled task: $TaskName"
} else {
    Write-Host "Scheduled task not found: $TaskName"
}
Write-Host "Running Docker containers were not stopped and trading data was not deleted."
