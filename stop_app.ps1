$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$pidFile = Join-Path $projectRoot "data\shopping-agent.pid"
$portFile = Join-Path $projectRoot "data\shopping-agent.port"

if (-not (Test-Path -LiteralPath $pidFile)) {
    Write-Host "No managed Shopping Agent process was found." -ForegroundColor Yellow
    exit 0
}

$savedPid = (Get-Content -LiteralPath $pidFile -Raw).Trim()
if ($savedPid -notmatch '^\d+$') {
    Remove-Item -LiteralPath $pidFile -Force
    Remove-Item -LiteralPath $portFile -Force -ErrorAction SilentlyContinue
    Write-Host "Invalid PID state was removed." -ForegroundColor Yellow
    exit 0
}

$savedPort = if (Test-Path -LiteralPath $portFile) { (Get-Content -LiteralPath $portFile -Raw).Trim() } else { "" }
if ($savedPort -notmatch '^\d+$') {
    Remove-Item -LiteralPath $pidFile -Force
    Remove-Item -LiteralPath $portFile -Force -ErrorAction SilentlyContinue
    Write-Host "Invalid port state was removed. No process was stopped." -ForegroundColor Yellow
    exit 0
}

$listenerProcessId = $null
$pattern = "^\s*TCP\s+127\.0\.0\.1:$savedPort\s+\S+\s+LISTENING\s+(\d+)\s*$"
foreach ($line in (netstat -ano -p TCP)) {
    if ($line -match $pattern) { $listenerProcessId = [int]$Matches[1]; break }
}

if (-not $listenerProcessId) {
    Remove-Item -LiteralPath $pidFile -Force
    Remove-Item -LiteralPath $portFile -Force -ErrorAction SilentlyContinue
    Write-Host "The background service has already exited. State files were removed." -ForegroundColor Yellow
    exit 0
}

if ($listenerProcessId -ne [int]$savedPid) {
    Write-Host "Port $savedPort belongs to PID $listenerProcessId, not recorded PID $savedPid. No process was stopped." -ForegroundColor Red
    exit 1
}

Stop-Process -Id ([int]$savedPid) -Force
Remove-Item -LiteralPath $pidFile -Force
Remove-Item -LiteralPath $portFile -Force -ErrorAction SilentlyContinue
Write-Host "Shopping Agent was stopped." -ForegroundColor Green
