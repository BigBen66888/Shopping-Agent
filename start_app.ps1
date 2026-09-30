$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$pidFile = Join-Path $projectRoot "data\shopping-agent.pid"
$portFile = Join-Path $projectRoot "data\shopping-agent.port"
$logDir = Join-Path $projectRoot "reports"
$stdoutLog = Join-Path $logDir "server.log"
$stderrLog = Join-Path $logDir "server-error.log"
$modelCache = Join-Path $projectRoot "data\model_cache"
# Force a fresh HTML and CSS URL whenever either frontend file changes.
$frontendIndex = Join-Path $projectRoot "frontend\index.html"
$frontendCss = Join-Path $projectRoot "frontend\ui.css"
$frontendVersion = [string](Get-Item -LiteralPath $frontendIndex).LastWriteTimeUtc.Ticks
if (Test-Path -LiteralPath $frontendCss) {
    $frontendVersion += "-" + (Get-Item -LiteralPath $frontendCss).LastWriteTimeUtc.Ticks
}

# 项目自带 BGE-M3 与 BGE-reranker-v2-m3 权重（data/model_cache），把 HF_HOME 指过去
# 就能完全离线加载。若指向默认的 ~/.cache/huggingface，transformers 找不到权重时会去
# 联网拉 4.6 GB。子进程继承当前进程环境变量，所以在这里设置即可生效。
if (Test-Path -LiteralPath $modelCache) {
    $env:HF_HOME = $modelCache
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
} else {
    Write-Host "Model cache not found at $modelCache; models may be downloaded from the internet." -ForegroundColor Yellow
}

function Get-ListeningProcessId([int]$Port) {
    $pattern = "^\s*TCP\s+127\.0\.0\.1:$Port\s+\S+\s+LISTENING\s+(\d+)\s*$"
    foreach ($line in (netstat -ano -p TCP)) {
        if ($line -match $pattern) { return [int]$Matches[1] }
    }
    return $null
}

if (-not (Test-Path -LiteralPath $pythonPath)) {
    Write-Host "Project virtual environment was not found: $pythonPath" -ForegroundColor Red
    Write-Host "Run this first in Shopping_agent: py -3 -m venv .venv" -ForegroundColor Yellow
    exit 1
}

& $pythonPath -c "import uvicorn" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Dependencies are missing. Installing requirements.txt ..." -ForegroundColor Yellow
    & $pythonPath -m pip install -r (Join-Path $projectRoot "requirements.txt")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (Test-Path -LiteralPath $pidFile) {
    $savedPid = (Get-Content -LiteralPath $pidFile -Raw).Trim()
    if ($savedPid -match '^\d+$') {
        $existingPort = if (Test-Path -LiteralPath $portFile) { (Get-Content -LiteralPath $portFile -Raw).Trim() } else { "8000" }
        $listenerProcessId = if ($existingPort -match '^\d+$') { Get-ListeningProcessId ([int]$existingPort) } else { $null }
        $isExistingServer = $listenerProcessId -eq [int]$savedPid
        if ($isExistingServer) {
            $serverStarted = $null
            try { $serverStarted = (Get-Process -Id ([int]$savedPid) -ErrorAction Stop).StartTime } catch {}
            $backendFiles = @(Get-ChildItem -LiteralPath (Join-Path $projectRoot "shopping_agent") -Recurse -File -Filter "*.py")
            $envFile = Join-Path $projectRoot ".env"
            if (Test-Path -LiteralPath $envFile) { $backendFiles += Get-Item -LiteralPath $envFile }
            $latestSourceChange = ($backendFiles | Measure-Object -Property LastWriteTime -Maximum).Maximum
            if ($serverStarted -and $latestSourceChange -le $serverStarted) {
                $url = "http://127.0.0.1:$existingPort"
                Write-Host "Shopping Agent is already running with current backend code (PID $savedPid)." -ForegroundColor Green
                if ($env:SHOPPING_AGENT_NO_BROWSER -ne "1") { Start-Process "$url/?v=$frontendVersion" }
                exit 0
            }
            Write-Host "Backend code changed; restarting the managed Shopping Agent process (PID $savedPid)." -ForegroundColor Yellow
            Stop-Process -Id ([int]$savedPid) -Force
            Start-Sleep -Milliseconds 300
        }
    }
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $portFile -Force -ErrorAction SilentlyContinue
}

function Test-LocalPortAvailable([int]$Port) {
    $probe = $null
    try {
        $probe = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
        $probe.Start()
        return $true
    } catch {
        return $false
    } finally {
        if ($probe) { $probe.Stop() }
    }
}

$selectedPort = 8000..8010 | Where-Object { Test-LocalPortAvailable $_ } | Select-Object -First 1
if (-not $selectedPort) {
    Write-Host "Ports 8000-8010 are all in use." -ForegroundColor Red
    exit 1
}
$url = "http://127.0.0.1:$selectedPort"

New-Item -ItemType Directory -Path $logDir -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path $pidFile) -Force | Out-Null

$serverProcess = $null
try {
    $serverProcess = Start-Process `
        -FilePath $pythonPath `
        -ArgumentList @("-m", "uvicorn", "shopping_agent.api:app", "--host", "127.0.0.1", "--port", "$selectedPort") `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -PassThru
} catch {
    # Start-Process 内部把环境变量装进大小写不敏感的字典，若同时存在 HTTP_PROXY 与
    # http_proxy 会抛 "Item has already been added"。PowerShell 无法只删掉其中一个，
    # 所以这里给出可操作的提示，而不是让用户面对一条看不懂的异常。
    Write-Host "Failed to launch the API process: $($_.Exception.Message)" -ForegroundColor Red
    if ($_.Exception.Message -match 'PROXY|proxy') {
        Write-Host "Detected duplicate proxy variables that differ only in letter case (HTTP_PROXY and http_proxy)." -ForegroundColor Yellow
        Write-Host "Remove one of them (Windows treats them as the same variable) and run this launcher again." -ForegroundColor Yellow
    }
    exit 1
}

Set-Content -LiteralPath $pidFile -Value $serverProcess.Id -Encoding ascii
Set-Content -LiteralPath $portFile -Value $selectedPort -Encoding ascii
Write-Host "Starting Shopping Agent (launcher PID $($serverProcess.Id)) ..." -ForegroundColor Cyan

$ready = $false
# The first launch over the full 2.75M-product corpus builds the BM25 inverted index,
# which takes several minutes; later launches restore it from data\bm25_cache.pkl.
# Wait up to 20 minutes instead of 30 seconds so the first run is not killed.
$maxAttempts = 1200
$startedAt = Get-Date
for ($attempt = 0; $attempt -lt $maxAttempts; $attempt++) {
    if ($serverProcess.HasExited) { break }
    try {
        $health = Invoke-RestMethod -Uri "$url/health" -TimeoutSec 2
        if ($health.status -eq "ok") {
            $ready = $true
            break
        }
    } catch {
        if ($attempt -gt 0 -and $attempt % 15 -eq 0) {
            $elapsed = [int]((Get-Date) - $startedAt).TotalSeconds
            Write-Host "Still preparing the index ... ${elapsed}s (see reports\server.log for progress)" -ForegroundColor DarkGray
        }
        Start-Sleep -Seconds 1
    }
}

if (-not $ready) {
    Write-Host "API startup failed. Read this log: $stderrLog" -ForegroundColor Red
    if (-not $serverProcess.HasExited) { Stop-Process -Id $serverProcess.Id -Force }
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $portFile -Force -ErrorAction SilentlyContinue
    exit 1
}

$listenerProcessId = Get-ListeningProcessId $selectedPort
if (-not $listenerProcessId) {
    Write-Host "API responded, but its listener process could not be verified." -ForegroundColor Red
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $portFile -Force -ErrorAction SilentlyContinue
    exit 1
}
Set-Content -LiteralPath $pidFile -Value $listenerProcessId -Encoding ascii

Write-Host "API is running in the background: $url (PID $listenerProcessId)" -ForegroundColor Green
Write-Host "Closing this window will not stop it. Double-click stop-shopping-agent.cmd to stop."
if ($env:SHOPPING_AGENT_NO_BROWSER -ne "1") { Start-Process "$url/?v=$frontendVersion" }
