Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  GPU Environment Rebuild Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Step 1: Kill ALL Python processes to release .pyd file locks
Write-Host "`n[1/5] Killing all Python processes to release file locks..." -ForegroundColor Yellow
Get-Process -Name python*, Python* -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3

# Step 2: Force-delete the old .venv
Write-Host "[2/5] Removing old .venv directory..." -ForegroundColor Yellow
if (Test-Path .venv) {
    cmd /c "rd /s /q .venv" 2>$null
    Start-Sleep -Seconds 2
    if (Test-Path .venv) {
        Write-Host "  WARNING: Some files still locked. Trying again..." -ForegroundColor Red
        cmd /c "rd /s /q .venv" 2>$null
        Start-Sleep -Seconds 2
    }
}
Write-Host "  .venv removed." -ForegroundColor Green

# Step 3: Create new venv (try 3.12, then 3.11, then system python)
Write-Host "[3/5] Creating new virtual environment..." -ForegroundColor Yellow
$pythonCmd = $null
if (Get-Command "py" -ErrorAction SilentlyContinue) {
    $versions = @("3.12", "3.11", "3.10")
    foreach ($v in $versions) {
        $result = py -$v --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $pythonCmd = "py -$v"
            Write-Host "  Using Python $v via py launcher" -ForegroundColor Green
            break
        }
    }
}
if (-not $pythonCmd) {
    $pythonCmd = "python"
    Write-Host "  Using system Python ($(python --version))" -ForegroundColor Yellow
}

Invoke-Expression "$pythonCmd -m venv .venv"
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "  FATAL: Failed to create .venv. Exiting." -ForegroundColor Red
    exit 1
}

# Step 4: Install dependencies
Write-Host "[4/5] Installing dependencies..." -ForegroundColor Yellow
.\.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
.\.venv\Scripts\pip.exe install -r requirements.txt --quiet
Write-Host "  Dependencies installed." -ForegroundColor Green

# Step 5: Verify
Write-Host "[5/5] Verifying installation..." -ForegroundColor Yellow
.\.venv\Scripts\python.exe -c "from groq import Groq; print('  Groq SDK: OK')"
.\.venv\Scripts\python.exe -c "import fastapi; print('  FastAPI: OK')"
.\.venv\Scripts\python.exe -c "from qdrant_client import QdrantClient; print('  Qdrant Client: OK')"

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  Environment ready!" -ForegroundColor Green
Write-Host "  Start server:  .\.venv\Scripts\uvicorn.exe app.main:app --host 0.0.0.0 --port 8000" -ForegroundColor Cyan
Write-Host "  Run benchmark: .\.venv\Scripts\python.exe scripts\run_benchmark.py" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Green
