Write-Host "Rebuilding Virtual Environment for CUDA GPU Acceleration..." -ForegroundColor Cyan
Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Try Python 3.12 (Supported by cu121 wheels), fallback to default python
$py312 = py -3.12 --version 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "Using Python 3.12..." -ForegroundColor Green
    py -3.12 -m venv .venv
} else {
    Write-Host "Python 3.12 not detected via py launcher, using system Python..." -ForegroundColor Yellow
    python -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt
.\.venv\Scripts\pip.exe uninstall -y llama-cpp-python 2>$null
.\.venv\Scripts\pip.exe install llama-cpp-python --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --prefer-binary

Write-Host "Environment rebuilt. Running Benchmark with local Micro-LLM..." -ForegroundColor Green
.\.venv\Scripts\python.exe scripts/run_benchmark.py
