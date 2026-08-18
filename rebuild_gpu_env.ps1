Write-Host "Rebuilding Virtual Environment for CUDA GPU Acceleration..." -ForegroundColor Cyan
Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Force Python 3.12 (Supported by cu121 wheels)
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt
.\.venv\Scripts\pip.exe uninstall -y llama-cpp-python
.\.venv\Scripts\pip.exe install llama-cpp-python --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

Write-Host "Environment rebuilt. Running Benchmark with RTX 3050..." -ForegroundColor Green
.\.venv\Scripts\python.exe scripts/run_benchmark.py
