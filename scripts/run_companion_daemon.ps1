# Companion Intelligence Daemon runner (uses tradebot env)
$ErrorActionPreference = "Stop"
$root = "C:\Users\Gladiator\trading-lab"
Set-Location $root

Write-Host "[daemon] Launching Intelligence Loop via conda run (env=tradebot)..."
# Print which python is used (sanity check)
conda run -n tradebot python -c "import sys; print('[python]', sys.executable)"

# Loop mode: adjust sleep/batch as desired
conda run -n tradebot python -m companion.ai_loop.daemon --loop --sleep 15 --consume-batch 3