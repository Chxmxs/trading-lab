param(
  [Parameter(Mandatory=$true)][string]$Strategy,
  [Parameter(Mandatory=$true)][string]$Symbol,
  [Parameter(Mandatory=$true)][string]$Timeframe,
  [string]$Start = "2025-01-01T00:00:00Z",
  [string]$End   = "2025-01-05T00:00:00Z",
  [int]$NSplits = 5,
  [string]$Embargo = "2D",
  [string]$LabelStoreDir = "label_store"
)

$ErrorActionPreference = "Stop"
$proj = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $proj
Set-Location $root

# Make project importable
$env:PYTHONPATH = $root

# IMPORTANT: you must have 'tradebot-mlfl' env already created and activated manually
# OR call: conda activate tradebot-mlfl
Write-Host "Using current Python: " (python -c "import sys; print(sys.executable)")

python ".\scripts\cv_generate_mlfl.py" `
  --strategy $Strategy `
  --symbol $Symbol `
  --timeframe $Timeframe `
  --start $Start `
  --end $End `
  --n_splits $NSplits `
  --embargo $Embargo `
  --label_store_dir $LabelStoreDir
