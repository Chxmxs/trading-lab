[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string[]]$Symbols,
  [Parameter(Mandatory=$true)][string]$Start,
  [Parameter(Mandatory=$true)][string]$End,
  [Parameter(Mandatory=$true)][string]$Tf,
  [string]$Registry = 'configs/data_registry.yaml'
)

# Normalize: if user passed one item like "BTCUSD,ETHUSD", split it.
$norm = @()
foreach ($s in $Symbols) { $norm += ($s -split ',') }
# trim whitespace
$norm = $norm | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' }

Write-Host "Loading bar data..."
python adapters/data_adapters.py query --symbols $norm --start $Start --end $End --tf $Tf --registry $Registry
