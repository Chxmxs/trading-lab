[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$Name,
  [ValidateSet('ohlcv','metric')][string]$Kind = 'ohlcv',
  [string]$Registry = 'configs/data_registry.yaml'
)

if ($Kind -eq 'ohlcv') {
  python adapters/data_adapters.py audit-ohlcv --name $Name --registry $Registry
}
else {
  python adapters/data_adapters.py audit-metric --name $Name --registry $Registry
}
