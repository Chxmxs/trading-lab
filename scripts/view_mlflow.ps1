# Opens MLflow UI if installed
param([string]$Store = ".\mlruns")
$ErrorActionPreference = "Stop"
Write-Host "Starting MLflow UI on $Store ..."
mlflow ui --backend-store-uri $Store
