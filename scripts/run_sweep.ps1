param([string]$ConfigPath = ".\configs\orchestrator.json")

$ErrorActionPreference = "Stop"
$proj = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $proj
Set-Location $root
$env:PYTHONPATH = $root

Write-Host "Launching orchestrator with $ConfigPath"
python ".\orchestrator\orchestrate.py" --config $ConfigPath
