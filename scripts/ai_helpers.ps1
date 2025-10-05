# -*- coding: utf-8 -*-
# Helpers for AI loop / MLflow (PowerShell 5 compatible)

function Invoke-PyJson {
    param([string]$PyCode)
    # Write temp script, run it, capture stdout, delete temp
    $tmp = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), "mlflow_query_$([guid]::NewGuid().ToString()).py")
    [System.IO.File]::WriteAllText($tmp, $PyCode, [System.Text.UTF8Encoding]::new($false))
    try {
        $out = & python $tmp 2>$null
    } catch {
        Write-Error "Python execution failed. Ensure 'python' is on PATH and mlflow is installed in this env."
        return $null
    } finally {
        Remove-Item -Path $tmp -Force -ErrorAction SilentlyContinue
    }
    return $out -join "`n"
}

function Get-FailedRuns {
    [CmdletBinding()]
    param(
        [int]$LookbackHours = 24,
        [int]$Limit = 50
    )
    $py = @"
import json, time
try:
    import mlflow
    import pandas as pd
except Exception as e:
    print(json.dumps({"error": "mlflow_import_failed", "detail": str(e)}))
    raise SystemExit(0)

lookback_s = $LookbackHours * 3600
now_ms = int(time.time() * 1000)
try:
    df = mlflow.search_runs(max_results=1000)
except Exception as e:
    print(json.dumps({"error": "mlflow_search_failed", "detail": str(e)}))
    raise SystemExit(0)

if df.empty:
    print("[]"); raise SystemExit(0)

# Filter not finished
if "status" in df.columns:
    df = df[df["status"] != "FINISHED"]

# Convert times and filter by lookback
for col in ("start_time","end_time"):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
if "end_time" in df.columns:
    df = df[df["end_time"].fillna(now_ms) >= now_ms - lookback_s*1000]

cols = [c for c in ["run_id","status","experiment_id","start_time","end_time"] if c in df.columns]
df = df[cols].head($Limit)
print(json.dumps(df.to_dict(orient="records")))
"@

    $json = Invoke-PyJson -PyCode $py
    if (-not $json) { return @() }
    try { return ($json | ConvertFrom-Json) } catch { return @() }
}

function Apply-FailedRun {
    [CmdletBinding()]
    param([Parameter(Mandatory=$true)][string]$RunId)
    $root = 'C:\Users\Gladiator\trading-lab'
    Push-Location $root
    try {
        Write-Host "Applying patches to run $RunId ..." -ForegroundColor Cyan
        & python -m companion.ai_loop.cli apply-patches "$RunId"
    } finally {
        Pop-Location
    }
}

function Apply-LatestFailedRun {
    [CmdletBinding()]
    param([int]$LookbackHours = 24)
    $items = Get-FailedRuns -LookbackHours $LookbackHours -Limit 1
    if (-not $items -or $items.Count -eq 0) {
        Write-Warning "No failed/unfinished runs found in last $LookbackHours hour(s)."
        return
    }
    $rid = $items[0].run_id
    if (-not $rid) {
        Write-Warning "No run_id available on the latest item."
        return
    }
    Apply-FailedRun -RunId $rid
}

Export-ModuleMember -Function Get-FailedRuns,Apply-FailedRun,Apply-LatestFailedRun