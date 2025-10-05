# refresh_master.ps1 — run nightly to update leaderboard & master list
Set-Location "$PSScriptRoot\.."
python -m companion.explorer.refresh_master
