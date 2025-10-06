param(
  [string]$Root = ".",
  [string]$Host = "127.0.0.1",
  [int]$Port = 8050
)
# PowerShell 5 launcher for Dash
$env:PYTHONIOENCODING="utf-8"
python - << "PY"
from companion.explorer.dashboard import run_app
run_app(root=r"$($Root)", host="$($Host)", port=$($Port))
PY
