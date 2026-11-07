$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$env:PYTHONPATH = $Root

if (-not (Test-Path "models/v3/model_bundle_v3.joblib")) {
    Write-Host "[V8] V3 base model missing. Running V3 first ..."
    powershell -ExecutionPolicy Bypass -File "scripts/run_v3.ps1"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

python scripts/train_v8.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/generate_backtest_v8.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "`n[V8] Spread-expert pipeline complete."
