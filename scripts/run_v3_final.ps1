$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not (Test-Path "models/v3/model_bundle_v3_dev.joblib")) {
    Write-Error "Run scripts/run_v3.ps1 first so calibration is learned on held-out validation."
    exit 1
}
python scripts/prepare_v3.py --final
$cache = "reports/v3/train_cache_final"
$kinds = @("q10","q50","q90","cat50","direction","spike_high","spike_low")
foreach ($k in $kinds) {
    Write-Host "`n[V3 FINAL] Training $k"
    python scripts/model_worker_v3.py --data-dir "$cache/matrix" --features "$cache/features.json" --kind $k --output "$cache/$k.joblib"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
python scripts/finalize_v3.py --final
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/generate_backtest_v3.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "`n[V3 FINAL] Final refit pipeline complete."
