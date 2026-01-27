<#
.SYNOPSIS
  Create and bootstrap a Python virtual environment (.venv) on Windows (PowerShell).

USAGE
  In PowerShell (run as normal user):
    .\scripts\setup_venv.ps1           # creates .venv and installs core requirements
    .\scripts\setup_venv.ps1 -Ensemble  # also installs ensemble extras (lightgbm/pycaret)

#>
param(
    [switch]$Ensemble
)

Set-StrictMode -Version Latest

$venvPath = Join-Path -Path (Get-Location) -ChildPath ".venv"
if (-not (Test-Path $venvPath)) {
    python -m venv $venvPath
} else {
    Write-Host ".venv already exists"
}

$activate = Join-Path $venvPath 'Scripts\Activate.ps1'
if (Test-Path $activate) {
    Write-Host "To activate the virtualenv run:`n  & $activate"
} else {
    Write-Host "Activation script not found at $activate"
}

Write-Host "Upgrading pip and installing requirements..."
& $venvPath\Scripts\python -m pip install --upgrade pip setuptools wheel
& $venvPath\Scripts\python -m pip install -r requirements.txt
if ($Ensemble) {
    Write-Host "Installing ensemble extras (may require build tools)..."
    & $venvPath\Scripts\python -m pip install -r requirements-ensemble.txt
}

Write-Host "Done. Activate with: & $activate"
