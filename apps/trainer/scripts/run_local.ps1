# Raccourci Windows : active le venv du depot et lance le runtime local.
#
#   .\apps\trainer\scripts\run_local.ps1 run build --positions 10000 --seed 456
#   .\apps\trainer\scripts\run_local.ps1 run train --version 0.3.0 --dataset datasets/dataset_v006
#   .\apps\trainer\scripts\run_local.ps1 run tournament --a champion --b minimax:8
#   .\apps\trainer\scripts\run_local.ps1 config
#
# Pre-requis (une fois) :
#   py -3.12 -m venv .venv
#   .\.venv\Scripts\pip install -e ".[dev,perf,train,export]"
#   copy songo.toml.example songo.toml   # puis ajuster data_root / num_workers

$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "venv introuvable : $python -- creer avec py -3.12 -m venv .venv" }

$env:SONGO_PROVIDER = "local"
Push-Location $repo
try {
    & $python -m songo_ai.cloud @args
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
