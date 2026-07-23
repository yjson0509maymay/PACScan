param(
    [string]$WorkspaceRoot = (Split-Path -Parent $PSScriptRoot | Split-Path -Parent)
)

$ErrorActionPreference = "Stop"
$pacscanPath = Join-Path $WorkspaceRoot "PACScan"
$braintensorPath = Join-Path $WorkspaceRoot "BRAINTENSOR"

if (-not (Test-Path -LiteralPath $pacscanPath)) {
    git clone https://github.com/yjson0509maymay/PACScan.git $pacscanPath
}
if (-not (Test-Path -LiteralPath $braintensorPath)) {
    git clone https://github.com/yjson0509maymay/BRAINTENSOR.git $braintensorPath
}

$pipelineScript = Join-Path $braintensorPath "01_Preprocessing\스크립트\preparing_ref21order_v1.py"
if (-not (Test-Path -LiteralPath $pipelineScript)) {
    throw "BRAINTENSOR 전처리 스크립트를 찾을 수 없습니다: $pipelineScript"
}

Push-Location $pacscanPath
try {
    python -m pip install -r requirements-local.txt
    Write-Host ""
    Write-Host "PACScan과 BRAINTENSOR 연결 준비 완료"
    Write-Host "PACScan:     $pacscanPath"
    Write-Host "BRAINTENSOR: $braintensorPath"
    Write-Host "실행: streamlit run app.py"
}
finally {
    Pop-Location
}
