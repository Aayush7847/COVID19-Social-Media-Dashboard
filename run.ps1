# PowerShell launcher. If scripts are blocked, run once:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\run.ps1
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$Host.UI.RawUI.WindowTitle = "COVID-19 Social Media Intelligence Dashboard"

function Fail($msg) {
    Write-Host ""
    Write-Host "[ERROR] $msg" -ForegroundColor Red
    Write-Host "See 'Troubleshooting' in README.md" -ForegroundColor Yellow
    Read-Host "Press Enter to close"
    exit 1
}

# 1. Find Python
$py = $null
foreach ($cand in @(@("py", "-3.11"), @("py", "-3"), @("python"))) {
    try {
        & $cand[0] $cand[1..($cand.Length - 1)] -c "import sys" 2>$null
        if ($LASTEXITCODE -eq 0) { $py = $cand; break }
    } catch { }
}
if (-not $py) { Fail "Python was not found. Install Python 3.11 from https://www.python.org/downloads/ (tick 'Add python.exe to PATH')." }
function RunPy { & $py[0] $py[1..($py.Length - 1)] @args }
Write-Host "[1/4] Python found:"; RunPy --version

# 2. Virtual environment
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Host "[2/4] Creating virtual environment (first run only)..."
    RunPy -m venv .venv
    if ($LASTEXITCODE -ne 0) { Fail "Could not create the virtual environment." }
} else { Write-Host "[2/4] Virtual environment already exists." }

# 3. Dependencies
$installed = Join-Path $PSScriptRoot ".venv\requirements.installed.txt"
$needInstall = (-not (Test-Path $installed)) -or ((Get-FileHash requirements.txt).Hash -ne (Get-FileHash $installed).Hash)
if ($needInstall) {
    Write-Host "[3/4] Installing dependencies (5-15 minutes the first time, PyTorch is large)..."
    & $venvPy -m pip install --upgrade pip
    & $venvPy -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { Fail "Installing the requirements failed." }
    Copy-Item requirements.txt $installed -Force
} else { Write-Host "[3/4] Dependencies already installed." }

# Pre-answer Streamlit's first-run e-mail prompt
$cred = Join-Path $env:USERPROFILE ".streamlit\credentials.toml"
if (-not (Test-Path $cred)) {
    New-Item -ItemType Directory -Force -Path (Split-Path $cred) | Out-Null
    Set-Content -Path $cred -Value "[general]`nemail = `"`""
}

# 4. Start
Write-Host "[4/4] Starting the dashboard - the browser opens automatically (or go to http://localhost:8501)."
Write-Host "      The first analysis downloads the AI models (internet needed). Ctrl+C stops the app."
& $venvPy -m streamlit run app.py --browser.gatherUsageStats false
if ($LASTEXITCODE -ne 0) { Fail "The app stopped with an error." }
