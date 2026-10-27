@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title COVID-19 Social Media Intelligence Dashboard

echo ============================================================
echo   COVID-19 Social Media Intelligence Dashboard - launcher
echo ============================================================
echo.

REM ---------- 1. Find Python (3.9 - 3.12 recommended: 3.11) ----------
set "PY="
where py >nul 2>nul
if not errorlevel 1 (
    py -3.11 -c "import sys" >nul 2>nul
    if not errorlevel 1 set "PY=py -3.11"
)
if not defined PY (
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 -c "import sys" >nul 2>nul
        if not errorlevel 1 set "PY=py -3"
    )
)
if not defined PY (
    where python >nul 2>nul
    if not errorlevel 1 set "PY=python"
)
if not defined PY goto :nopython

%PY% -c "import sys; sys.exit(0 if (3,9) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>nul
if errorlevel 1 (
    echo [WARNING] Your Python version may not be supported by PyTorch/EasyOCR yet.
    echo           Python 3.11 is recommended: https://www.python.org/downloads/release/python-3119/
    echo.
)
echo [1/4] Python found:
%PY% --version

REM ---------- 2. Virtual environment ----------
if not exist ".venv\Scripts\python.exe" (
    echo [2/4] Creating virtual environment ^(first run only^)...
    %PY% -m venv .venv
    if errorlevel 1 goto :fail
) else (
    echo [2/4] Virtual environment already exists.
)

REM ---------- 3. Dependencies (only when requirements.txt changed) ----------
fc /b requirements.txt ".venv\requirements.installed.txt" >nul 2>nul
if errorlevel 1 (
    echo [3/4] Installing dependencies. This can take 5-15 minutes the first time
    echo       ^(PyTorch is large^). Please keep this window open...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto :fail
    copy /y requirements.txt ".venv\requirements.installed.txt" >nul
) else (
    echo [3/4] Dependencies already installed.
)

REM Streamlit asks for an e-mail on first start; pre-answer it so nothing blocks.
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    mkdir "%USERPROFILE%\.streamlit" >nul 2>nul
    ( echo [general]& echo email = "" ) > "%USERPROFILE%\.streamlit\credentials.toml"
)

REM ---------- 4. Start ----------
echo [4/4] Starting the dashboard. Your browser will open automatically.
echo       If it does not, open http://localhost:8501
echo       NOTE: the first analysis downloads the AI models (internet needed, a few hundred MB).
echo       Press Ctrl+C in this window to stop the app.
echo.
".venv\Scripts\python.exe" -m streamlit run app.py --browser.gatherUsageStats false
if errorlevel 1 goto :fail
goto :end

:nopython
echo [ERROR] Python was not found.
echo         Install Python 3.11 from https://www.python.org/downloads/
echo         and tick "Add python.exe to PATH" in the installer, then run this file again.
goto :fail

:fail
echo.
echo [ERROR] Setup or start-up failed. Read the messages above. Common fixes:
echo         - Check your internet connection ^(first run downloads packages^)
echo         - Delete the ".venv" folder and run run.bat again
echo         - See "Troubleshooting" in README.md
pause
exit /b 1

:end
endlocal
