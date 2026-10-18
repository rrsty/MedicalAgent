@echo off
title SmartMedicalConsultation - Quick Start
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo ============================================
echo   SmartMedicalConsultation - Quick Start
echo   (Chinese output follows in Python step)
echo ============================================
echo.

rem ---- 1. Virtual env (create + install deps if missing) ----
if not exist ".venv\Scripts\python.exe" (
    echo [SETUP] venv not found. Creating and installing dependencies, first run takes 5-10 minutes ...
    python -m venv .venv
    if errorlevel 1 goto :fail
    ".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
    ".venv\Scripts\python.exe" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
    ".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt
    if errorlevel 1 goto :fail
    echo [SETUP] dependencies installed.
    echo.
)

rem ---- 2. Orchestrated start: docker - data - frontend - server - browser ----
".venv\Scripts\python.exe" scripts\quick_start.py %*
goto :eof

:fail
echo.
echo [ERROR] setup failed. Check your network and retry, see log above.
pause
