@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

cd /d "%~dp0"
title KWS Polygon - Wake-Word Training

echo ======================================================================
echo   KWS Training Polygon (Wake-Word Training Web UI)
echo ======================================================================
echo.

set "PYTHON_EXE="

if exist "%~dp0..\dist_app\runtime\python.exe" (
    set "PYTHON_EXE=%~dp0..\dist_app\runtime\python.exe"
) else if exist "%~dp0..\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe"
) else if exist "%~dp0..\runtime\python.exe" (
    set "PYTHON_EXE=%~dp0..\runtime\python.exe"
) else (
    where python >nul 2>nul
    if !errorlevel! equ 0 (
        set "PYTHON_EXE=python"
    )
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] Python runtime not found!
    echo Please make sure .venv or dist_app\runtime exists.
    pause
    exit /b 1
)

set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

echo [INFO] Python found: "%PYTHON_EXE%"
echo [INFO] Starting web server...
echo.

"%PYTHON_EXE%" app.py

if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with error code !errorlevel!
    pause
)
