@echo off
chcp 65001 > nul
title Yandex Music Assistant
cd /d "%~dp0"
"%~dp0runtime\python.exe" "%~dp0main.py"
if errorlevel 1 (
    echo.
    echo Application exited with error.
    pause
)
