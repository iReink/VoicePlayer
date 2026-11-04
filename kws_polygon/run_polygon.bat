@echo off
chcp 65001 > nul
setlocal

cd /d "%~dp0"
title KWS Polygon - Испытательный стенд Wake-Word

echo ======================================================================
echo   Запуск испытательного полигона KWS (Wake-Word Polygon)
echo ======================================================================
echo.

set "PYTHON_EXE=%~dp0..\dist_app\runtime\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [X] Ошибка: Python не найден по пути "%PYTHON_EXE%"!
    pause
    exit /b 1
)

set PYTHONIOENCODING=utf-8
"%PYTHON_EXE%" app.py

pause
