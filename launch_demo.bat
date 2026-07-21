@echo off
cd /d "%~dp0"
echo ============================================================
echo   Visual Traffic System -- Web Demo Launcher
echo ============================================================
echo.
echo   Starting web server at http://localhost:5000
echo   Press Ctrl+C to stop.
echo.
echo   If the browser doesn't open automatically:
echo     Open http://localhost:5000 manually
echo ============================================================
echo.

REM Try Anaconda Python first, fall back to system Python
if exist "E:\Anaconda\python.exe" (
    set PYTHON=E:\Anaconda\python.exe
) else (
    set PYTHON=python
)

set PYTHONIOENCODING=utf-8
%PYTHON% frontend\server.py
pause
