@echo off
rem Runs NEXS on Windows and restarts it automatically if it stops.
cd /d "%~dp0"
rem Prefer the project virtualenv; fall back to the system Python.
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
:loop
"%PY%" -m nexs.app
echo NEXS stopped, restarting in 5 seconds...
timeout /t 5 /nobreak >nul
goto loop
