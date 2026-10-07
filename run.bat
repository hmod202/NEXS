@echo off
rem Runs NEXS on Windows and restarts it automatically if it stops.
cd /d "%~dp0"
:loop
python -m nexs.app
echo NEXS stopped, restarting in 5 seconds...
timeout /t 5 /nobreak >nul
goto loop
