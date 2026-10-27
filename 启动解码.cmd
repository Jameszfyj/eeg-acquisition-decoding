@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -B -X utf8 -m handbci.continuous_online
) else (
  python -B -X utf8 -m handbci.continuous_online
)
if errorlevel 1 pause
