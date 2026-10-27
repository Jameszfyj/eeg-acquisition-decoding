@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -B -X utf8 -m handbci.studio.server
) else (
  python -B -X utf8 -m handbci.studio.server
)
if errorlevel 1 pause
