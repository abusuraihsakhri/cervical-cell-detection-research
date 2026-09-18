@echo off
cd /d "%~dp0"
python diagnose_roboflow.py
echo.
echo ============================================
echo Script finished. Log saved under logs\ folder.
echo Press any key to close this window...
echo ============================================
pause >nul
