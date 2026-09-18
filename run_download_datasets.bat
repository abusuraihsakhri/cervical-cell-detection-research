@echo off
cd /d "%~dp0"
python download_datasets.py
echo.
echo ============================================
echo Script finished. Log saved under logs\ folder.
echo Press any key to close this window...
echo ============================================
pause >nul
