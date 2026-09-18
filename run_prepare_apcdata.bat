@echo off
cd /d "%~dp0"
python prepare_apcdata.py --raw-dir "data\apcdata\APCData cervical cytology cells" --val-fraction 0.15
echo.
echo ============================================
echo Script finished. Log saved under logs\ folder.
echo Press any key to close this window...
echo ============================================
pause >nul
