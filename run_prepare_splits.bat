@echo off
cd /d "%~dp0"
python prepare_splits.py --dataset-dir data\sipakmed\mirror_a --val-fraction 0.15
echo.
echo ============================================
echo Script finished. Log saved under logs\ folder.
echo Press any key to close this window...
echo ============================================
pause >nul
