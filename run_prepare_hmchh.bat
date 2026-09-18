@echo off
cd /d "%~dp0"
python prepare_hmchh.py --images-dir "D:\pap_model\JPEGImages_extracted\JPEGImages" --annotations-dir "D:\pap_model\Annotations_extracted\anno_copy" --out-dir "D:\pap_model\HMCHH_YOLO_prepared" --val-fraction 0.15
echo.
echo ============================================
echo Script finished. Log saved under logs\ folder.
echo Press any key to close this window...
echo ============================================
pause >nul
