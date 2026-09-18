@echo off
setlocal
cd /d "%~dp0"

:: Set local file serving for Label Studio to access data/dense_eval
set "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true"
set "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=%~dp0data\dense_eval"

echo =========================================================================
echo Starting Label Studio for SIPaKMeD Dense 40 Verification
echo Local Files Root: %LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT%
echo =========================================================================
echo.
echo Instructions:
echo 1. Open your browser at http://localhost:8080 (or as shown below).
echo 2. Create or open project: 'SIPaKMeD Dense 40'.
echo 3. In Project Settings -^> Labeling Interface -^> Code:
echo    Paste contents of data\dense_eval\sipakmed_dense40\labelstudio_config.xml
echo 4. In Project -^> Import:
echo    Upload data\dense_eval\sipakmed_dense40\labelstudio_tasks.json
echo 5. Review the 40 fields (green = Normal, orange = Abnormal):
echo    - Delete false candidate boxes
echo    - Add boxes on unannotated visible cells
echo    - Click Submit for each image
echo =========================================================================
echo.

".venv-labelstudio\Scripts\label-studio.exe" start --port 8080
