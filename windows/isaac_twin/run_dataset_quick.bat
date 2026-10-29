@echo off
echo Melting Rainbow - DATASET v2 (quick): 600 training images (400 unique leftover renders + 10 clean), about 10-12 minutes.
echo Calibrated colors, exposure 1/256 s, opacity 0.6 (organizers' value). In memory only.
echo You can keep using the laptop lightly; keep it plugged in. Do not close this window.
cd /d "%USERPROFILE%\Downloads\Copy of csi_digital_twin\csi_digital_twin_v1" || goto :err
if exist twin_dataset_v2_quick rmdir /s /q twin_dataset_v2_quick
"C:\isaac\env\Scripts\python.exe" render_dataset.py --renders 400 --clean-copies 20 --out_dir twin_dataset_v2_quick > twin_dataset_v2_quick_console.txt 2>&1
echo Exit code: %ERRORLEVEL% >> twin_dataset_v2_quick_console.txt
echo.
echo FINISHED. Summary:
powershell -NoProfile -Command "if (Test-Path twin_dataset_v2_quick\render_log.txt) { Get-Content twin_dataset_v2_quick\render_log.txt -Tail 4 } else { Get-Content twin_dataset_v2_quick_console.txt -Tail 15 }"
echo.
echo Tell Claude "dataset quick done".
pause
exit /b 0
:err
echo Could not find the twin folder. Tell Claude.
pause
exit /b 1
