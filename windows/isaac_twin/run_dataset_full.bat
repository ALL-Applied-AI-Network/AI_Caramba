@echo off
echo Melting Rainbow - DATASET v2 (full): 2000 training images (1400 unique leftover renders + 10 clean), about 35-40 minutes.
echo Calibrated colors, exposure 1/256 s, opacity 0.6 (organizers' value). In memory only.
echo You can keep using the laptop lightly; keep it plugged in. Do not close this window.
cd /d "%USERPROFILE%\Downloads\Copy of csi_digital_twin\csi_digital_twin_v1" || goto :err
if exist twin_dataset_v2 rmdir /s /q twin_dataset_v2
"C:\isaac\env\Scripts\python.exe" render_dataset.py --renders 1400 --clean-copies 60 --out_dir twin_dataset_v2 > twin_dataset_v2_console.txt 2>&1
echo Exit code: %ERRORLEVEL% >> twin_dataset_v2_console.txt
echo.
echo FINISHED. Summary:
powershell -NoProfile -Command "if (Test-Path twin_dataset_v2\render_log.txt) { Get-Content twin_dataset_v2\render_log.txt -Tail 4 } else { Get-Content twin_dataset_v2_console.txt -Tail 15 }"
echo.
echo Tell Claude "dataset full done".
pause
exit /b 0
:err
echo Could not find the twin folder. Tell Claude.
pause
exit /b 1
