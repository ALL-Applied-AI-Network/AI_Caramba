@echo off
echo Melting Rainbow - CALIBRATED trial set: 46 images at exposure 1/256 s, calibrated orange (about 2-3 minutes).
echo Labels: off-color only when the change vs the clean vial is visible (dE 5 or more).
echo In memory only. Organizers' files and liquid opacity are not changed.
cd /d "%USERPROFILE%\Downloads\Copy of csi_digital_twin\csi_digital_twin_v1" || goto :err
if exist twin_trials_calibrated rmdir /s /q twin_trials_calibrated
"C:\isaac\env\Scripts\python.exe" render_trials.py --ev -8 --tolerance 5 --out_dir twin_trials_calibrated > twin_trials_calibrated_console.txt 2>&1
echo Exit code: %ERRORLEVEL% >> twin_trials_calibrated_console.txt
echo.
echo FINISHED. Summary:
powershell -NoProfile -Command "if (Test-Path twin_trials_calibrated\render_log.txt) { Get-Content twin_trials_calibrated\render_log.txt -Tail 5 } else { Get-Content twin_trials_calibrated_console.txt -Tail 15 }"
echo.
echo Tell Claude "calibrated set done".
pause
exit /b 0
:err
echo Could not find the twin folder. Tell Claude.
pause
exit /b 1
