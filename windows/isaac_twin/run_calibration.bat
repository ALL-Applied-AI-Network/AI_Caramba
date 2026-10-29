@echo off
echo Melting Rainbow - EXPOSURE CALIBRATION: ROUND 2: 10 clean recipes x 12 exposure times (about 3-5 minutes).
echo Camera exposure only, in memory. Organizers' files and liquid opacity are not changed.
cd /d "%USERPROFILE%\Downloads\Copy of csi_digital_twin\csi_digital_twin_v1" || goto :err
if exist calibration_round1 rmdir /s /q calibration_round1
if exist calibration ren calibration calibration_round1
"C:\isaac\env\Scripts\python.exe" calibrate_exposure.py > calibration_console.txt 2>&1
echo Exit code: %ERRORLEVEL% >> calibration_console.txt
echo.
echo FINISHED. Summary:
powershell -NoProfile -Command "if (Test-Path calibration\calibration_log.txt) { Get-Content calibration\calibration_log.txt -Tail 5 } else { Get-Content calibration_console.txt -Tail 15 }"
echo.
echo Tell Claude "calibration done".
pause
exit /b 0
:err
echo Could not find the twin folder. Tell Claude.
pause
exit /b 1
