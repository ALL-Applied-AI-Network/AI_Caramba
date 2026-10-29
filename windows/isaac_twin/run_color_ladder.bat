@echo off
echo Melting Rainbow - CALIBRATION ROUND 3: color ladder at exposure 1/256 s (36 images, about 2-3 minutes).
echo In memory only. Organizers' files and liquid opacity are not changed.
cd /d "%USERPROFILE%\Downloads\Copy of csi_digital_twin\csi_digital_twin_v1" || goto :err
if exist calibration_ladder rmdir /s /q calibration_ladder
"C:\isaac\env\Scripts\python.exe" calibrate_exposure.py --ladder --ev -8 --out_dir calibration_ladder > calibration_ladder_console.txt 2>&1
echo Exit code: %ERRORLEVEL% >> calibration_ladder_console.txt
echo.
echo FINISHED. Summary:
powershell -NoProfile -Command "if (Test-Path calibration_ladder\calibration_log.txt) { Get-Content calibration_ladder\calibration_log.txt -Tail 4 } else { Get-Content calibration_ladder_console.txt -Tail 15 }"
echo.
echo Tell Claude "ladder done".
pause
exit /b 0
:err
echo Could not find the twin folder. Tell Claude.
pause
exit /b 1
