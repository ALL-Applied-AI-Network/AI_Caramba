@echo off
echo Melting Rainbow - OPACITY TEST (only run this if Cruz approved it): color ladder at liquid opacity 0.8, 0.9, 1.0
echo 108 images, about 3-4 minutes. In memory only - the organizers' files keep opacity 0.6.
cd /d "%USERPROFILE%\Downloads\Copy of csi_digital_twin\csi_digital_twin_v1" || goto :err
if exist calibration_opacity rmdir /s /q calibration_opacity
"C:\isaac\env\Scripts\python.exe" calibrate_exposure.py --ladder --ev -8 --opacity 0.8 0.9 1.0 --out_dir calibration_opacity > calibration_opacity_console.txt 2>&1
echo Exit code: %ERRORLEVEL% >> calibration_opacity_console.txt
echo.
echo FINISHED. Summary:
powershell -NoProfile -Command "if (Test-Path calibration_opacity\calibration_log.txt) { Get-Content calibration_opacity\calibration_log.txt -Tail 4 } else { Get-Content calibration_opacity_console.txt -Tail 15 }"
echo.
echo Tell Claude "opacity done".
pause
exit /b 0
:err
echo Could not find the twin folder. Tell Claude.
pause
exit /b 1
