@echo off
REM Melting Rainbow - command-line check of organizers' video 5 (no browser). Writes inspection.mp4 + logs.
set PY=C:\isaac\env\Scripts\python.exe
if not exist "%PY%" set PY=python
set CODE=%~dp0..\melting-rainbow
set DATA=%USERPROFILE%\Downloads\csi-vial-fill-dataset\csi-vial-fill-dataset
"%PY%" "%CODE%\demo\live_inspect.py" --video "%DATA%\vial_fill\videos\vial_fill_video_5.mp4" --batch-size 3 --out "%~dp0..\results\demo_video5"
echo.
echo Done. Results are in %~dp0..\results\demo_video5
pause
