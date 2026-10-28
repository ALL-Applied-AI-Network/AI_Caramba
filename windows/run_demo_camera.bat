@echo off
REM Melting Rainbow - live inspection from USB camera 0, NO recipe list: every 3 vials is one batch
REM (the 3rd vial of each batch is the clean reference). Press q in the video window to stop.
REM To use a recipe list instead, replace  --batch-size 3  with  --plan "%CODE%\demo\example_plan.csv"
set PY=C:\isaac\env\Scripts\python.exe
if not exist "%PY%" set PY=python
set CODE=%~dp0..\melting-rainbow
set OUT=%~dp0..\results\demo_camera
"%PY%" "%CODE%\demo\live_inspect.py" --camera 0 --batch-size 3 --out "%OUT%" --show
echo.
echo Done. Results are in %OUT%
pause
