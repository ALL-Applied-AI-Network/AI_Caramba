@echo off
REM Melting Rainbow - operator station on ANY recorded video.
REM Easiest: drag a video file (.mp4) onto this .bat file. Or double-click it and paste the video's full path.
REM The camera must be framed like the organizers' vial_fill camera (same holder position) for the numbers to mean anything.
set PY=C:\isaac\env\Scripts\python.exe
if not exist "%PY%" set PY=python
set CODE=%~dp0..\melting-rainbow
set VID=%~1
if "%VID%"=="" set /p VID=Paste the full path of the video file and press Enter: 
set VID=%VID:"=%
set AI=
if exist "%CODE%\operator_ui\pair_model.pt" set AI=--ai-model "%CODE%\operator_ui\pair_model.pt"
"%PY%" "%CODE%\operator_ui\operator_app.py" --video "%VID%" --autostart --speed 8 --batch-size 3 %AI% --out "%~dp0..\results\operator_output_any_video"
pause
