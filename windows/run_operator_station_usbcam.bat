@echo off
REM Melting Rainbow - operator station on a USB webcam (camera 1; the laptop's built-in camera is usually 0).
REM If the picture is the wrong camera: type 0 or 2 in the page's Source box and press Start again.
REM Saves the raw camera video + alignment + logs in results\operator_output. Close this window to quit.
set PY=C:\isaac\env\Scripts\python.exe
if not exist "%PY%" set PY=python
set CODE=%~dp0..\melting-rainbow
set AI=
if exist "%CODE%\operator_ui\pair_model.pt" set AI=--ai-model "%CODE%\operator_ui\pair_model.pt"
"%PY%" "%CODE%\operator_ui\operator_app.py" --camera 1 --batch-size 3 %AI% --out "%~dp0..\results\operator_output"
pause
