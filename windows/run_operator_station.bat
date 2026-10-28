@echo off
REM Melting Rainbow - operator station on USB camera 0. A browser page opens at http://localhost:8765 - press Start.
REM AI second opinion is on if operator_ui\pair_model.pt exists. Close this window to quit.
set PY=C:\isaac\env\Scripts\python.exe
if not exist "%PY%" set PY=python
set CODE=%~dp0..\melting-rainbow
set AI=
if exist "%CODE%\operator_ui\pair_model.pt" set AI=--ai-model "%CODE%\operator_ui\pair_model.pt"
"%PY%" "%CODE%\operator_ui\operator_app.py" --camera 0 --batch-size 3 %AI% --out "%~dp0..\results\operator_output"
pause
