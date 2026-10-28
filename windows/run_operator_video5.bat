@echo off
REM Melting Rainbow - operator station playing the organizers video 5 at 8x (practice / filming the UI).
set PY=C:\isaac\env\Scripts\python.exe
if not exist "%PY%" set PY=python
set CODE=%~dp0..\melting-rainbow
set DATA=%USERPROFILE%\Downloads\csi-vial-fill-dataset\csi-vial-fill-dataset
set AI=
if exist "%CODE%\operator_ui\pair_model.pt" set AI=--ai-model "%CODE%\operator_ui\pair_model.pt"
"%PY%" "%CODE%\operator_ui\operator_app.py" --video "%DATA%\vial_fill\videos\vial_fill_video_5.mp4" --autostart --speed 8 --batch-size 3 %AI% --out "%~dp0..\results\operator_output_video5"
pause
