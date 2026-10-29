@echo off
setlocal
echo ============================================================
echo  Melting Rainbow: Isaac Sim 6.0 setup + one test render
echo  Big download (many GB). Can take 30-60 minutes. Leave it open.
echo ============================================================
echo.

if not exist C:\isaac mkdir C:\isaac
cd /d C:\isaac

echo [1/6] Creating Python 3.12 environment in C:\isaac\env ...
if not exist env\Scripts\python.exe (
    py -3.12 -m venv env || goto :err
)
call env\Scripts\activate.bat || goto :err

echo [2/6] Updating pip ...
python -m pip install --upgrade pip || goto :err

echo [3/6] Installing PyTorch (CUDA 12) ...
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 || goto :err

echo [4/6] Installing Isaac Sim 6.0.1 from pypi.nvidia.com ...
pip install "isaacsim[all,extscache]==6.0.1.0" --extra-index-url https://pypi.nvidia.com || goto :err

echo [5/6] Installing OpenCV and NumPy ...
pip install opencv-python numpy || goto :err

echo [6/6] Test render of recipe 7 from the digital twin ...
echo      Isaac Sim will ask you to accept NVIDIA's license the first time. Type Yes to continue.
echo      The first render can sit for several minutes while the GPU warms up.
cd /d "%USERPROFILE%\Downloads\Copy of csi_digital_twin\csi_digital_twin_v1" || goto :err
if not exist renders mkdir renders
python tools\render_example.py --camera FillCam --stage variants/R7.usda --frame 657 --out renders/R7_test.png || goto :err

echo.
echo DONE. Opening renders\R7_test.png - compare it with previews\fill_station_rtx.png
start "" "renders\R7_test.png"
pause
exit /b 0

:err
echo.
echo ************************************************************
echo  FAILED at the step above. Copy the last ~20 lines of text
echo  in this window and paste them to Claude.
echo ************************************************************
pause
exit /b 1
