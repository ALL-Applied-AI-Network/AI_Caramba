@echo off
echo Melting Rainbow - PAPER SHEET TEST: same 46 vials as before, with a white sheet behind the vial like the real line (about 2-3 min).
echo In memory only. Organizers' files are not changed.
cd /d "%USERPROFILE%\Downloads\Copy of csi_digital_twin\csi_digital_twin_v1" || goto :err
if exist twin_trials_paper rmdir /s /q twin_trials_paper
"C:\isaac\env\Scripts\python.exe" render_trials.py --ev -8 --paper --out_dir twin_trials_paper > twin_trials_paper_console.txt 2>&1
echo Exit code: %ERRORLEVEL% >> twin_trials_paper_console.txt
echo.
echo FINISHED. Summary:
powershell -NoProfile -Command "if (Test-Path twin_trials_paper\render_log.txt) { Get-Content twin_trials_paper\render_log.txt -Tail 3 } else { Get-Content twin_trials_paper_console.txt -Tail 15 }"
echo.
echo Tell Claude "paper test done".
pause
exit /b 0
:err
echo Could not find the twin folder. Tell Claude.
pause
exit /b 1
