# Digital-twin launchers (Windows + NVIDIA Isaac Sim 6.0)

Used to build the 2,000-image training set from the organizers' digital twin (`csi_digital_twin_v1`), which must be
unzipped at `%USERPROFILE%\Downloads\Copy of csi_digital_twin\csi_digital_twin_v1`. The scripts they run are in
`melting-rainbow/lane4_twin/` (copy them into the twin folder first).

| Launcher | What it does |
|---|---|
| `setup_isaac.bat` | one-time: Python 3.12 env in `C:\isaac\env`, PyTorch (CUDA), Isaac Sim 6.0.1, test render |
| `run_calibration.bat` | exposure sweep to match the twin camera to the real camera |
| `run_color_ladder.bat`, `run_opacity_ladder.bat`, `run_paper_test.bat` | calibration checks (liquid color, opacity, white paper) |
| `run_trials_calibrated.bat` | renders labeled carryover trials |
| `run_dataset_quick.bat` / `run_dataset_full.bat` | the training set (quick test / full 2,000 images, ~35–40 min) |
