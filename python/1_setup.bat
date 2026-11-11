@echo off
cd /d "%~dp0"
echo Setting up Waste Disposal Coach... (log: setup_log.txt)
(
echo === python ===
where python
python --version
echo === pip install ===
python -m pip install -r requirements.txt
echo === tests ===
python test_coach.py
) > setup_log.txt 2>&1
type setup_log.txt
echo.
echo SETUP FINISHED
pause
