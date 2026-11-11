@echo off
cd /d "%~dp0"
echo Starting the coach. Keys: q quit, r region, c contaminated, s snapshot
python waste_coach.py --region india 2> run_log.txt
type run_log.txt
pause
