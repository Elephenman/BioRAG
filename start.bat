@echo off
echo Starting BioRAG Engine...
cd /d "%~dp0"
python -m biorag.engine
pause