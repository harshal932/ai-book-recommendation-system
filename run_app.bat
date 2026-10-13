@echo off
title AI Book Recommendation System Launcher
echo ============================================================
echo   Starting AI-Based Intelligent Book Recommendation System
echo ============================================================
echo.
cd /d "%~dp0"
echo Checking Python environment...
python --version
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b 1
)

echo Starting Streamlit server...
echo Local Access:   http://localhost:8501
echo Network Access: Check console output below for LAN IP.
echo.
python -m streamlit run app.py
pause
