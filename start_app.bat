@echo off
REM Skin Disease Classification - Windows Launcher
REM This script starts the Streamlit web application

echo.
echo ========================================
echo  Skin Disease Classification Web App
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "skin_env" (
    echo Creating virtual environment...
    python -m venv skin_env
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        echo TIP: Try running: python fix_pytorch.py
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call skin_env\Scripts\activate.bat

REM Install requirements if needed
if not exist "skin_env\Lib\site-packages\streamlit" (
    echo Installing requirements...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install requirements
        echo TIP: Try running: python fix_pytorch.py first
        pause
        exit /b 1
    )
)

REM Start Streamlit app
echo Starting Streamlit web application...
echo.
echo The app will open in your default web browser
echo URL: http://localhost:8501
echo.
echo WARNING: Keep this window open while using the app
echo Press Ctrl+C to stop the server
echo.

streamlit run app.py

echo.
echo 👋 Thanks for using Skin Disease Classification!
pause