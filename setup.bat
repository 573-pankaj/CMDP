@echo off
REM Setup script for PDNAC-NC Algorithm (Windows)
REM Automatically creates virtual environment and installs dependencies

echo ==========================================
echo PDNAC-NC Algorithm Setup Script
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo X Error: Python is not installed
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

REM Display Python version
echo Checking Python version...
python --version
echo.

REM Create virtual environment
echo Creating virtual environment...
if exist venv (
    echo ! Virtual environment already exists
    set /p REPLY="Do you want to remove it and create a new one? (y/n): "
    if /i "%REPLY%"=="y" (
        rmdir /s /q venv
        python -m venv venv
        echo Virtual environment created
    )
) else (
    python -m venv venv
    echo Virtual environment created
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo X Error: Failed to activate virtual environment
    pause
    exit /b 1
)
echo Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip -q
if errorlevel 1 (
    echo ! Warning: pip upgrade had issues, continuing...
)
echo pip upgraded
echo.

REM Install dependencies
if exist requirements.txt (
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo X Error: Failed to install dependencies
        pause
        exit /b 1
    )
    echo All dependencies installed
) else if exist requirements-minimal.txt (
    echo Installing minimal dependencies...
    pip install -r requirements-minimal.txt
    if errorlevel 1 (
        echo X Error: Failed to install dependencies
        pause
        exit /b 1
    )
    echo Minimal dependencies installed
) else (
    echo Installing core dependencies manually...
    pip install torch>=2.0.0 numpy>=1.21.0
    if errorlevel 1 (
        echo X Error: Failed to install dependencies
        pause
        exit /b 1
    )
    echo Core dependencies installed
)
echo.

REM Verify installation
echo Verifying installation...
python -c "import torch; import numpy as np; print(f'PyTorch {torch.__version__} installed'); print(f'NumPy {np.__version__} installed'); print(f'CUDA available: {torch.cuda.is_available()}')"
if errorlevel 1 (
    echo X Verification failed
    pause
    exit /b 1
)
echo.

REM Success message
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo To activate the virtual environment in the future, run:
echo   venv\Scripts\activate
echo.
echo To run the demo:
echo   python demo_pdnac_nc.py
echo.
echo To deactivate when done:
echo   deactivate
echo.
echo ==========================================
echo.
pause
