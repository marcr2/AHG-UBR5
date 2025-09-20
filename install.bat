@echo off
echo ========================================
echo   AHG-UBR5 - NEW INSTALLER
echo ========================================
echo.

echo Current directory: %CD%
echo.

echo Step 1: Check Python...
python --version
if errorlevel 1 (
    echo ERROR: Python not found
    pause
    exit /b 1
)

echo.
echo Step 2: Remove any existing .venv...
if exist ".venv" (
    echo Found existing .venv, removing...
    rmdir /s /q .venv
) else (
    echo No existing .venv found
)

echo.
echo Step 3: Create new .venv...
python -m venv .venv
if errorlevel 1 (
    echo ERROR: Failed to create .venv
    pause
    exit /b 1
)
echo .venv created successfully!

echo.
echo Step 4: Activate .venv...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate .venv
    pause
    exit /b 1
)
echo .venv activated successfully!

echo.
echo Step 5: Install requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)

echo.
echo ========================================
echo   INSTALLATION COMPLETED!
echo ========================================
pause
