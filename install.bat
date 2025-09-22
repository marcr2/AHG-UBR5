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
echo Step 6: Create necessary directories...
if not exist "data" mkdir data
if not exist "data\logs" mkdir data\logs
if not exist "data\vector_db" mkdir data\vector_db
if not exist "data\vector_db\chroma_db" mkdir data\vector_db\chroma_db
if not exist "data\scraped_papers" mkdir data\scraped_papers
if not exist "data\processed_papers" mkdir data\processed_papers
if not exist "data\exports" mkdir data\exports
echo Directories created successfully!

echo.
echo ========================================
echo   INSTALLATION COMPLETED!
echo ========================================
pause
