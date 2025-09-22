@echo off
echo Starting AHG-UBR5 Research Processor GUI...
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo ERROR: Virtual environment not found!
    echo Please run 'install.bat' first to set up the environment.
    pause
    exit /b 1
)

REM Activate virtual environment
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo Virtual environment activated!
echo.

REM Ensure data directories exist
if not exist "data\logs" mkdir data\logs
if not exist "data\vector_db\chroma_db" mkdir data\vector_db\chroma_db

python src/interfaces/gui_main.py
pause
