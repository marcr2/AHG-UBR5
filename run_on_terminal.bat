@echo off
REM AHG-UBR5 Research Processor - Run Script
REM AI-Powered Scientific Hypothesis Generator for UBR5 Protein Research
REM Simple launcher for daily use (after installation)

echo.
echo ============================================================
echo    AHG-UBR5 RESEARCH PROCESSOR - STARTING UP
echo ============================================================
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo ERROR: Virtual environment not found!
    echo Please run 'install.bat' first to set up the environment.
    echo.
    echo The installer will:
    echo - Create virtual environment
    echo - Install dependencies
    echo - Set up data directories
    echo.
    pause
    exit /b 1
)

REM Check if main.py exists
if not exist "src\interfaces\main.py" (
    echo ERROR: main.py not found!
    echo Please ensure you're in the correct directory.
    pause
    exit /b 1
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    echo Try running 'install.bat' again to recreate the environment.
    pause
    exit /b 1
)

echo Virtual environment activated!

echo.
echo Checking data structure...

REM Ensure data directories exist (in case they were deleted)
if not exist "data" (
    echo Creating missing data directories...
    mkdir data
    mkdir data\embeddings
    mkdir data\embeddings\xrvix_embeddings
    mkdir data\embeddings\xrvix_embeddings\biorxiv
    mkdir data\embeddings\xrvix_embeddings\medrxiv
    mkdir data\embeddings\xrvix_embeddings\pubmed
    mkdir data\embeddings\xrvix_embeddings\ubr5_api
    mkdir data\logs
    mkdir data\scraped_data
    mkdir data\scraped_data\paperscraper_dumps
    mkdir data\vector_db
    mkdir data\vector_db\chroma_db
    mkdir data\backups
    echo Data directories created!
)

echo.
echo ============================================================
echo    LAUNCHING AHG-UBR5 RESEARCH PROCESSOR
echo ============================================================
echo.

REM Launch the GUI program
python src/interfaces/gui_main.py

REM Check if the program exited with an error
if errorlevel 1 (
    echo.
    echo ============================================================
    echo    PROGRAM EXITED WITH ERROR
    echo ============================================================
    echo.
    echo The program encountered an error. Please check the output above.
    echo.
    echo Common issues and solutions:
    echo - Missing API keys: Add your Google AI API key to keys.json
    echo - Network issues: Check your internet connection
    echo - Dependency issues: Try running 'install.bat' again
    echo - Permission issues: Try running as administrator
    echo.
    echo For help, check the README.md file or contact support.
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo    PROGRAM EXITED SUCCESSFULLY
echo ============================================================
echo.
echo Thank you for using AHG-UBR5 Research Processor!
echo.
pause