@echo off
REM AHG-UBR5 Research Processor - Installation Script
REM AI-Powered Scientific Hypothesis Generator for UBR5 Protein Research
REM Run this script ONCE to set up the environment

echo.
echo ============================================================
echo    AHG-UBR5 RESEARCH PROCESSOR - INSTALLATION
echo ============================================================
echo.
echo This script will set up your environment for the first time.
echo You only need to run this once.
echo.

REM Check if Python is installed
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    echo Download from: https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Display Python version
echo Python version:
python --version
echo.

REM Check if virtual environment already exists
if exist ".venv" (
    echo Virtual environment already exists.
    echo Do you want to recreate it? (y/n)
    set /p recreate=
    if /i "%recreate%"=="y" (
        echo Removing existing virtual environment...
        rmdir /s /q .venv
    ) else (
        echo Using existing virtual environment.
        goto :skip_venv
    )
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    echo Make sure Python is properly installed and you have write permissions
    pause
    exit /b 1
)
echo Virtual environment created successfully!

:skip_venv

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo Virtual environment activated!

echo.
echo Installing dependencies...

REM Check if requirements.txt exists
if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found
    echo Please ensure you're in the correct directory
    pause
    exit /b 1
)

REM Install dependencies
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    echo Please check your internet connection and try again
    echo.
    echo Common solutions:
    echo - Check your internet connection
    echo - Try running as administrator
    echo - Update pip: python -m pip install --upgrade pip
    pause
    exit /b 1
)

echo Dependencies installed successfully!

echo.
echo Setting up data structure...

REM Create data directory structure
if not exist "data" (
    echo Creating data directories...
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
    echo Data structure created!
) else (
    echo Data directories already exist.
)

echo.
echo Checking API configuration...

REM Check if keys.json exists
if not exist "keys.json" (
    echo WARNING: keys.json not found
    echo You may need to configure your API keys for full functionality
    echo.
    echo To add your Google AI API key, create a keys.json file with:
    echo {
    echo   "google_ai_api_key": "your_api_key_here"
    echo }
    echo.
    echo You can also configure this later when running the program.
) else (
    echo API configuration file found.
)

echo.
echo ============================================================
echo    INSTALLATION COMPLETED SUCCESSFULLY!
echo ============================================================
echo.
echo Your AHG-UBR5 Research Processor is now ready to use!
echo.
echo Next steps:
echo 1. Run 'run.bat' to start the program
echo 2. Configure your API keys in keys.json if needed
echo.
echo Installation files created:
echo - .venv\ (virtual environment)
echo - data\ (data directories)
echo - search_keywords_config.json (will be created when you use custom keywords)
echo.
echo You can now delete this install.bat file if you want.
echo.
pause
