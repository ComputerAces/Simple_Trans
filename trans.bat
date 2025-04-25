@echo off
REM Subnet Core Game Launcher
REM Wackedout Out, games (c) 2025

echo Starting Subnet Core...
echo =============================================

REM Check if venv directory exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create virtual environment!
        echo Please ensure Python is installed and in your PATH.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
) else (
    echo Found existing virtual environment.
)

REM Create instance directory for SQLite database if it doesn't exist
if not exist instance (
    echo Creating instance directory for database...
    mkdir instance
    if errorlevel 1 (
        echo Failed to create instance directory!
        pause
        exit /b 1
    )
    echo Instance directory created successfully.
) else (
    echo Found existing instance directory.
)

REM Activate the virtual environment
echo Activating virtual environment...
call venv\Scripts\activate
if errorlevel 1 (
    echo Failed to activate virtual environment!
    pause
    exit /b 1
)

REM Check if requirements.txt exists and install requirements
if exist requirements.txt (
    echo Installing required packages...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install required packages!
        pause
        exit /b 1
    )
    echo Required packages installed successfully.
) else (
    echo Warning: requirements.txt not found!
    pause
)

REM Run the application and pass all arguments
echo Starting Simple_Trans...
python main.py %*

REM Keep the window open if there's an error
if errorlevel 1 (
    echo Application exited with an error!
    pause
    exit /b 1
)

REM Deactivate the virtual environment
call venv\Scripts\deactivate

exit /b 0