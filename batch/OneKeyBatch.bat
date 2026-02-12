@echo off

REM Check if uv is installed
python -m pip list | findstr uv >nul
if errorlevel 1 (
    echo Error: uv is not installed. Please install uv first using 'pip install uv'
    pause
    exit /b 1
)

REM Run the batch processor using uv
REM uv will use the project's virtual environment (.venv) if it exists
cd /D "%~dp0"
cd ..
uv run python batch\utils\batch_processor.py
pause
