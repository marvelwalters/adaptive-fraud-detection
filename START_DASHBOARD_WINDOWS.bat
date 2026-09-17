@echo off
setlocal
cd /d "%~dp0"

echo ==============================================
echo Adaptive AI Fraud Detection Dashboard
echo ==============================================
echo.

echo [1/3] Checking Python...
py --version >nul 2>&1
if errorlevel 1 (
    python --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python was not found. Install Python and add it to PATH.
        pause
        exit /b 1
    )
    set "PYTHON=python"
) else (
    set "PYTHON=py"
)

echo [2/3] Installing/checking required packages...
%PYTHON% -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Package installation failed.
    pause
    exit /b 1
)

echo [3/3] Regenerating the deterministic experiment and starting Streamlit...
%PYTHON% train_model.py
if errorlevel 1 (
    echo ERROR: Model training failed.
    pause
    exit /b 1
)

%PYTHON% -m streamlit run app.py
if errorlevel 1 (
    echo ERROR: Streamlit could not start.
    pause
    exit /b 1
)

endlocal
