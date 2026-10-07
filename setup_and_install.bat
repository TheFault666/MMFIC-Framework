@echo off
title MMFIC Trowler - Automated Setup and Run
echo =========================================================
echo   MMFIC Trowler: Automated Windows Installation ^& Run
echo =========================================================
echo.

:: 1. Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Python is NOT detected on this system.
    echo [*] Starting automated Python installation...
    echo.
    
    :: Define Python version and paths
    set "PYTHON_EXE_URL=https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe"
    set "INSTALLER_NAME=python_installer.exe"
    set "PYTHON_INSTALL_DIR=%USERPROFILE%\AppData\Local\Programs\Python\Python310"
    
    echo [*] Downloading Python 3.10.11 Installer via PowerShell...
    powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PYTHON_EXE_URL%' -OutFile '%INSTALLER_NAME%'"
    
    if not exist "%INSTALLER_NAME%" (
        echo [ERROR] Failed to download Python installer. Please check your internet connection.
        pause
        exit /b
    )
    
    echo [*] Installing Python silently in user space ^(optimal folder^)...
    echo     ^(This may take 1-2 minutes. Please wait...^)
    REM Run installer silently: user space only (no admin needed), add to path
    start /wait "" %INSTALLER_NAME% /quiet InstallAllUsers=0 PrependPath=1 Include_test=0 Include_doc=0 AssociateFiles=1
    
    :: Clean up installer
    del %INSTALLER_NAME%
    
    :: Update temporary path variables for the current command prompt session
    set "PATH=%PYTHON_INSTALL_DIR%;%PYTHON_INSTALL_DIR%\Scripts;%PATH%"
    
    :: Re-verify installation
    python --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo.
        echo [!] Python installation complete. 
        echo [!] Please restart this command prompt window and run this script again to refresh environment paths.
        pause
        exit /b
    ) else (
        echo [+] Python installed successfully and environment paths updated!
        echo.
    )
) else (
    echo [+] Python detected:
    python --version
    echo.
)

:: 2. Create Virtual Environment
if not exist "venv" (
    echo [*] Creating Python Virtual Environment ^(venv^)...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b
    )
)

:: 3. Activate Virtual Environment and Install dependencies
echo [*] Activating virtual environment...
call venv\Scripts\activate

echo [*] Upgrading pip...
python -m pip install --upgrade pip >nul

echo [*] Installing required modules (scikit-learn, numpy, stem, etc.)...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Installation of python modules failed.
    pause
    exit /b
)

:: 4. Install Playwright Browser Binaries
echo [*] Downloading Playwright browser engine ^(Chromium^)...
playwright install chromium

:: 5. Handle API Key
if not exist ".env" (
    echo.
    echo ---------------------------------------------------------
    echo   Gemini API Key Configuration
    echo ---------------------------------------------------------
    echo   This framework uses Gemini to discover forum layouts.
    echo   ^(Optional: Press Enter to skip if using Offline mode^)
    echo.
    set /p API_KEY="  Enter your Gemini API Key: "
    if not "%API_KEY%"=="" (
        echo GEMINI_API_KEY=%API_KEY% > .env
        echo [+] API Key saved to .env
    ) else (
        echo [!] Running without API Key ^(Offline / Local Mode Only^).
    )
    echo.
)

:: 6. Launch Application
echo.
echo =========================================================
echo   Select Launch Mode:
echo =========================================================
echo   1. Launch CLI (Interactive Terminal)
echo   2. Launch Web Dashboard (Streamlit/Flask UI)
echo =========================================================
set /p MODE="Select mode (1 or 2): "

if "%MODE%"=="2" (
    echo [*] Launching MMFIC Trowler Web Dashboard...
    python dashboard.py
) else (
    echo [*] Launching MMFIC Trowler CLI...
    python main.py
)

pause