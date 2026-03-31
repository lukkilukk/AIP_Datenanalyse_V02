@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=C:\Users\luk3b\anaconda3\envs\AIP_Assistant\python.exe"
set "DEFAULT_INPUT_DIR=%SCRIPT_DIR%analysedaten\eingang"

title AIP Datenanalyse
color 0B
mode con cols=96 lines=30 >nul 2>&1
cls

if not exist "%PYTHON_EXE%" (
  call :line
  echo   AIP DATENANALYSE
  call :line
  echo.
  echo   [FEHLER] Conda-Umgebung AIP_Assistant nicht gefunden.
  echo   Python-Pfad:
  echo   %PYTHON_EXE%
  echo.
  pause
  exit /b 1
)

if "%~1"=="/?" goto :usage
if /I "%~1"=="-h" goto :usage
if /I "%~1"=="--help" goto :usage

if "%~1"=="" (
  call :select_input_file
  if errorlevel 1 (
    cls
    call :line
    echo   AIP DATENANALYSE
    call :line
    echo.
    echo   [ABBRUCH] Keine Datei ausgewaehlt.
    echo.
    pause
    exit /b 1
  )
  set "FILE_DISPLAY=%SELECTED_FILE%"
) else (
  set "FILE_DISPLAY=%~1"
)

cls
call :line
echo   AIP DATENANALYSE
call :line
echo.
echo   Status    : Gestartet
echo   Datei     : %FILE_DISPLAY%
echo   Umgebung  : AIP_Assistant
echo   Hinweis   : Abbruch mit STRG+C
echo   Hinweis   : Fenster bleibt am Ende offen
echo.
call :line
echo.

if "%~1"=="" (
  "%PYTHON_EXE%" "%SCRIPT_DIR%run_analysis.py" "%SELECTED_FILE%"
) else (
  "%PYTHON_EXE%" "%SCRIPT_DIR%run_analysis.py" %*
)
set "EXITCODE=%ERRORLEVEL%"

echo.
call :line
if "%EXITCODE%"=="0" (
  echo   [ERFOLGREICH] Analyse erfolgreich abgeschlossen.
) else if "%EXITCODE%"=="130" (
  echo   [ABBRUCH] Analyse wurde vom Benutzer abgebrochen.
) else (
  echo   [FEHLER] Analyse ist mit Fehler beendet. Exit-Code: %EXITCODE%
)
call :line
echo.
pause
exit /b %EXITCODE%

:usage
cls
call :line
echo   AIP DATENANALYSE
call :line
echo.
echo   Startmoeglichkeiten
echo   1. Doppelklick auf run.cmd und Datei im Auswahlfenster waehlen
echo   2. In der Shell: .\run.cmd DATEI.txt
echo   3. Datei per Drag-and-drop auf run.cmd ziehen
echo.
echo   Standardordner fuer Messdateien
echo   analysedaten\eingang\
echo.
pause
exit /b 0

:select_input_file
set "SELECTED_FILE="
for /f "usebackq delims=" %%I in (`powershell.exe -NoProfile -ExecutionPolicy Bypass -STA -File "%SCRIPT_DIR%select_input_file.ps1" -InitialDirectory "%DEFAULT_INPUT_DIR%" -FallbackDirectory "%SCRIPT_DIR%"`) do set "SELECTED_FILE=%%I"
if not defined SELECTED_FILE exit /b 1
exit /b 0

:line
echo ================================================================================
exit /b 0
